# neural_network_evaluator.py

from types import SimpleNamespace

import numpy as np

# Imports for C backend
import ctypes
import subprocess
import sys
from pathlib import Path
from contextlib import suppress

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.value_precision import set_value_precision
from SI_Toolkit.Compile import CompileAdaptive
from SI_Toolkit.computation_library import NumpyLibrary, ComputationLibrary


class neural_network_evaluator:
    _computation_library = NumpyLibrary()
    _hls4ml_cache = {}  # Cache for hls4ml converted networks

    def __init__(self, net_name, path_to_models, batch_size, input_precision='float', nn_evaluator_mode='normal'):

        self.net_name = net_name
        self.path_to_models = path_to_models
        self.batch_size = batch_size
        self.input_precision = input_precision
        self.nn_evaluator_mode = nn_evaluator_mode

        a = SimpleNamespace()

        a.path_to_models = path_to_models
        a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        if self.nn_evaluator_mode == 'hls4ml':
            self._computation_library = NumpyLibrary()
            
            # Create a cache key based on network parameters
            cache_key = f"{net_name}_{path_to_models}_{batch_size}_{input_precision}"
            
            # Check if we already have a converted network in cache
            if cache_key in self._hls4ml_cache:
                print(f"Using cached hls4ml network for {net_name}")
                self.net = self._hls4ml_cache[cache_key]
            else:
                # Convert network to HLS form using temporary directory to avoid file system issues
                from SI_Toolkit.HLS4ML.hls4ml_functions import convert_model_with_hls4ml
                print(f"Converting network {net_name} to hls4ml format (this may take a while)...")
                self.net, _ = convert_model_with_hls4ml(self.net, use_temp_dir=True)
                self.net.compile()
                
                # Cache the converted network
                self._hls4ml_cache[cache_key] = self.net
                print(f"Cached hls4ml network for {net_name}")
        elif self.nn_evaluator_mode == 'C':
            if batch_size is not None and batch_size > 1:
                raise ValueError("C implementation does not support batch size > 1")
            self._computation_library = NumpyLibrary()
            self._setup_c_backend()
        elif self.net_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            from SI_Toolkit.Functions.Pytorch.Network import get_device
            self._computation_library = PyTorchLibrary()
            self.device = get_device()
            self.net.reset()
            self.net.eval()
        elif self.net_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self._computation_library = TensorFlowLibrary()

        self.normalization_info = get_norm_info_for_net(self.net_info)
        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.net_info.outputs,
                                                                self.lib)

        if batch_size is not None and batch_size > 1:
            self.net_input_normed = self.lib.to_variable(
                np.zeros([batch_size, len(self.net_info.inputs)], dtype=np.float32), self.lib.float32)
        else:
            self.net_input_normed = self.lib.to_variable(
                np.zeros([len(self.net_info.inputs), ], dtype=np.float32), self.lib.float32)

        self.step_compilable = CompileAdaptive(self._step_compilable)

    @classmethod
    def clear_hls4ml_cache(cls):
        """Clear the hls4ml conversion cache. Useful for memory management or when networks change."""
        cls._hls4ml_cache.clear()
        print("hls4ml cache cleared")

    @classmethod
    def get_cache_info(cls):
        """Get information about the current hls4ml cache."""
        return {
            'cache_size': len(cls._hls4ml_cache),
            'cached_networks': list(cls._hls4ml_cache.keys())
        }


    def step(self, net_input):

        net_input = self.lib.to_tensor(net_input, self.lib.float32)

        if self.lib.lib == 'Pytorch':
            net_input = net_input.to(self.device)

        net_output = self.step_compilable(net_input)

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        if self.lib.ndim(net_output) == 1:
            net_output = net_output[self.lib.newaxis, self.lib.newaxis, :]
        else:
            net_output =  net_output

        return net_output

    def _step_compilable(self, net_input):

        self.lib.assign(self.net_input_normed, self.normalize_inputs(net_input))

        net_input = self.lib.reshape(self.net_input_normed, (-1, 1, len(self.net_info.inputs)))
        net_input = set_value_precision(net_input, self.input_precision, lib=self.lib)

        if self.nn_evaluator_mode == 'hls4ml':  # Covers just the case for hls4ml, when the model is hls model
            net_output = self.net.predict(net_input)
        elif self.nn_evaluator_mode == 'C':  # Covers the case for C implementation

            # --- copy into C buffer without allocations -----------------------
            np.copyto(self._c_in_np, net_input.astype(np.float32, copy=False))

            # --- C forward pass (releases GIL) --------------------------------
            self._c_eval(self._c_in_arr, self._c_out_arr)

            # --- NumPy view -------------------------------------
            net_output = np.array(self._c_out_np)
        else:
            net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return net_output

    @property
    def computation_library(self) -> "type[ComputationLibrary]":
        if self._computation_library is None:
            raise NotImplementedError("Controller class needs to specify its computation library")
        return self._computation_library

    @property
    def lib(self) -> "type[ComputationLibrary]":
        """Shortcut to make easy using functions from computation library, this is also used by CompileAdaptive to recognize library"""
        return self.computation_library

    def compose_input(self, inputs_dict):

        net_input = [inputs_dict[key] for key in self.net_info.inputs]
        net_input = [self.lib.to_tensor(inp, self.lib.float32) for inp in net_input]
        net_input = self.lib.stack(net_input, axis=-1)

        return net_input


    def _setup_c_backend(self):
        """
        Locate C_implementation/, compile it into a shared object if necessary
        and create ctypes function handles + reusable buffers.
        """
        c_dir = Path(self.path_to_models) / self.net_name / 'C_implementation'
        if not c_dir.exists():
            raise FileNotFoundError(f'C implementation missing: {c_dir}')

        # --- choose platform‑specific filenames ----------------------------
        ext = { 'linux': '.so', 'darwin': '.dylib', 'win32': '.dll' }[sys.platform]
        lib_name = f'libnetwork{ext}'
        lib_path = c_dir / lib_name

        # --- build if not yet present --------------------------------------
        if not lib_path.exists():
            cmd = [
                'gcc', '-O3', '-fPIC', '-shared',
                'network.c',
                '-lm', '-o', lib_name
            ]
            # Windows: assume mingw‑w64 in PATH; otherwise adapt accordingly.
            subprocess.check_call(cmd, cwd=c_dir)

        # --- load the library ----------------------------------------------
        self._c_lib = ctypes.CDLL(str(lib_path))

        # --- set up signatures --------------------------------------------
        self._c_eval = self._c_lib.C_Network_Evaluate
        self._c_eval.argtypes = (
            ctypes.POINTER(ctypes.c_float),  # inputs
            ctypes.POINTER(ctypes.c_float),  # outputs
        )
        self._c_eval.restype = None

        # optional initialisers (GRU/LSTM) – call if they exist
        with suppress(AttributeError):
            self._c_lib.InitializeGRUStates()
        with suppress(AttributeError):
            self._c_lib.InitializeLSTMStates()

        # --- reusable C buffers -------------------------------------------
        self._n_in = len(self.net_info.inputs)
        self._n_out = len(self.net_info.outputs)

        self._c_in_arr = (ctypes.c_float * self._n_in)()
        self._c_out_arr = (ctypes.c_float * self._n_out)()

        # quick NumPy views (no copy) for each buffer
        self._c_in_np = np.ctypeslib.as_array(self._c_in_arr)
        self._c_out_np = np.ctypeslib.as_array(self._c_out_arr)
        print(f'Using C backend for {self.net_info.net_full_name}')


