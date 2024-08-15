from types import SimpleNamespace

import numpy as np

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.value_precision import set_value_precision
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive
from SI_Toolkit.computation_library import NumpyLibrary, ComputationLibrary


class neural_network_evaluator:
    _computation_library = NumpyLibrary

    def __init__(self, net_name, path_to_models, batch_size, input_precision='float', hls4ml=False):

        self.net_name = net_name
        self.path_to_models = path_to_models
        self.batch_size = batch_size
        self.input_precision = input_precision
        self.hls4ml = hls4ml

        a = SimpleNamespace()

        a.path_to_models = path_to_models
        a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        if self.hls4ml:
            self._computation_library = NumpyLibrary
            # Convert network to HLS form
            from SI_Toolkit.HLS4ML.hls4ml_functions import convert_model_with_hls4ml
            self.net, _ = convert_model_with_hls4ml(self.net)
            self.net.compile()
        elif self.net_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            self._computation_library = PyTorchLibrary
        elif self.net_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self._computation_library = TensorFlowLibrary

        if self.lib.lib == 'Pytorch':
            from SI_Toolkit.Functions.Pytorch.Network import get_device
            self.device = get_device()
            self.net.reset()
            self.net.eval()

        self.normalization_info = get_norm_info_for_net(self.net_info)
        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.net_info.outputs,
                                                                self.lib)

        self.net_input_normed = self.lib.to_variable(
            np.zeros([len(self.net_info.inputs), ], dtype=np.float32), self.lib.float32)

        self.step_compilable = CompileAdaptive(self._step_compilable)


    def step(self, net_input):

        net_input = self.lib.to_tensor(net_input, self.lib.float32)

        if self.lib.lib == 'Pytorch':
            net_input = net_input.to(self.device)

        net_output = self.step_compilable(net_input)

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        if self.lib.ndim(net_output) == 1:
            return net_output[self.lib.newaxis, self.lib.newaxis, :]
        else:
            return net_output

    def _step_compilable(self, net_input):

        self.lib.assign(self.net_input_normed, self.normalize_inputs(net_input))

        net_input = self.lib.reshape(self.net_input_normed, (-1, 1, len(self.net_info.inputs)))
        net_input = set_value_precision(net_input, self.input_precision, lib=self.lib)

        if self.lib.lib == 'Numpy':  # Covers just the case for hls4ml, when the model is hls model
            net_output = self.net.predict(net_input)
        else:
            net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return net_output

    @property
    def computation_library(self) -> "type[ComputationLibrary]":
        if self._computation_library == None:
            raise NotImplementedError("Controller class needs to specify its computation library")
        return self._computation_library

    @property
    def lib(self) -> "type[ComputationLibrary]":
        """Shortcut to make easy using functions from computation library, this is also used by CompileAdaptive to recognize library"""
        return self.computation_library
