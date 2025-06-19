# TF2C.py

import os
import re
import shutil
import subprocess
import numpy as np
from types import SimpleNamespace
from SI_Toolkit.Functions.General.Initialization import get_net
from SI_Toolkit.Functions.General.TerminalContentManager import TerminalContentManager

from SI_Toolkit.Functions.General.NumpyNetworks import NumpyGRUNetwork, NumpyLSTMNetwork

# ---------------------------------------------------------------------------
# Generic helpers for variable‑depth networks
# ---------------------------------------------------------------------------
def _save_float_array(path, name, arr, transpose=False):
    """Append a 1‑D C float array to <path>/network_parameters.c"""
    if transpose:
        arr = arr.T
    with open(os.path.join(path, "network_parameters.c"), "a") as f:
        f.write(f"const float {name}[] = {{\n")
        f.write(", ".join(map(str, arr.flatten())))
        f.write("\n};\n")

# Set a seed for reproducibility
seed_value = 1
np.random.seed(seed_value)


def tf2C(path_to_models, net_name, batch_size):

    with (TerminalContentManager(os.path.join(path_to_models, net_name, "C_implementation", "terminal_output.txt"))):
        # Import network
        a = SimpleNamespace()
        a.path_to_models = path_to_models
        a.net_name = net_name
        net, net_info = get_net(
            a, time_series_length=1, batch_size=batch_size, stateful=True, remove_redundant_dimensions=True
        )

        # ------------------------------------------------
        # Detect whether this model is GRU-, LSTM- or Dense-based
        # ------------------------------------------------

        first_layer_type = type(net.layers[0]).__name__
        is_gru_model = ('GRU' in first_layer_type)
        is_lstm_model = ('LSTM' in first_layer_type)

        # Define input
        input_length = len(net_info.inputs)

        if is_gru_model or is_lstm_model:
            input_data = np.random.rand(1, 1, input_length).astype(np.float32)
        else:
            input_data = np.random.rand(1, input_length).astype(np.float32)
        inputs_string = ", ".join(map(str, input_data.flatten()))
        print(f"\nInput data: {inputs_string} \n")

        # +++ If GRU/LSTM, also initialize the hidden (and cell) states to random values +++
        if is_gru_model or is_lstm_model:
            # Two-layer GRU => net.layers = [GRU1, GRU2, Dense_output]
            # The reset_states() method expects shape = (batch_size, units)
            rnn1_units = net.layers[0].output_shape[-1]
            rnn2_units = net.layers[1].output_shape[-1]

            # Create random initial states (just once, shape=(1, units))
            h1_init = np.random.rand(1, rnn1_units).astype(np.float32)
            h2_init = np.random.rand(1, rnn2_units).astype(np.float32)

            # Force the TF GRUs/LSTMs to start from these states
            if is_gru_model:
                net.layers[0].reset_states(h1_init)
                net.layers[1].reset_states(h2_init)
            else:                                               # LSTM case
                c1_init = np.random.rand(1, rnn1_units).astype(np.float32)
                c2_init = np.random.rand(1, rnn2_units).astype(np.float32)
                net.layers[0].reset_states([h1_init, c1_init])
                net.layers[1].reset_states([h2_init, c2_init])

        # Now, check Python output to compare with C output
        output_tf = net(input_data).numpy().flatten()
        print("Python (TensorFlow) output:")
        print(output_tf)
        print("\n")

        # ------------------------------------------------
        # Compare with NumPy-based replica if it's a GRU/LSTM model
        # ------------------------------------------------
        if is_gru_model:
            print("Testing NumPy-based GRU implementation...")

            # 1) Build the NumPy-based network replica
            numpy_net = NumpyGRUNetwork(net)

            # 2) Evaluate with same initial states and same input
            np_output = numpy_net.forward(
                input_data,
                h_inits=[h1_init, h2_init]  # match the states we set in TF
            ).flatten()
        elif is_lstm_model:
            print("Testing NumPy-based LSTM implementation...")
            numpy_net = NumpyLSTMNetwork(net)
            np_output = numpy_net.forward(
                input_data,
                h_inits=[h1_init, h2_init],
                c_inits=[c1_init, c2_init]
            ).flatten()
        else:
            numpy_net = None
            np_output = None

        if numpy_net is not None:
            difference = np.max(np.abs(output_tf - np_output))
            print(f"NumPy replica output: {np_output}")
            print(f"Max absolute difference (TF vs NumPy) = {difference:.6g}")
            if difference < 1e-6:
                print("SUCCESS: Outputs match closely!\n")
            else:
                print("WARNING: Outputs differ. Check gate order/biases.\n")

        # Define the target directory
        target_directory = os.path.join(path_to_models, net_name, 'C_implementation')

        # Check if target directory exists, if not, create it
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Path to the main.c and network.c files (assumed to be in the same folder as this script)
        source_main_c = os.path.join(os.path.dirname(__file__), 'main.c')
        source_network_c = os.path.join(os.path.dirname(__file__), 'network.c')
        header_network_h = os.path.join(os.path.dirname(__file__), 'network.h')

        # Copy the necessary C files to the target directory
        shutil.copy(source_main_c, target_directory)
        shutil.copy(source_network_c, target_directory)

        # Read network.h so we can update the macros
        with open(header_network_h, 'r') as f:
            network_h_content = f.read()

        # ------------------------------------------------
        # If it is a GRU model (two GRU layers + linear output)
        # ------------------------------------------------
        if is_gru_model:
            print("Detected GRU-based model. Preparing code accordingly...\n")

            # Set macro indicating a GRU model
            network_h_content = network_h_content.replace(
                "#define INPUT_SIZE      // Overwritten by python",
                f"#define IS_GRU 1\n#define IS_LSTM 0\n#define INPUT_SIZE {net.input_shape[-1]}      // Overwritten by python"
            )

            # For a 2-layer GRU, net.layers = [GRU1, GRU2, Dense_output]
            #   net.layers[0]: GRU1
            #   net.layers[1]: GRU2
            #   net.layers[2]: final Dense
            # Each GRU has weights: [kernel, recurrent_kernel, bias].
            # The final Dense has [weights, bias].

            # Extract shapes
            gru1_units = net.layers[0].output_shape[-1]
            gru2_units = net.layers[1].output_shape[-1]
            output_size = net.layers[2].output_shape[-1]

            # Update #define placeholders in network.h
            network_h_content = re.sub(
                r"^#\s*define\s+GRU1_UNITS\b.*$",
                f"#define GRU1_UNITS {gru1_units}",
                network_h_content,
                flags=re.M
            )
            network_h_content = re.sub(
                r"^#\s*define\s+GRU2_UNITS\b.*$",
                f"#define GRU2_UNITS {gru2_units}",
                network_h_content,
                flags=re.M
            )
            network_h_content = network_h_content.replace(
                "#define LAYER3_SIZE     // Overwritten by python (used in Dense mode)",
                f"#define LAYER3_SIZE {output_size}     // Overwritten by python (used in Dense mode)"
            )

            # Write the updated network.h
            with open(os.path.join(target_directory, 'network.h'), 'w') as f:
                f.write(network_h_content)

            # Retrieve GRU1 weights
            gru1_kernel, gru1_recurrent_kernel, gru1_bias = net.layers[0].get_weights()
            # Retrieve GRU2 weights
            gru2_kernel, gru2_recurrent_kernel, gru2_bias = net.layers[1].get_weights()
            # Retrieve final Dense
            dense_weights, dense_bias = net.layers[2].get_weights()

            # +++ IMPORTANT +++
            # Do NOT sum the biases anymore. For reset_after=True, we need both halves.
            # Merge bias subarrays if Keras stored them as shape (2, 3*units)
            if len(gru1_bias.shape) == 2:
                gru1_bias = gru1_bias.reshape(-1)
            if len(gru2_bias.shape) == 2:
                gru2_bias = gru2_bias.reshape(-1)

            # Save into network_parameters.c
            def save_as_cc_combined(array, varname):
                with open(os.path.join(target_directory, "network_parameters.c"), "a") as f:
                    f.write(f"const float {varname}[] = {{\n")
                    f.write(", ".join(map(str, array.flatten())))
                    f.write("\n};\n")

            param_c_path = os.path.join(target_directory, "network_parameters.c")
            if os.path.exists(param_c_path):
                os.remove(param_c_path)

            # +++ Save initial states so that C code can see them +++
            save_as_cc_combined(h1_init.flatten(), "initial_h1")
            save_as_cc_combined(h2_init.flatten(), "initial_h2")

            # Save GRU1
            save_as_cc_combined(gru1_kernel,              "gru1_kernel")
            save_as_cc_combined(gru1_recurrent_kernel,    "gru1_recurrent_kernel")
            save_as_cc_combined(gru1_bias,                "gru1_bias")
            # Save GRU2
            save_as_cc_combined(gru2_kernel,              "gru2_kernel")
            save_as_cc_combined(gru2_recurrent_kernel,    "gru2_recurrent_kernel")
            save_as_cc_combined(gru2_bias,                "gru2_bias")

            # Save final Dense (linear) layer.  Transpose for matMul convenience.
            save_as_cc_combined(dense_weights.T, "weights3")
            save_as_cc_combined(dense_bias,      "bias3")

        # ------------------------------------------------
        # If it is a LSTM model (two LSTM layers + linear output)
        # ------------------------------------------------
        elif is_lstm_model:
            print("Detected LSTM-based model. Preparing code accordingly...\n")

            network_h_content = network_h_content.replace(
                "#define INPUT_SIZE      // Overwritten by python",
                f"#define IS_GRU 0\n#define IS_LSTM 1\n#define INPUT_SIZE {net.input_shape[-1]}      // Overwritten by python"
            )

            lstm1_units = net.layers[0].output_shape[-1]
            lstm2_units = net.layers[1].output_shape[-1]
            output_size = net.layers[2].output_shape[-1]

            network_h_content = re.sub(
                r"^#\s*define\s+LSTM1_UNITS\b.*$",
                f"#define LSTM1_UNITS {lstm1_units}",
                network_h_content,
                flags=re.M,
            )

            network_h_content = re.sub(
                r"^#\s*define\s+LSTM2_UNITS\b.*$",
                f"#define LSTM2_UNITS {lstm2_units}",
                network_h_content,
                flags=re.M,
            )

            network_h_content = network_h_content.replace(
                "#define LAYER3_SIZE     // Overwritten by python (used in Dense mode)",
                f"#define LAYER3_SIZE {output_size}     // Overwritten by python (used in Dense mode)"
            )

            # Write the updated network.h
            with open(os.path.join(target_directory, 'network.h'), 'w') as f:
                f.write(network_h_content)

            lstm1_kernel, lstm1_recurrent, lstm1_bias = net.layers[0].get_weights()
            lstm2_kernel, lstm2_recurrent, lstm2_bias = net.layers[1].get_weights()
            dense_weights, dense_bias                 = net.layers[2].get_weights()

            def _merge_lstm_bias(b, units):
                # Case A: already stored as two 4U bias vectors
                if b.ndim == 2 and b.shape[1] == 4 * units:
                    return b.sum(axis=0)
                # Case B: cuDNN format => flat 8U vector
                if b.ndim == 1 and b.size == 8 * units:
                    return b.reshape(2, 4 * units).sum(axis=0)
                # Otherwise assume it’s already a flat 4U vector
                return b

            lstm1_bias = _merge_lstm_bias(lstm1_bias, lstm1_units)
            lstm2_bias = _merge_lstm_bias(lstm2_bias, lstm2_units)

            def save(arr, name):
                with open(os.path.join(target_directory, "network_parameters.c"), "a") as f:
                    f.write(f"const float {name}[] = {{\n")
                    f.write(", ".join(map(str, arr.flatten())))
                    f.write("\n};\n")

            param_c_path = os.path.join(target_directory, "network_parameters.c")
            if os.path.exists(param_c_path):
                os.remove(param_c_path)

            # Save initial states
            save(h1_init, "initial_h1"); save(c1_init, "initial_c1")
            save(h2_init, "initial_h2"); save(c2_init, "initial_c2")

            # Save weights
            save(lstm1_kernel,    "lstm1_kernel")
            save(lstm1_recurrent, "lstm1_recurrent_kernel")
            save(lstm1_bias,      "lstm1_bias")
            save(lstm2_kernel,    "lstm2_kernel")
            save(lstm2_recurrent, "lstm2_recurrent_kernel")
            save(lstm2_bias,      "lstm2_bias")
            save(dense_weights.T, "weights3")
            save(dense_bias,      "bias3")


        # ------------------------------------------------
        # Otherwise, the original Dense-based approach
        # ------------------------------------------------
        else:
            print("Detected Dense-based model. Using existing Dense conversion...\n")

            # Insert #define IS_GRU 0 and IS_LSTM 0 on top
            network_h_content = network_h_content.replace(
                "#define INPUT_SIZE      // Overwritten by python",
                f"#define IS_GRU 0\n#define IS_LSTM 0\n#define INPUT_SIZE {net.input_shape[1]}      // Overwritten by python"
            )

            # The rest is the original code which sets LAYER1_SIZE, LAYER2_SIZE, LAYER3_SIZE
            layer1_size = net.layers[0].output_shape[1]
            layer2_size = net.layers[2].output_shape[1]
            layer3_size = net.layers[4].output_shape[1]

            network_h_content = network_h_content.replace(
                "#define LAYER1_SIZE     // Overwritten by python (used in Dense mode)",
                f"#define LAYER1_SIZE {layer1_size}     // Overwritten by python (used in Dense mode)"
            )
            network_h_content = network_h_content.replace(
                "#define LAYER2_SIZE     // Overwritten by python (used in Dense mode)",
                f"#define LAYER2_SIZE {layer2_size}     // Overwritten by python (used in Dense mode)"
            )
            network_h_content = network_h_content.replace(
                "#define LAYER3_SIZE     // Overwritten by python (used in Dense mode)",
                f"#define LAYER3_SIZE {layer3_size}     // Overwritten by python (used in Dense mode)"
            )

            # Write the modified network.h to the target directory
            with open(os.path.join(target_directory, 'network.h'), 'w') as f:
                f.write(network_h_content)

            # Convert the Keras model to C (weights and biases)
            weights1, bias1 = net.layers[0].get_weights()  # First Dense layer
            weights2, bias2 = net.layers[2].get_weights()  # Second Dense layer
            weights3, bias3 = net.layers[4].get_weights()  # Third Dense layer (if applicable)

            # Combine all weights and biases into one .c file
            def save_as_cc_combined(array, varname, transpose=False):
                if transpose:
                    array = array.T  # Transpose the array for compatibility with C matrix multiplication
                with open(os.path.join(target_directory, "network_parameters.c"), "a") as f:
                    f.write(f"const float {varname}[] = {{\n")
                    f.write(", ".join(map(str, array.flatten())))
                    f.write("\n};\n")

            # Remove existing network_parameters.c file if it exists
            param_c_path = os.path.join(target_directory, "network_parameters.c")
            if os.path.exists(param_c_path):
                os.remove(param_c_path)

            # Save all weights and biases to the same .c file
            save_as_cc_combined(weights1, "weights1", transpose=True)
            save_as_cc_combined(bias1,    "bias1")
            save_as_cc_combined(weights2, "weights2", transpose=True)
            save_as_cc_combined(bias2,    "bias2")
            save_as_cc_combined(weights3, "weights3", transpose=True)
            save_as_cc_combined(bias3,    "bias3")

        # Prepare input data and pass it as arguments to the C program
        input_str = " ".join(map(str, input_data.flatten()))

        # Compile the C program
        def compile_c_code():
            result = subprocess.run(
                ['gcc', '-std=c99', '-O3', os.path.join(target_directory, 'main.c'), '-o', os.path.join(target_directory, 'network_test'),
                 '-lm'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Compilation failed!")
                print(result.stderr)

        compile_c_code()

        bold_start = "\033[1m"
        bold_end = "\033[0m"
        print(f"{bold_start}C program output:{bold_end} \n")
        # Run the compiled C program with input passed from Python
        def run_c_code():
            result = subprocess.run([os.path.join(target_directory, 'network_test')] + input_str.split(),
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print("Execution failed!")
                print(result.stderr)
            else:
                print(result.stdout)

        run_c_code()

        print(f"\n{bold_start}Timing of Python (TensorFlow) network for comparison:{bold_end} \n")

        import timeit
        import tensorflow as tf

        # 1. Timing without XLA compilation
        def run_without_compilation():
            output_tf = net(input_data)

        # Time executions without TF compilation
        nr_runs_without_compilation = 1000
        time_without_compilation = timeit.timeit(run_without_compilation, number=nr_runs_without_compilation, setup=lambda: run_without_compilation())
        print(f"Total time for {nr_runs_without_compilation} runs without TF compilation: {time_without_compilation:.3f} seconds")
        print(f"Average time per call: {time_without_compilation*1.0e6/nr_runs_without_compilation:.2f} us \n")

        # 2. Enabling TF Compilation
        @tf.function  # Enable XLA compilation with the decorator
        def run_with_compilation():
            return net(input_data)

        # Time executions with TF compilation
        nr_runs_with_compilation = 10000
        time_with_compilation = timeit.timeit(lambda: run_with_compilation(), number=nr_runs_with_compilation, setup=lambda: run_with_compilation())
        print(f"Total time for {nr_runs_with_compilation} runs with TF compilation: {time_with_compilation:.3f} seconds")
        print(f"Average time per call: {time_with_compilation*1.0e6/nr_runs_with_compilation:.2f} us")

        # Clean up: Delete the copied C files after compilation and execution
        os.remove(os.path.join(target_directory, 'main.c'))
        os.remove(os.path.join(target_directory, 'network_test'))

        print("Temporary C files have been deleted.")
