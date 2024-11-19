# TF2C.py
import os
import shutil
import subprocess
import numpy as np
from types import SimpleNamespace
from SI_Toolkit.Functions.General.Initialization import get_net
from SI_Toolkit.Functions.General.TerminalContentManager import TerminalContentManager


# Set a seed for reproducibility
seed_value = 1
np.random.seed(seed_value)


def tf2C(path_to_models, net_name, batch_size):

    with TerminalContentManager(os.path.join(path_to_models, net_name, "C_implementation", "terminal_output.txt")):
        # Import network
        a = SimpleNamespace()
        a.path_to_models = path_to_models
        a.net_name = net_name
        batch_size = batch_size
        net, net_info = get_net(
            a, time_series_length=1, batch_size=batch_size, stateful=True, remove_redundant_dimensions=True
        )

        # Define input
        input_length = len(net_info.inputs)
        input_data = np.random.rand(1, input_length).astype(np.float32)
        inputs_string = ", ".join(map(str, input_data.flatten()))
        print(f"\nInput data: {inputs_string} \n")

        # Now, check Python output to compare with C output
        output_tf = net(input_data).numpy().flatten()
        print("Python (TensorFlow) output:")
        print(output_tf)
        print("\n")

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

        # Modify network.c to update the layer sizes dynamically based on the TensorFlow model
        input_size = net.input_shape[1]  # Input size from the model
        layer1_size = net.layers[0].output_shape[1]  # First Dense layer size
        layer2_size = net.layers[2].output_shape[1]  # Second Dense layer size
        layer3_size = net.layers[4].output_shape[1]  # Third Dense layer size

        with open(header_network_h, 'r') as f:
            network_h_content = f.read()

        # Update the #define values for input and layer sizes
        network_h_content = network_h_content.replace("#define INPUT_SIZE", f"#define INPUT_SIZE {input_size}")
        network_h_content = network_h_content.replace("#define LAYER1_SIZE", f"#define LAYER1_SIZE {layer1_size}")
        network_h_content = network_h_content.replace("#define LAYER2_SIZE", f"#define LAYER2_SIZE {layer2_size}")
        network_h_content = network_h_content.replace("#define LAYER3_SIZE", f"#define LAYER3_SIZE {layer3_size}")

        # Write the modified network.c to the target directory
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
        if os.path.exists(os.path.join(target_directory, "network_parameters.c")):
            os.remove(os.path.join(target_directory, "network_parameters.c"))

        # Save all weights and biases to the same .c file
        save_as_cc_combined(weights1, "weights1", transpose=True)
        save_as_cc_combined(bias1, "bias1")
        save_as_cc_combined(weights2, "weights2", transpose=True)
        save_as_cc_combined(bias2, "bias2")
        save_as_cc_combined(weights3, "weights3", transpose=True)
        save_as_cc_combined(bias3, "bias3")

        # Prepare input data and pass it as arguments to the C program
        input_str = " ".join(map(str, input_data.flatten()))

        # Compile the C program
        def compile_c_code():
            result = subprocess.run(
                ['gcc', os.path.join(target_directory, 'main.c'), '-o', os.path.join(target_directory, 'network_test'),
                 '-lm'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Compilation failed!")
                print(result.stderr)
            else:
                pass
                # print("Compilation successful! \n\n")

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

