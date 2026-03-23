"""
VHDL package generation utilities for hls4ml conversion.

This module provides functions to generate and update VHDL package files
with neural network parameters.
"""

from pathlib import Path


def generate_mlp_top_pkg_content(input_neurons, output_neurons, input_data_bits, output_data_bits):
    """
    Generate the content for mlp_top_pkg.vhd file.
    
    Args:
        input_neurons (int): Number of input neurons
        output_neurons (int): Number of output neurons
        input_data_bits (int): Number of bits per input data
        output_data_bits (int): Number of bits per output data
        
    Returns:
        str: VHDL package content
    """
    return f"""package mlp_top_pkg is
    constant MLP_INPUT_NEURONS    : integer := {input_neurons};
    constant MLP_INPUT_DATA_BITS  : integer := {input_data_bits};
    constant MLP_INPUT_BIT_WIDTH  : integer := MLP_INPUT_NEURONS * MLP_INPUT_DATA_BITS;

    constant MLP_OUTPUT_NEURONS   : integer := {output_neurons};
    constant MLP_OUTPUT_DATA_BITS : integer := {output_data_bits};
    constant MLP_OUTPUT_BIT_WIDTH : integer := MLP_OUTPUT_NEURONS * MLP_OUTPUT_DATA_BITS;
end package mlp_top_pkg;
"""


def copy_and_update_mlp_pkg(source_mlp_pkg_path, target_vhdl_dir, input_neurons, output_neurons, input_data_bits, output_data_bits):
    """
    Copy mlp_top_pkg.vhd to vhdl folder and update it with network parameters.
    
    Args:
        source_mlp_pkg_path (Path or None): Path to source mlp_top_pkg.vhd (optional)
        target_vhdl_dir (Path): Target VHDL directory
        input_neurons (int): Number of input neurons
        output_neurons (int): Number of output neurons
        input_data_bits (int): Number of bits per input data
        output_data_bits (int): Number of bits per output data
    """
    # Ensure target directory exists
    target_vhdl_dir.mkdir(parents=True, exist_ok=True)
    target_mlp_pkg = target_vhdl_dir / "mlp_top_pkg.vhd"
    
    # Generate the updated package content
    new_content = generate_mlp_top_pkg_content(
        input_neurons, output_neurons, input_data_bits, output_data_bits
    )
    
    # Write the updated file
    with open(target_mlp_pkg, 'w') as f:
        f.write(new_content)
    
    print(f"Created/updated {target_mlp_pkg} with:")
    print(f"  Input neurons: {input_neurons}")
    print(f"  Input data bits: {input_data_bits}")
    print(f"  Output neurons: {output_neurons}")
    print(f"  Output data bits: {output_data_bits}")


def create_mlp_pkg_from_config(network_config, target_vhdl_dir, source_mlp_pkg_path=None):
    """
    Create mlp_top_pkg.vhd from network configuration.
    
    Args:
        network_config (dict): Network configuration from parse_network_config
        target_vhdl_dir (Path): Target VHDL directory
        source_mlp_pkg_path (Path or None): Optional source file path
    """
    copy_and_update_mlp_pkg(
        source_mlp_pkg_path,
        target_vhdl_dir,
        network_config['input_neurons'],
        network_config['output_neurons'],
        network_config['input_data_bits'],
        network_config['output_data_bits']
    )
