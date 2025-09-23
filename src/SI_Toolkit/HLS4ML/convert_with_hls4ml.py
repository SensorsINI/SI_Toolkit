import yaml
import os
import shutil
import hls4ml
import tempfile
import time
from pathlib import Path

from types import SimpleNamespace

from SI_Toolkit.Functions.General.Initialization import get_net
from SI_Toolkit.HLS4ML.hls4ml_functions import convert_model_with_hls4ml
from SI_Toolkit.Functions.General.TerminalContentManager import TerminalContentManager
from SI_Toolkit.HLS4ML.network_parser import parse_network_config
from SI_Toolkit.HLS4ML.vhdl_package_generator import create_mlp_pkg_from_config


def convert_with_hls4ml():

    config_hls = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_hls.yml'), 'r'), Loader=yaml.FullLoader)

    # Parameters:
    a = SimpleNamespace()
    batch_size = config_hls['batch_size']
    a.path_to_models = config_hls['path_to_models']
    a.net_name = config_hls['net_name']

    # Create unique temporary directory in home folder to avoid long path issues
    # This allows multiple conversions to run simultaneously
    home_dir = Path.home()
    unique_id = f"{int(time.time() * 1000)}_{os.getpid()}"  # timestamp + process ID for uniqueness
    temp_dir = os.path.join(home_dir, f'hls4ml_temp_{unique_id}')
    temp_output_dir = os.path.join(temp_dir, a.net_name)
    
    # Final destination for the converted network with output folder name suffix
    hls4ml_output_folder_name = config_hls.get('hls4ml_output_folder_name', 'default')
    final_output_dir = os.path.join(a.path_to_models, a.net_name, f'hls4ml_{hls4ml_output_folder_name}')
    
    try:
        with TerminalContentManager(os.path.join(temp_dir, 'terminal_output.txt')):
            # Import network
            # Create a copy of the network suitable for inference (stateful and with sequence length one)
            net, net_info = \
                get_net(a, time_series_length=1,
                        batch_size=batch_size, stateful=True, remove_redundant_dimensions=True)

            # Convert model using temporary directory
            hls_model, hls_model_config = convert_model_with_hls4ml(net, temp_output_dir=temp_dir)

            hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

            # Synthesis
            hls_model.build(reset=False, csim=True, synth=True, cosim=True, validation=True, export=True, vsynth=True)
            # hls_model.build(csim=False)

            # Reports
            hls4ml.report.read_vivado_report(temp_dir)
            
            # Create final output directory
            if os.path.exists(final_output_dir):
                shutil.rmtree(final_output_dir)
            os.makedirs(final_output_dir, exist_ok=True)
            
            # Copy only the files we need: hls4ml_config.yml, terminal_output.txt, config_hls.yml, verilog/, vhdl/
            files_to_copy = ['hls4ml_config.yml', 'terminal_output.txt']
            for file_name in files_to_copy:
                src_file = os.path.join(temp_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, final_output_dir)
            
            # Also copy the original config_hls.yml file for future reference
            config_hls_src = os.path.join('SI_Toolkit_ASF', 'config_hls.yml')
            if os.path.exists(config_hls_src):
                shutil.copy2(config_hls_src, final_output_dir)
            
            # Copy verilog and vhdl folders if they exist
            for folder_name in ['verilog', 'vhdl']:
                src_folder = os.path.join(temp_dir, 'myproject_prj', 'solution1', 'impl', folder_name)
                if os.path.exists(src_folder):
                    dst_folder = os.path.join(final_output_dir, folder_name)
                    shutil.copytree(src_folder, dst_folder)
            
            # Copy and update mlp_top_pkg.vhd in the vhdl folder
            vhdl_dir = Path(final_output_dir) / 'vhdl'
            if vhdl_dir.exists():
                # Parse network configuration
                network_config = parse_network_config(a.net_name, config_hls['PRECISION'])
                
                # Copy and update mlp_top_pkg.vhd
                source_mlp_pkg = Path('FPGA/NeuralNetworks/Interfaces/mlp_top_pkg.vhd')
                if not source_mlp_pkg.exists():
                    print(f"Warning: Source mlp_top_pkg.vhd not found at {source_mlp_pkg}")
                    source_mlp_pkg = None
                
                # Create the updated mlp_top_pkg.vhd
                create_mlp_pkg_from_config(network_config, vhdl_dir, source_mlp_pkg)
            
            print(f"Converted network successfully moved to: {final_output_dir}")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

