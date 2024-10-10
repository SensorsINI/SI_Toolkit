import yaml
import os
import shutil
import hls4ml

from types import SimpleNamespace

from SI_Toolkit.Functions.General.Initialization import get_net
from SI_Toolkit.HLS4ML.hls4ml_functions import convert_model_with_hls4ml
from SI_Toolkit.Functions.General.TerminalContentManager import TerminalContentManager


def convert_with_hls4ml():

    config_hls = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_hls.yml'), 'r'), Loader=yaml.FullLoader)

    # Parameters:
    a = SimpleNamespace()
    batch_size = config_hls['batch_size']
    a.path_to_models = config_hls['path_to_models']
    a.net_name = config_hls['net_name']

    with TerminalContentManager(os.path.join(config_hls['output_dir'], 'terminal_output.txt')):
        # Import network
        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        net, net_info = \
            get_net(a, time_series_length=1,
                    batch_size=batch_size, stateful=True, remove_redundant_dimensions=True)

        path_to_network = os.path.join(a.path_to_models, a.net_name)
        path_to_hls_network = os.path.join(config_hls['output_dir'], a.net_name)
        shutil.copytree(path_to_network, path_to_hls_network)


        hls_model, hls_model_config = convert_model_with_hls4ml(net)

        hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)


        # Synthesis
        hls_model.build(reset=False, csim=True, synth=True, cosim=True, validation=True, export=True, vsynth=True)
        # hls_model.build(csim=False)

        # Reports
        hls4ml.report.read_vivado_report(config_hls['output_dir'])

