import yaml
import os
import hls4ml
from SI_Toolkit.load_and_normalize import load_yaml

config_hls = load_yaml(os.path.join('SI_Toolkit_ASF', 'config_hls.yml'))
os.environ['PATH'] = config_hls['path_to_hls_installation'] + ":" + os.environ['PATH']


def print_dict(d, indent=0):
    """From hls4ml files"""
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


def convert_model_with_hls4ml(net, granularity='model'):

    config = hls4ml.utils.config_from_keras_model(net, granularity='name')

    # config['Flows'] = ['vivado:fifo_depth_optimization']
    # hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

    config['Model']['Precision'] = config_hls['PRECISION']['intermediate_results']
    # Iterate through all layers in the HLS configuration
    for layer_name, layer_config in config['LayerName'].items():
        if 'input' in layer_name:
            layer_config['Precision']['result'] = config_hls['PRECISION']['input_and_output']
            layer_config['Precision']['weight'] = config_hls['PRECISION']['weights_and_biases']  # Not sure what does
            layer_config['Precision']['bias'] = config_hls['PRECISION']['weights_and_biases']
        # Check if the layer is an activation layer (assuming 'activation' in the name)
        elif 'activation' in layer_name:
            # Set precision for activation layers
            layer_config['Precision']['result'] = config_hls['PRECISION']['activations']
        else:
            # Set precision for other layers
            layer_config['Precision']['result'] = config_hls['PRECISION']['intermediate_results']
            layer_config['Precision']['weight'] = config_hls['PRECISION']['weights_and_biases']
            layer_config['Precision']['bias'] = config_hls['PRECISION']['weights_and_biases']

    config['LayerName'][list(config["LayerName"].keys())[-1]]['Precision']['result'] = config_hls['PRECISION']['input_and_output']  # Output of last layer


    config['Model']['Strategy'] = config_hls['Strategy']
    config['Model']['ReuseFactor'] = config_hls['ReuseFactor']

    print_dict(config)

    hls_model = hls4ml.converters.convert_from_keras_model(net,
                                                           hls_config=config,
                                                           output_dir=config_hls['output_dir'],
                                                           backend=config_hls['backend'],
                                                           ## !!!! If the path is long it crashes. Depending on how long is the path it crashes at different places.
                                                           part=config_hls['part'],                                                     # board=config_hls['board'],
    )

    hls_model.compile()

    return hls_model, config


# TODO: Not used yet as I don't have a way to feed dataset
#   Also the issue of setting precision for different granularity is not solved.
def hls4ml_numerical_model_profiling(net, data):

    hls_model, hls_model_config = convert_model_with_hls4ml(net, granularity='name')

    for layer in hls_model_config['LayerName'].keys():
        hls_model_config['LayerName'][layer]['Trace'] = True

    hls4ml.model.profiling.numerical(model=net, hls_model=hls_model, X=data)

