import os
import yaml

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

from types import SimpleNamespace

from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Network import plot_weights_distribution, get_activation_statistics
from SI_Toolkit.Functions.TF.Dataset import Dataset

# Parameters:

a = SimpleNamespace()
a.batch_size = 1
a.path_to_models = './SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Model/Models/'
a.net_name = 'Dense-6IN-32H1-32H2-5OUT-0'
testing_files = ['SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Model/Recordings/Train/']
path_to_parameters_distribution_histograms = None

a.shift_labels = 1
a.wash_out_len = 0
a.post_wash_out_len = 1


# Import network
# Create a copy of the network suitable for inference (stateful and with sequence length one)
model, net_info = \
    get_net(a, time_series_length=1,
            batch_size=a.batch_size, stateful=True, remove_redundant_dimensions=False)

model.summary()

if path_to_parameters_distribution_histograms is None:
    path_to_parameters_distribution_histograms = os.path.join(net_info.path_to_net, 'parameters_histograms')

try:
    os.makedirs(path_to_parameters_distribution_histograms)
except FileExistsError:
    pass

plot_weights_distribution(model, show=False, path_to_save=path_to_parameters_distribution_histograms)

if testing_files:
    paths_to_datafiles = get_paths_to_datafiles(testing_files)
    testing_dfs = load_data(paths_to_datafiles)
    if net_info.normalize:
        normalization_info = get_norm_info_for_net(net_info)
        testing_dfs = normalize_df(testing_dfs, normalization_info)
    testing_dataset = Dataset(testing_dfs, a, shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)
    activation_statistics_datasets = [testing_dataset]
    get_activation_statistics(model, activation_statistics_datasets, path_to_save=path_to_parameters_distribution_histograms)


