import os
import yaml

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

from types import SimpleNamespace

from SI_Toolkit.Functions.General.Initialization import get_net
from SI_Toolkit.Functions.TF.Network import plot_weights_distribution

# Parameters:

a = SimpleNamespace()
batch_size = 1
a.path_to_models = './SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Imitation/Models/'
a.net_name = 'Dense-7IN-64H1-64H2-1OUT-4'


# Import network
# Create a copy of the network suitable for inference (stateful and with sequence length one)
model, net_info = \
    get_net(a, time_series_length=1,
            batch_size=1, stateful=True, remove_redundant_dimensions=False)

plot_weights_distribution(model, show=True, path_to_save=None)

