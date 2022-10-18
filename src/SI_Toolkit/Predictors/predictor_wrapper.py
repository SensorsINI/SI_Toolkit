import os
import yaml
from copy import deepcopy as dcp
from types import MappingProxyType


# predictors config
predictors_config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_predictors.yml'), 'r'), Loader=yaml.FullLoader)


class PredictorWrapper:
    def __init__(self):

        self.horizon = None
        self.batch_size = None

        self.predictor = None

        self.predictors_config = self.predictors_config = MappingProxyType(predictors_config['predictors'])  # Makes it read only
        self.predictor_name_default: str = predictors_config['predictor_name_default']

        self.predictor_name: str = self.predictor_name_default
        self.predictor_config = dcp(self.predictors_config[self.predictor_name])
        self.predictor_type: str = self.predictor_config['predictor_type']
        self.model_name: str = self.predictor_config['model_name']


    def configure(self, batch_size, horizon, predictor_specification=None, compile_standalone=False):

        self.update_predictor_config_from_specification(predictor_specification)

        compile_standalone = {'disable_individual_compilation': not compile_standalone}

        self.batch_size = batch_size
        self.horizon = horizon

        if self.predictor_type == 'neural':
            from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
            self.predictor = predictor_autoregressive_tf(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'GP':
            from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
            self.predictor = predictor_autoregressive_GP(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'ODE':
            from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
            self.predictor = predictor_ODE(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config)

        elif self.predictor_type == 'ODE_TF':
            from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
            self.predictor = predictor_ODE_tf(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        else:
            raise NotImplementedError('Type of the predictor not recognised.')

    def configure_with_compilation(self, batch_size, horizon, predictor_specification=None):
        """
        To get max performance
        use this for standalone predictors (like in Brunton test)
        but not predictors within controllers
        """
        self.configure(batch_size, horizon, predictor_specification, compile_standalone=True)

    def update_predictor_config_from_specification(self, predictor_specification: str = None):

        if predictor_specification is None:  # The default values are not modified
            return
        if predictor_specification == 'default':
            self.predictor_name: str = self.predictor_name_default
            self.predictor_config = dcp(self.predictors_config['predictors'][self.predictor_name])
            self.predictor_type: str = self.predictor_config['predictor_type']
            self.model_name: str = self.predictor_config['model_name']


        predictor_name = None
        model_name = None
        model_name_contains_path_to_model = False

        # Sort out predictor name from component specification
        predictor_specification_components = predictor_specification.split(":")

        # Search if predictor with this name exists:

        if predictor_specification_components[0] == 'ODE':
            predictor_name = 'ODE_default'
        if predictor_specification_components[0] == 'ODE_TF':
            predictor_name = 'ODE_TF_default'
        if predictor_specification_components[0] == 'neural':
            predictor_name = 'neural_default'
        if predictor_specification_components[0] == 'GP':
            predictor_name = 'GP_default'

        if predictor_name is None:
            for predefined_predictor in self.predictors_config.keys():
                if predictor_specification_components[0] == predefined_predictor:
                    predictor_name = predictor_specification_components[0]

        # Search if the specification gives a network name from which you can construct predictor
        if predictor_name is None:

            networks_names = ['Dense', 'RNN', 'GRU', 'DeltaGRU', 'LSTM']
            if any(network_name in predictor_specification for network_name in networks_names):
                predictor_name = 'neural_default'
                model_name = predictor_specification_components[0]
            elif 'SGP' in predictor_specification:
                predictor_name = 'GP_default'
                model_name = predictor_specification_components[0]

            if len(os.path.normpath(model_name).split(os.path.sep)) > 1:
                model_name_contains_path_to_model = True

        if predictor_name is None:
            raise ValueError('{} is an invalid predictor specification'.format(predictor_specification))

        if len(predictor_specification_components) > 1 and model_name is None:
            model_name = predictor_specification_components[1]

        self.predictor_name = predictor_name
        self.predictor_config = dcp(predictors_config['predictors'][self.predictor_name])
        self.predictor_type = self.predictor_config['predictor_type']
        if model_name is not None:
            self.predictor_config['model_name'] = model_name
        self.model_name = self.predictor_config['model_name']

        if model_name_contains_path_to_model is True:  # We want to delete the default path to model if model_name contains one
            self.predictor_config['path_to_model'] = None

    def predict(self, s, Q):
        return self.predictor.predict(s, Q)

    def predict_tf(self, s, Q):  # TODO: This function should disappear: predict() should manage the right library
        return self.predictor.predict_tf(s, Q)

    def update(self, Q0, s):
        if self.predictor_type == 'neural':
            self.predictor.update_internal_state_tf(s=s, Q0=Q0)


