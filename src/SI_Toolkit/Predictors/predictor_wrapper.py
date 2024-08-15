import os
from typing import Optional
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary
from SI_Toolkit.load_and_normalize import load_yaml
from copy import deepcopy as dcp
from types import MappingProxyType, SimpleNamespace


# predictors config
predictors_config = load_yaml(os.path.join('SI_Toolkit_ASF', 'config_predictors.yml'), 'r')

NETWORK_NAMES = ['Dense', 'RNN', 'GRU', 'DeltaGRU', 'LSTM', 'Custom']

class PredictorWrapper:
    """Wrapper class for creating a predictor.
    
    1) Instantiate this wrapper without parameters within the controller class
    2) Pass the instance of this wrapper to the optimizer, without the need to already know specifics about it
    3) Call this wrapper's `configure` method in controller class to set optimization-specific parameters
    """
    def __init__(self):

        self.horizon = None
        self.batch_size = None

        self.num_states = None
        self.num_control_inputs = None

        self.predictor = None

        self.predictors_config = MappingProxyType(predictors_config['predictors'])  # Makes it read only
        self.predictor_name_default: str = predictors_config['predictor_name_default']

        self.predictor_name: str = self.predictor_name_default
        self.predictor_config = dcp(self.predictors_config[self.predictor_name])
        self.predictor_type: str = self.predictor_config['predictor_type']
        self.model_name: str = self.predictor_config['model_name']

    def configure(self, batch_size: int, horizon: int, dt: float, computation_library: "Optional[type[ComputationLibrary]]"=None, variable_parameters: SimpleNamespace=None, predictor_specification=None, compile_standalone=False, mode=None, hls=False):
        """Assign optimization-specific parameters to finalize instance creation.

        :param batch_size: Batch size equals the number of parallel rollouts of the optimizer.
        :type batch_size: int
        :param horizon: Number of MPC horizon steps
        :type horizon: int
        :param dt: Used to compute state trajectory rollouts
        :type dt: float
        :param computation_library: Whether to use NumPy / TensorFlow / PyTorch, defaults to None
        :type computation_library: Optional[type[ComputationLibrary]], optional
        :param variable_parameters: Parameters of the model which change during experiment and need to be realoaded
        :param predictor_specification: Name of predictor to use or path to neural network, defaults to None
        :type predictor_specification: _type_, optional
        :param compile_standalone: Whether to decorate the internal step function with its own compilation call, defaults to False
        :type compile_standalone: bool, optional
        :raises NotImplementedError: If the predictor type is not known
        :raises ValueError: Type of the predictor not recognised
        """
        self.update_predictor_config_from_specification(predictor_specification)

        compile_standalone = {'disable_individual_compilation': not compile_standalone}

        self.batch_size = batch_size
        self.horizon = horizon

        if self.predictor_type == 'neural':
            from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
            self.predictor = predictor_autoregressive_neural(horizon=self.horizon, batch_size=self.batch_size, variable_parameters=variable_parameters, dt=dt, mode=mode, hls=hls, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'GP':
            from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
            self.predictor = predictor_autoregressive_GP(horizon=self.horizon, batch_size=self.batch_size, variable_parameters=variable_parameters, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'ODE_v0':
            from SI_Toolkit.Predictors.predictor_ODE_v0 import predictor_ODE_v0
            self.predictor = predictor_ODE_v0(horizon=self.horizon, dt=dt, batch_size=self.batch_size, variable_parameters=variable_parameters, **self.predictor_config)

        elif self.predictor_type == 'ODE':
            from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
            if computation_library is None:  # TODO: Remove it after making sure that the predictor gets the right library everywhere it is used.
                computation_library = TensorFlowLibrary
            self.predictor = predictor_ODE(horizon=self.horizon, dt=dt, batch_size=self.batch_size, variable_parameters=variable_parameters, **self.predictor_config, **compile_standalone)

        else:
            raise NotImplementedError('Type of the predictor not recognised.')

        self.num_states = self.predictor.num_states
        self.num_control_inputs = self.predictor.num_control_inputs
        
        # computation_library defaults to None. In that case, do not check for conformity.
        if computation_library is not None and computation_library not in self.predictor.supported_computation_libraries:
            raise ValueError(f"Predictor {self.predictor.__class__.__name__} does not support {computation_library.__name__}")

    def configure_with_compilation(self, batch_size, horizon, dt, predictor_specification=None, mode=None, hls=False):
        """
        To get max performance
        use this for standalone predictors (like in Brunton test)
        but not predictors within controllers
        """
        self.configure(batch_size, horizon, dt, predictor_specification=predictor_specification, compile_standalone=True, mode=mode, hls=hls)

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
        if predictor_specification_components[0] == 'ODE_v0':
            predictor_name = 'ODE_v0_default'
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

            if any(network_name in predictor_specification for network_name in NETWORK_NAMES):
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

    def predict_core(self, s, Q):  # TODO: This function should disappear: predict() should manage the right library
        return self.predictor.predict_core(s, Q)

    def update(self, Q0, s):
        if self.predictor_type == 'neural':
            s = self.predictor.lib.to_tensor(s, self.predictor.lib.float32)
            Q0 = self.predictor.lib.to_tensor(Q0, self.predictor.lib.float32)
            self.predictor.update_internal_state_tf(s=s, Q0=Q0)

    def copy(self):
        """
        Makes a copy of a predictor, specification get preserved, configuration (batch_size, horizon) not
        The predictor needs to be reconfigured, however the specification needs not to be provided. 
        """
        predictor_copy = PredictorWrapper()   
        
        predictor_copy.predictor_name = self.predictor_name
        predictor_copy.predictor_config = dcp(self.predictor_config)
        predictor_copy.predictor_type = self.predictor_type
        predictor_copy.model_name = self.model_name
        
        return predictor_copy

