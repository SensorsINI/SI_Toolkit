from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary
import numpy as np


class template_predictor:
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)
    
    def __init__(self, batch_size: int, control_inputs=None, state_variables=None) -> None:
        self.batch_size = batch_size

        # State variables: must be passed (from config_predictors.yml or net_info)
        if state_variables is not None:
            self.state_variables = np.array(state_variables)
            self.predictor_initial_input_features = self.state_variables
            self.predictor_output_features = self.state_variables
            self.num_states = len(self.state_variables)
        else:
            # Will be set by subclass (e.g., neural predictor from net_info)
            self.state_variables = None
            self.predictor_initial_input_features = np.array([])
            self.predictor_output_features = np.array([])
            self.num_states = 0
        
        # Control inputs: must be passed (from config_predictors.yml or net_info)
        if control_inputs is not None:
            self.predictor_external_input_features = np.array(control_inputs)
        else:
            self.predictor_external_input_features = np.array([])  # Will be set by subclass

        self.num_control_inputs = len(self.predictor_external_input_features)

    def predict_core(self, s: 'tf.Tensor', Q: 'tf.Tensor'):
        """Predict the whole MPC horizon using tensorflow

        :param s: Initial state [batch_size x state_dim]
        :type s: tf.Tensor
        :param Q: Control inputs [batch_size x horizon_length x control_dim]
        :type Q: tf.Tensor
        """
        raise NotImplementedError()
    
    def predict(self, s: np.ndarray, Q: np.ndarray):
        """Predict the whole MPC horizon using numpy

        :param s: Initial state [batch_size x state_dim]
        :type s: np.ndarray
        :param Q: Control inputs [batch_size x horizon_length x control_dim]
        :type Q: np.ndarray
        """
        raise NotImplementedError()
