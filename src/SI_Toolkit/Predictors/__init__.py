from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary
import numpy as np

from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import (
    CONTROL_INPUTS_FOR_PREDICTOR, STATE_VARIABLES_FOR_PREDICTOR
)

class template_predictor:
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)
    
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

        # Default, can be overwritten
        self.predictor_initial_input_features = STATE_VARIABLES_FOR_PREDICTOR
        self.predictor_external_input_features = CONTROL_INPUTS_FOR_PREDICTOR
        self.predictor_output_features = STATE_VARIABLES_FOR_PREDICTOR

        self.num_states = len(STATE_VARIABLES_FOR_PREDICTOR)
        self.num_control_inputs = len(CONTROL_INPUTS_FOR_PREDICTOR)

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
    