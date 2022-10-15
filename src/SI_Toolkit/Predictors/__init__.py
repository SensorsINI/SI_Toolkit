import tensorflow as tf
import numpy as np

class template_predictor:
    def __init__(self, horizon: float, batch_size: int) -> None:
        self.horizon = horizon
        self.batch_size = batch_size
    
    def predict_tf(self, s: tf.Tensor, Q: tf.Tensor):
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
    