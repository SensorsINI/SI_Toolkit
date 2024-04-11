"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

from typing import Callable, Optional
from SI_Toolkit.Predictors import template_predictor
import numpy as np
from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization_v0 import STATE_VARIABLES
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization_v0 import next_state_predictor_ODE_v0


class predictor_ODE_v0(template_predictor):
    supported_computation_libraries = {NumpyLibrary}  # Overwrites default from parent
    
    def __init__(
        self,
        horizon: int,
        dt: float,
        intermediate_steps: int,
        batch_size=1,
        variable_parameters=None,
        **kwargs
    ):
        super().__init__(horizon=horizon, batch_size=batch_size)

        self.initial_state = None
        self.output = None

        self.next_step_predictor = next_state_predictor_ODE_v0(
            dt=dt,
            intermediate_steps=intermediate_steps,
            batch_size=batch_size,
            variable_parameters=variable_parameters,
        )

    def predict(self, initial_state: np.ndarray, Q: np.ndarray, params=None) -> np.ndarray:

        self.initial_state = initial_state

        if Q.ndim == 3:  # Q.shape = [batch_size, timesteps, features]
            self.batch_size = Q.shape[0]
        elif Q.ndim == 2:  # Q.shape = [timesteps, features]
            self.batch_size = 1
            Q = Q[np.newaxis, :, :]
        elif Q.ndim == 1:  # Q.shape = [features]
            self.batch_size = 1
            Q = Q[np.newaxis, np.newaxis, :]
        else:
            raise ValueError()

        # Make sure the input is at least 2d
        if self.initial_state.ndim == 1:
            self.initial_state = self.initial_state[np.newaxis, :]

        if self.initial_state.shape[0] == 1 and Q.shape[0] != 1:  # Predicting multiple control scenarios for the same initial state
            self.initial_state = np.tile(self.initial_state, (self.batch_size, 1))
        elif self.initial_state.shape[0] == Q.shape[0]:  # For each control scenario there is separate initial state provided
            pass
        else:
            raise ValueError('Batch size of control input contradict batch size of initial state')

        self.output = np.zeros((self.batch_size, self.horizon + 1, len(STATE_VARIABLES.tolist())), dtype=np.float32)
        self.output[:, 0, :] = self.initial_state

        for k in range(self.horizon):
            self.output[:, k + 1, :] = self.next_step_predictor.step(self.output[:, k, :], Q[:, k, :])

        return self.output if (self.batch_size > 1) else np.squeeze(self.output)

    def update_internal_state(self, Q0, s=None):
        pass


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE_v0 import predictor_ODE_v0
predictor = predictor_ODE_v0(horizon, 0.02, 10)
'''

    timer_predictor(initialisation)

