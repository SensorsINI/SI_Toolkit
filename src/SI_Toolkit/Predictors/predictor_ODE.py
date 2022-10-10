"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

from typing import Callable, Optional
import numpy as np
from CartPole.state_utilities import STATE_VARIABLES
from Control_Toolkit.others.environment import TensorType
from SI_Toolkit_ASF.predictors_customization import next_state_predictor_ODE
from SI_Toolkit.Predictors import predictor


class predictor_ODE(predictor):
    def __init__(
        self,
        horizon: int,
        dt: float,
        intermediate_steps: int,
        step_fun: Optional[Callable[[TensorType, TensorType, float], TensorType]]=None,
        batch_size=1,
        **kwargs
    ):
        super().__init__(horizon=horizon, batch_size=batch_size)

        self.initial_state = None
        self.output = None

        self.next_step_predictor = next_state_predictor_ODE(
            dt=dt, intermediate_steps=intermediate_steps, batch_size=batch_size, step_fun=step_fun
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
            self.output[:, k + 1, :] = self.next_step_predictor.step(self.output[:, k, :], Q[:, k, :], params)

        return self.output if (self.batch_size > 1) else np.squeeze(self.output)

    def update_internal_state(self, Q0, s=None):
        pass


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
predictor = predictor_ODE(horizon, 0.02, 10)
'''

    timer_predictor(initialisation)

