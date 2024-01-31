import os
from SI_Toolkit.Predictors import template_predictor
from SI_Toolkit.computation_library import TensorFlowLibrary


from SI_Toolkit_ASF.predictors_customization_tf import next_state_predictor_ODE_tf, STATE_VARIABLES
from SI_Toolkit.Functions.TF.Compile import CompileTF

import tensorflow as tf

lib = TensorFlowLibrary

from SI_Toolkit.Predictors.autoregression import check_dimensions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

class predictor_ODE_tf(template_predictor):
    supported_computation_libraries = {TensorFlowLibrary}  # Overwrites default from parent
    
    def __init__(self,
                 horizon: int,
                 dt: float,
                 intermediate_steps=10,
                 disable_individual_compilation=False,
                 batch_size=1,
                 variable_parameters=None,
                 **kwargs):
        super().__init__(horizon=horizon, batch_size=batch_size)
        self.lib = lib
        self.disable_individual_compilation = disable_individual_compilation

        self.initial_state = self.lib.zeros(shape=(1, len(STATE_VARIABLES)))
        self.output = None
        self.disable_individual_compilation = disable_individual_compilation

        self.dt = dt
        self.intermediate_steps = intermediate_steps

        self.next_step_predictor = next_state_predictor_ODE_tf(
            dt,
            intermediate_steps,
            self.batch_size,
            variable_parameters=variable_parameters,
            disable_individual_compilation=True,
        )

        if disable_individual_compilation:
            self.predict_tf = self._predict_tf
        else:
            self.predict_tf = CompileTF(self._predict_tf)


    def predict(self, initial_state, Q):
        initial_state = self.lib.to_tensor(initial_state, dtype=self.lib.float32)
        Q = self.lib.to_tensor(Q, dtype=self.lib.float32)

        initial_state, Q = check_dimensions(initial_state, Q, self.lib)

        self.batch_size = self.lib.shape(Q)[0]
        self.initial_state = initial_state

        output = self.predict_tf(self.initial_state, Q)

        return output.numpy()


    def _predict_tf(self, initial_state, Q):

        self.output = tf.TensorArray(self.lib.float32, size=self.horizon + 1, dynamic_size=False)
        self.output = self.output.write(0, initial_state)

        next_state = initial_state

        for k in self.lib.arange(0, self.horizon):
            next_state = self.next_step_predictor.step(next_state, Q[:, k, :])
            self.output = self.output.write(k + 1, next_state)

        self.output = self.lib.permute(self.output.stack(), perm=[1, 0, 2])

        return self.output

    def update_internal_state(self, Q, s=None):
        pass





if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
predictor = predictor_ODE_tf(horizon, 0.02, 10)
'''

    timer_predictor(initialisation)
