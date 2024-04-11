import os
from SI_Toolkit.Predictors import template_predictor
from SI_Toolkit.computation_library import TensorFlowLibrary, PyTorchLibrary, NumpyLibrary

from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import next_state_predictor_ODE, STATE_VARIABLES, CONTROL_INPUTS
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive

from SI_Toolkit.Predictors.autoregression import autoregression_loop, check_dimensions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


class model_interface:
    def __init__(self, single_step_predictor):
        self.model = single_step_predictor

    def __call__(self, model_input):
        Q = model_input[:, 0, :len(CONTROL_INPUTS)]
        s = model_input[:, 0, len(CONTROL_INPUTS):]
        return self.model.step(s, Q)


class predictor_ODE(template_predictor):
    supported_computation_libraries = {TensorFlowLibrary, PyTorchLibrary, NumpyLibrary}  # Overwrites default from parent
    
    def __init__(self,
                 horizon: int,
                 dt: float,
                 computation_library=TensorFlowLibrary,
                 intermediate_steps=10,
                 disable_individual_compilation=False,
                 batch_size=1,
                 variable_parameters=None,
                 **kwargs):
        super().__init__(horizon=horizon, batch_size=batch_size)
        self.lib = computation_library
        self.disable_individual_compilation = disable_individual_compilation

        self.initial_state = self.lib.zeros(shape=(1, len(STATE_VARIABLES)))
        self.output = None
        self.disable_individual_compilation = disable_individual_compilation

        self.dt = dt
        self.intermediate_steps = intermediate_steps

        self.next_step_predictor = next_state_predictor_ODE(
            dt,
            intermediate_steps,
            self.lib,
            self.batch_size,
            variable_parameters=variable_parameters,
            disable_individual_compilation=True,
        )
        self.params = self.next_step_predictor.params
        self.model = model_interface(self.next_step_predictor)

        self.AL: autoregression_loop = autoregression_loop(
            model_inputs_len=len(STATE_VARIABLES)  + len(CONTROL_INPUTS),
            model_outputs_len=len(STATE_VARIABLES),
            batch_size=self.batch_size,
            lib=self.lib,
            differential_model_autoregression_helper_instance=None,
        )

        if disable_individual_compilation:
            self.predict_core = self._predict_core
        else:
            self.predict_core = CompileAdaptive(self._predict_core)


    def predict(self, initial_state, Q):

        initial_state = self.lib.to_tensor(initial_state, dtype=self.lib.float32)
        Q = self.lib.to_tensor(Q, dtype=self.lib.float32)

        self.initial_state, Q = check_dimensions(initial_state, Q, self.lib)

        self.batch_size = self.lib.shape(Q)[0]

        output = self.predict_core(self.initial_state, Q)

        return output.numpy()


    def _predict_core(self, initial_state, Q):

        self.output = self.AL.run(
            model=self.model,
            horizon=self.horizon,
            initial_input=initial_state,
            external_input_left=Q,
        )

        self.output = self.lib.concat((initial_state[:, self.lib.newaxis, :], self.output), axis=1)

        return self.output

    def update_internal_state(self, Q, s=None):
        pass





if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
predictor = predictor_ODE(horizon, 0.02, 10)
'''

    timer_predictor(initialisation)
