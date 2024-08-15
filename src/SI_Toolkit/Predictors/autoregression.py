from SI_Toolkit.computation_library import TensorFlowLibrary

from SI_Toolkit.Functions.General.Normalising import get_scaling_function_for_output_of_differential_network

from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import (CONTROL_INPUTS,
                                                                          STATE_INDICES,
                                                                          STATE_VARIABLES)

import numpy as np


class autoregression_loop:
    def __init__(
            self,
            model_inputs_len,
            model_outputs_len,
            batch_size,
            lib,
            differential_model_autoregression_helper_instance=None,
    ):
        self.lib = lib
        self.dmah = differential_model_autoregression_helper_instance

        self.model_inputs_len = model_inputs_len
        self.model_outputs_len = model_outputs_len
        self.batch_size = batch_size


        if self.lib == TensorFlowLibrary:
            from tensorflow import TensorArray
            self.TensorArray = TensorArray

    def run(
            self,
            model,
            horizon,
            initial_input,
            external_input_left=None,
            external_input_right=None,
            predictor='neural',
    ):

        if self.lib.lib == 'TF':
            outputs = self.TensorArray(self.lib.float32, size=horizon)
        else:
            outputs = self.lib.zeros([self.batch_size, horizon, self.model_outputs_len])

        model_input = initial_input

        if predictor == 'gp' or horizon == 1:
            # The only difference it that for gp-predictor first interation of the for loop is done outside of the loop
            # Otherwise tf.function throws error.
            # This can be corrected back and only general loop used as soon as GPs are loaded in α not compiled state.

            ############### Oth ITERATION! ####################
            if external_input_left is not None:
                model_input = self.lib.concat([external_input_left[:, 0, :], model_input], axis=1)
            if external_input_right is not None:
                model_input = self.lib.concat([model_input, external_input_right[:, 0, :]], axis=1)

            model_input = self.lib.reshape(model_input, shape=[-1, 1, self.model_inputs_len])

            model_output = self.evaluate_model(model, model_input)

            model_output = self.lib.reshape(model_output, [-1, self.model_outputs_len])

            output = model_output
            model_input = model_output

            if self.lib.lib == 'TF':
                outputs = outputs.write(0, output)
            else:
                outputs[:, 0, :] = output

            ##################### END OF 0th ITERATION ######################
            arange = self.lib.arange(1, horizon)
        else:
            arange = self.lib.arange(0, horizon)

        if horizon >  1:
            for i in arange:

                if external_input_left is not None:
                    model_input = self.lib.concat([external_input_left[:, i, :], model_input], axis=1)
                if external_input_right is not None:
                    model_input = self.lib.concat([model_input, external_input_right[:, i, :]], axis=1)

                model_input = self.lib.reshape(model_input, shape=[-1, 1, self.model_inputs_len])

                model_output = self.evaluate_model(model, model_input)

                model_output = self.lib.reshape(model_output, [-1, self.model_outputs_len])

                if self.dmah:
                    output, model_input = self.dmah.get_output_and_next_model_input(model_output)
                else:
                    output = model_output
                    model_input = model_output

                if self.lib.lib == 'TF':
                    outputs = outputs.write(i, output)
                else:
                    outputs[:, i, :] = output

        if self.lib.lib == 'TF':
            outputs = self.lib.permute(outputs.stack(), [1, 0, 2])

        return outputs

    def evaluate_model(self, model, model_input):
        if self.lib.lib == 'Numpy':  # Covers just the case for hls4ml, when the model is hls model
            return model.predict(model_input)
        else:
            return model(model_input)



class differential_model_autoregression_helper:
    def __init__(self,
                inputs,
                outputs,
                normalization_info,
                dt,
                batch_size,
                lib,
                ):
        self.lib = lib

        if normalization_info is not None:
            self.rescale_output_diff_model = get_scaling_function_for_output_of_differential_network(
                normalization_info,
                outputs,
                dt,
                self.lib
            )
        else:
            self.rescale_output_diff_model = lambda x: x * dt

        outputs_names_after_integration = np.array([x[2:] for x in outputs])

        self.indices_state_to_output = self.lib.to_tensor([STATE_INDICES.get(key) for key in outputs_names_after_integration],
                                                          dtype=self.lib.int64)
        output_indices = {x: np.where(outputs_names_after_integration == x)[0][0] for x in outputs_names_after_integration}
        self.indices_output_to_input = self.lib.to_tensor(
            [output_indices.get(key) for key in inputs[len(CONTROL_INPUTS):]], dtype=self.lib.int64)

        starting_point = self.lib.zeros([batch_size, len(outputs_names_after_integration)], dtype=self.lib.float32)
        self.starting_point = self.lib.to_variable(starting_point, self.lib.float32)

    def set_starting_point(self, starting_point):
        self.lib.assign(self.starting_point, self.lib.gather_last(starting_point, self.indices_state_to_output))

    def get_output_and_next_model_input(self, differential_model_output):
        self.lib.assign(self.starting_point,
                        self.starting_point + self.rescale_output_diff_model(differential_model_output))
        output = self.starting_point
        next_model_input = self.lib.gather_last(output, self.indices_output_to_input)
        return output, next_model_input


def check_dimensions(s, Q, lib):
    # Make sure the input is at least 2d
    if s is not None:
        if lib.ndim(s) == 1:
            s = s[lib.newaxis, :]

    if lib.ndim(Q) == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif lib.ndim(Q) == 2:  # Q.shape = [timesteps, features]
        Q = Q[lib.newaxis, :, :]
    else:  # Q.shape = [features;  rank(Q) == 1
        Q = Q[lib.newaxis, lib.newaxis, :]

    return s, Q