from SI_Toolkit.computation_library import TensorFlowLibrary

from SI_Toolkit.Functions.General.Normalising import get_scaling_function_for_output_of_differential_network

from SI_Toolkit_ASF.predictors_customization import (CONTROL_INPUTS,
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
    ):

        if self.lib.lib == 'TF':
            outputs = self.TensorArray(self.lib.float32, size=horizon)
        else:
            outputs = self.lib.zeros([self.batch_size, horizon, self.model_outputs_len])

        if self.dmah:
            self.dmah.set_starting_point(initial_input)

        next_model_input = initial_input
        for i in self.lib.arange(horizon):

            model_input = next_model_input
            if external_input_left is not None:
                model_input = self.lib.concat([external_input_left[:, i, :], model_input], axis=1)
            if external_input_right is not None:
                model_input = self.lib.concat([model_input, external_input_right[:, i, :]], axis=1)

            model_input = self.lib.reshape(model_input, shape=[-1, 1, self.model_inputs_len])

            model_output = model(model_input)

            model_output = self.lib.reshape(model_output, [-1, self.model_outputs_len])

            if self.dmah:
                output, next_model_input = self.dmah.get_output_and_next_model_input(model_output)
            else:
                output = model_output
                next_model_input = model_output

            if self.lib.lib == 'TF':
                outputs = outputs.write(i, output)
            else:
                outputs[:, i, :] = output

        if self.lib.lib == 'TF':
            outputs = self.lib.permute(outputs.stack(), [1, 0, 2])

        return outputs


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

        self.rescale_output_diff_model = get_scaling_function_for_output_of_differential_network(
            normalization_info,
            outputs,
            dt,
            self.lib
        )

        outputs_names = np.array([x[2:] for x in outputs])

        self.indices_state_to_output = self.lib.to_tensor([STATE_INDICES.get(key) for key in outputs_names],
                                                          dtype=self.lib.int64)
        output_indices = {x: np.where(outputs_names == x)[0][0] for x in outputs_names}
        self.indices_output_to_input = self.lib.to_tensor(
            [output_indices.get(key) for key in inputs[len(CONTROL_INPUTS):]], dtype=self.lib.int64)

        starting_point = self.lib.zeros([batch_size, len(STATE_VARIABLES)], dtype=self.lib.float32)
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