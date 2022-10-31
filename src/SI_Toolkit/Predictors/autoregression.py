from SI_Toolkit.Functions.General.Normalising import get_scaling_function_for_output_of_differential_network

from SI_Toolkit_ASF.predictors_customization import (CONTROL_INPUTS,
                                                     STATE_INDICES,
                                                     STATE_VARIABLES)

import numpy as np

class differential_model_autoregression_helper:
    def __init__(self,
                inputs,
                outputs,
                normalization_info,
                dt,
                batch_size,
                horizon,
                state_len,
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
