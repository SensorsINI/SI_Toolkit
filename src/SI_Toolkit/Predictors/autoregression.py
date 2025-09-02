from SI_Toolkit.computation_library import TensorFlowLibrary

from SI_Toolkit.Functions.General.Normalising import get_scaling_function_for_output_of_differential_network

from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import (CONTROL_INPUTS_FOR_PREDICTOR,
                                                                          STATE_VARIABLES_FOR_PREDICTOR)

import numpy as np




class autoregression_loop:
    def __init__(
            self,
            model,
            model_inputs_len,
            model_outputs_len,
            lib,
            differential_model_autoregression_helper_instance=None,
    ):
        self.lib = lib
        self.model = model
        self.dmah = differential_model_autoregression_helper_instance

        self.model_inputs_len = model_inputs_len
        self.model_outputs_len = model_outputs_len

        self.evaluate_model = self.evaluate_model_factory()


        if isinstance(self.lib, TensorFlowLibrary):
            from tensorflow import TensorArray
            self.TensorArray = TensorArray

    def evaluate_model_factory(self):
        if callable(self.model):
            evaluate_model = self.model  # direct invocation e.g. model(input_data), TF, PyTorch.
        else:
            evaluate_model = self.model.predict  # hls4ml

        return evaluate_model

    # infer dynamic horizon from exogenous inputs (TF: tensor; PT/NP: int)
    def _infer_steps(self, external_input_left, external_input_right):
        def _time_len(x):
            if x is None:
                return None
            return self.lib.shape(x)[1]

        L = _time_len(external_input_left)
        R = _time_len(external_input_right)

        if self.lib.lib == 'TF':
            import tensorflow as tf
            if (external_input_left is None) and (external_input_right is None):
                return tf.convert_to_tensor(1, dtype=tf.int32)
            if (external_input_left is not None) and (external_input_right is not None):
                with tf.control_dependencies([
                    tf.debugging.assert_equal(L, R, message="left/right lengths must match")
                ]):
                    return tf.cast(L, tf.int32)
            return tf.cast(L if L is not None else R, tf.int32)
        else:
            if (external_input_left is None) and (external_input_right is None):
                return 1
            if (external_input_left is not None) and (external_input_right is not None):
                if L != R:
                    raise ValueError("left/right lengths must match")
                return int(L)
            return int(L if L is not None else R)

    def horizon_step(self, model_input, current_external_input_left, current_external_input_right, dm_state=None):
        if current_external_input_left is not None:
            model_input = self.lib.concat([current_external_input_left, model_input], axis=1)
        if current_external_input_right is not None:
            model_input = self.lib.concat([model_input, current_external_input_right], axis=1)

        model_input = self.lib.reshape(model_input, shape=[-1, 1, self.model_inputs_len])

        model_output = self.evaluate_model(model_input)

        model_output = self.lib.reshape(model_output, [-1, self.model_outputs_len])

        if self.dmah is not None:
            if dm_state is None:
                raise ValueError("dm_state must be provided when using a differential model helper.")
            output, next_model_input, next_dm_state = self.dmah.apply(dm_state, model_output)
            return output, next_model_input, next_dm_state
        else:
            output = model_output
            next_model_input = model_output
            return output, next_model_input, None

    def run(
            self,
            initial_input,
            external_input_left=None,
            external_input_right=None,
            dm_state_init=None,   # ← NEW: optional full-length seed for differential models
    ):

        # derive horizon at runtime from exogenous inputs
        horizon = self._infer_steps(external_input_left, external_input_right)

        if self.lib.lib == 'TF':
            outputs = self.TensorArray(self.lib.float32, size=horizon)
        else:
            outputs = self.lib.zeros([self.lib.shape(initial_input)[0], int(horizon), self.model_outputs_len])

        model_input = initial_input

        if self.dmah is not None:
            if dm_state_init is not None:
                # Use the caller-provided full state (ordered exactly as outputs after integration).
                # This avoids dimensionality/normalization mismatches when some state features
                # are not present among the network inputs.
                dm_state = dm_state_init
            else:
                # Backward-compatible fallback (subset inferred from inputs).
                # This may be shorter than outputs if some state features are not inputs.
                dm_state = self.lib.gather_last(model_input, self.dmah.indices_output_to_input)
        else:
            dm_state = self.lib.zeros([self.lib.shape(model_input)[0], 0], dtype=self.lib.float32)

        ############### Oth ITERATION! ####################
        if external_input_left is not None:
            model_input = self.lib.concat([external_input_left[:, 0, :], model_input], axis=1)
        if external_input_right is not None:
            model_input = self.lib.concat([model_input, external_input_right[:, 0, :]], axis=1)

        model_input = self.lib.reshape(model_input, shape=[-1, 1, self.model_inputs_len])

        model_output = self.evaluate_model(model_input)

        model_output = self.lib.reshape(model_output, [-1, self.model_outputs_len])

        if self.dmah is not None:
            output, model_input, dm_state = self.dmah.apply(dm_state, model_output)
        else:
            dm_state = self.lib.zeros([self.lib.shape(model_input)[0], 0])
            output = model_output
            model_input = model_output

        if self.lib.lib == 'TF':
            outputs = outputs.write(0, output)
        else:
            outputs[:, 0, :] = output

        ##################### END OF 0th ITERATION ######################

        # backend-aware loop with steps_rem; 0 ⇒ no-op
        if self.lib.lib == 'TF':
            steps_rem = self.lib.cast(horizon - 1, self.lib.int32)
            start_idx = self.lib.cast(1, self.lib.int32)
        else:
            steps_rem = int(horizon) - 1
            start_idx = 1

        def loop_body(i, outputs, current_input, dm_state):
            step = i
            if self.lib.lib == "Pytorch":
                if getattr(step, "ndim", 0) == 0:
                    step = step.unsqueeze(0)
            elif self.lib.lib == "TF":
                step = self.lib.reshape(step, (1,))

            def _take_step(x):
                if x is None:
                    return None
                slice_ = self.lib.gather(x, step, axis=1)
                return self.lib.squeeze(slice_, 1)

            left = _take_step(external_input_left)
            right = _take_step(external_input_right)

            output, nxt_in, nxt_dm = self.horizon_step(current_input, left, right, dm_state)

            if self.dmah is None:
                nxt_dm = dm_state  # carry the dummy tensor forward

            if self.lib.lib == "TF":
                outputs = outputs.write(i, output)
            elif self.lib.lib == "Pytorch":
                outputs = outputs.index_copy(1, step, output.unsqueeze(1))
            else:  # NumPy
                outputs[:, int(step), :] = output

            return (i + 1, outputs, nxt_in, nxt_dm)

        _, outputs, _, _ = self.lib.loop(
            loop_body,
            (outputs, model_input, dm_state),
            steps_rem,
            start_idx
        )


        if self.lib.lib == 'TF':
            outputs = self.lib.permute(outputs.stack(), [1, 0, 2])

        return outputs



class differential_model_autoregression_helper:
    def __init__(self,
                inputs,
                outputs,
                normalization_info,
                dt,
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

        STATE_INDICES = {x: np.where(STATE_VARIABLES_FOR_PREDICTOR == x)[0][0] for x in STATE_VARIABLES_FOR_PREDICTOR}
        self.indices_state_to_output = self.lib.to_tensor([STATE_INDICES.get(key) for key in outputs_names_after_integration],
                                                          dtype=self.lib.int64)
        output_indices = {x: np.where(outputs_names_after_integration == x)[0][0] for x in outputs_names_after_integration}

        num_controls_present = sum(1 for name in inputs if name in CONTROL_INPUTS_FOR_PREDICTOR)

        self.indices_output_to_input = self.lib.to_tensor(
            [output_indices.get(key) for key in inputs[num_controls_present:]], dtype=self.lib.int64)

    def apply(self, dm_state, differential_model_output):
        next_dm_state = dm_state + self.rescale_output_diff_model(differential_model_output)
        output = next_dm_state
        next_model_input = self.lib.gather_last(output, self.indices_output_to_input)
        return output, next_model_input, next_dm_state


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
