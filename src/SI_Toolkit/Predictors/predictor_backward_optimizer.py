# predictor_backward_optimizer.py
#
# Backward predictor for Brunton tests.
# Supports multiple modes: network, optimizer, hybrid.

import numpy as np
from tqdm import trange
from SI_Toolkit.Predictors import template_predictor
from SI_Toolkit.computation_library import NumpyLibrary, TensorFlowLibrary


class predictor_backward_optimizer(template_predictor):
    """
    Predictor that wraps backward trajectory prediction for Brunton tests.
    
    Supports three modes based on Settings.FORGED_HISTORY_MODE:
    - 'network': Use neural network only (same as B:Dense-..., no optimization)
    - 'optimizer': Use FastBackwardOptimizer (sequential L-BFGS, very fast and accurate)
    - 'hybrid': Use network as initial guess, then refine with optimizer
    
    Note: The optimizer uses 2 control inputs (angular_control, translational_control).
    If the input controls have more features (e.g., including mu), the mu is extracted
    and used for the friction coefficient.
    """
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary)

    # Control indices: [angular_control, mu, translational_control]
    ANGULAR_CONTROL_IDX = 0
    MU_IDX = 1
    TRANSLATIONAL_CONTROL_IDX = 2

    def __init__(
        self,
        dt=None,
        batch_size=None,
        variable_parameters=None,
        disable_individual_compilation=False,
        mode=None,
        model_name=None,
        path_to_model=None,
        **kwargs
    ):
        super().__init__(batch_size=batch_size)
        self.dt = dt
        self.prediction_mode = mode  # autoregressive_backward etc.
        self.lib = NumpyLibrary()
        
        # Store model configuration for neural predictor
        self._model_name = model_name
        self._path_to_model = path_to_model
        
        # Get the forged history mode from Settings
        from utilities.Settings import Settings
        self.forged_history_mode = getattr(Settings, "FORGED_HISTORY_MODE", "optimizer")
        
        # Initialize predictors based on mode
        self._fast_optimizer = None
        self._neural_predictor = None
        
        if self.forged_history_mode in ("optimizer", "hybrid"):
            # Use the new FastBackwardOptimizer
            from utilities.FastBackwardOptimizer import FastBackwardPredictor
            mu = getattr(Settings, "FRICTION_FOR_CONTROLLER", None)
            self._fast_optimizer = FastBackwardPredictor(
                dt=abs(dt) if dt else 0.01,
                mu=mu,
                max_iter_per_step=30,
                tol=1e-5,
                method='L-BFGS-B',
                verbose=False,
            )
        
        if self.forged_history_mode in ("network", "hybrid"):
            # Initialize neural network predictor for network/hybrid modes
            self._init_neural_predictor(batch_size, dt, mode, disable_individual_compilation, model_name, path_to_model)
        
        if self.forged_history_mode == "optimizer":
            # Set feature info for optimizer-only mode
            # Optimizer uses full 10-state vector and 3 control inputs
            from utilities.state_utilities import STATE_VARIABLES
            self.predictor_initial_input_features = np.array(STATE_VARIABLES)
            self.predictor_external_input_features = np.array(['angular_control', 'mu', 'translational_control'])
            self.predictor_output_features = np.array(STATE_VARIABLES)
        
        # Store for diagnostics
        self._last_converged = True
        self._last_stats = {}

    def _init_neural_predictor(self, batch_size, dt, mode, disable_individual_compilation, model_name=None, path_to_model=None):
        """Initialize the neural network predictor for network mode."""
        from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
        
        if model_name is None:
            raise ValueError(
                "model_name is required for backward_optimizer predictor in 'network' or 'hybrid' mode. "
                "Please set 'model_name' in config_predictors.yml under backward_optimizer_default."
            )
        if path_to_model is None:
            raise ValueError(
                "path_to_model is required for backward_optimizer predictor in 'network' or 'hybrid' mode. "
                "Please set 'path_to_model' in config_predictors.yml under backward_optimizer_default."
            )
        
        self._neural_predictor = predictor_autoregressive_neural(
            model_name=model_name,
            path_to_model=path_to_model,
            dt=dt,
            batch_size=batch_size,
            mode=mode,
            disable_individual_compilation=disable_individual_compilation,
        )
        
        # Copy feature info from neural predictor
        self.predictor_initial_input_features = self._neural_predictor.predictor_initial_input_features
        self.predictor_external_input_features = self._neural_predictor.predictor_external_input_features
        self.predictor_output_features = self._neural_predictor.predictor_output_features
        self.lib = self._neural_predictor.lib

    def _extract_controls_and_mu(self, Q: np.ndarray):
        """
        Extract control inputs and mu from the control array.
        
        Args:
            Q: Control array [..., control_dim] where control_dim >= 2
            
        Returns:
            controls: [angular_control, translational_control] array
            mu: Mean friction coefficient (or None if not in controls)
        """
        control_dim = Q.shape[-1]
        
        if control_dim == 2:
            # Already [angular_control, translational_control]
            return Q, None
        elif control_dim == 3:
            # [angular_control, mu, translational_control]
            controls = np.stack([
                Q[..., self.ANGULAR_CONTROL_IDX],
                Q[..., self.TRANSLATIONAL_CONTROL_IDX]
            ], axis=-1)
            mu = float(np.mean(Q[..., self.MU_IDX]))
            return controls, mu
        else:
            # Fallback: take first and last as angular and translational
            controls = np.stack([Q[..., 0], Q[..., -1]], axis=-1)
            return controls, None

    def predict(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Predict backward trajectory.
        
        Args:
            initial_state: Current state [batch_size, state_dim] or [state_dim]
            Q: Control sequence [batch_size, horizon, control_dim] or [horizon, control_dim]
            
        Returns:
            Backward states [batch_size, horizon+1, state_dim], including initial state
        """
        if self.forged_history_mode == "network":
            return self._predict_network(initial_state, Q)
        elif self.forged_history_mode == "optimizer":
            return self._predict_optimizer(initial_state, Q)
        elif self.forged_history_mode == "hybrid":
            return self._predict_hybrid(initial_state, Q)
        else:
            raise ValueError(f"Unknown forged_history_mode: {self.forged_history_mode}")

    def _predict_network(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Predict using neural network only."""
        if self._neural_predictor is None:
            raise RuntimeError("Neural predictor not initialized. Set FORGED_HISTORY_MODE='network'")
        return self._neural_predictor.predict(initial_state, Q)

    def _predict_optimizer(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Predict using FastBackwardOptimizer (sequential L-BFGS)."""
        if self._fast_optimizer is None:
            raise RuntimeError("Fast optimizer not initialized. Set FORGED_HISTORY_MODE='optimizer'")
        
        # Handle dimensions
        if initial_state.ndim == 1:
            initial_state = initial_state[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        if Q.ndim == 2:
            Q = Q[None, :, :]
            
        batch_size = initial_state.shape[0]
        horizon = Q.shape[1]
        state_dim = initial_state.shape[1]
        
        # Extract controls and mu
        controls, mu = self._extract_controls_and_mu(Q)
        
        # Update mu if extracted
        if mu is not None:
            self._fast_optimizer.optimizer.set_mu(mu)
        
        # Process each trajectory
        outputs = []
        all_converged = True
        
        iterator = trange(batch_size, desc="BP optimizer", leave=False) if batch_size > 10 else range(batch_size)
        
        for b in iterator:
            state_b = initial_state[b]
            # Controls from prepare_predictor_inputs are [newest, ..., oldest]
            # But optimizer expects [oldest, ..., newest], so reverse them
            controls_b = controls[b, ::-1, :]  # [horizon, 2] reversed
            
            # Run fast optimizer
            past_states = self._fast_optimizer.predict(state_b, controls_b)
            
            if past_states is None:
                self._last_converged = False
                all_converged = False
                past_states = np.full((horizon, state_dim), np.nan, dtype=np.float32)
            else:
                self._last_converged = True
            
            # Prepend initial state: [initial_state, past_states...]
            # past_states is [horizon, state_dim] oldest to newest
            # Output should be [horizon+1, state_dim] with initial_state at [0] and oldest past at [-1]
            trajectory = np.concatenate([state_b[None, :], past_states[::-1, :]], axis=0)
            outputs.append(trajectory)
        
        self._last_stats = self._fast_optimizer.last_stats
        if not all_converged and batch_size > 1:
            n_failed = batch_size - sum(1 for o in outputs if not np.any(np.isnan(o)))
            print(f"[BP optimizer] {n_failed}/{batch_size} trajectories failed")
        
        output = np.stack(outputs, axis=0)  # [batch_size, horizon+1, state_dim]
        
        if squeeze_batch:
            output = output[0]
            
        return output

    def _predict_hybrid(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Predict using NN trajectory + per-step optimizer refinement.
        
        Approach:
        1. Run NN for full trajectory (efficient batch call)
        2. For each step: use NN's prediction as initial guess for optimizer
        3. Each step is refined to satisfy dynamics exactly
        
        This combines NN's speed with optimizer's dynamical consistency.
        """
        if self._neural_predictor is None:
            raise RuntimeError("Neural predictor not initialized for hybrid mode")
        if self._fast_optimizer is None:
            raise RuntimeError("Fast optimizer not initialized for hybrid mode")
        
        # Handle dimensions
        if initial_state.ndim == 1:
            initial_state = initial_state[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        if Q.ndim == 2:
            Q = Q[None, :, :]
        
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]
        horizon = Q.shape[1]
        
        # Step 1: Get full NN trajectory prediction (efficient batch call)
        neural_output = self._neural_predictor.predict(initial_state, Q)
        # neural_output: [batch, horizon+1, nn_state_dim]
        
        # Extract controls and mu for optimizer
        controls, mu = self._extract_controls_and_mu(Q)
        if mu is not None:
            self._fast_optimizer.optimizer.set_mu(mu)
        
        # Get state augmenter to convert NN output to full state
        from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import StateAugmenter
        nn_features = list(self._neural_predictor.predictor_output_features)
        from utilities.state_utilities import STATE_VARIABLES
        augmenter = StateAugmenter(nn_features, lib=None, target_features=STATE_VARIABLES,
                                    strip_derivative_prefix=True)
        
        # Also need to augment anchor states (initial_state) to 10 dimensions
        # The input state may be 8-dim from NN or 10-dim from ODE predictor
        if state_dim != len(STATE_VARIABLES):
            # Need separate augmenter for the initial state (not derivative outputs)
            from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import augment_states_numpy
            initial_state_aug, _ = augment_states_numpy(
                initial_state[:, None, :], 
                list(self._neural_predictor.predictor_initial_input_features),
                STATE_VARIABLES,
                verbose=False
            )
            initial_state_aug = initial_state_aug[:, 0, :]  # [batch, 10]
        else:
            initial_state_aug = initial_state
        
        outputs = []
        all_stats = {'n_refined': 0, 'n_failed': 0}
        iterator = trange(batch_size, desc="BP hybrid", leave=False) if batch_size > 10 else range(batch_size)
        
        for b in iterator:
            state_b = initial_state_aug[b]  # Use augmented state [10]
            # Controls from prepare_predictor_inputs are [newest, ..., oldest]
            # But optimizer expects [oldest, ..., newest], so reverse them
            controls_b = controls[b, ::-1, :]
            
            # Get NN trajectory (exclude initial state)
            nn_traj_b = neural_output[b, 1:, :]  # [horizon, nn_state_dim], newest→oldest
            
            # Augment to full 10-state vector
            # augment_to_target_order returns (augmented, features) tuple
            nn_traj_b_aug, _ = augmenter.augment_to_target_order(
                nn_traj_b[None, :, :], verbose=False
            )
            nn_traj_b_aug = nn_traj_b_aug[0]  # [horizon, 10]
            
            # Reverse to get oldest→newest order for optimizer
            nn_traj_oldest_first = nn_traj_b_aug[::-1, :]  # [horizon, state_dim]
            
            # Step 2: Per-step refinement using NN predictions as initial guesses
            full_state_dim = len(STATE_VARIABLES)
            past_states = np.zeros((horizon, full_state_dim), dtype=np.float32)
            x = state_b.copy()  # Already augmented to 10-dim
            
            # Step-by-step refinement using NN predictions as initial guesses
            for h in range(horizon):
                u = controls_b[horizon - 1 - h]
                nn_guess = nn_traj_oldest_first[horizon - 1 - h]
                x_refined, converged = self._fast_optimizer.optimizer.single_step_backward(
                    x, u, x_init=nn_guess, max_iter=10
                )
                if converged:
                    all_stats['n_refined'] += 1
                else:
                    all_stats['n_failed'] += 1
                past_states[horizon - 1 - h] = x_refined
                x = x_refined
            
            trajectory = np.concatenate([state_b[None, :], past_states[::-1, :]], axis=0)
            outputs.append(trajectory)
        
        self._last_stats = all_stats
        output = np.stack(outputs, axis=0)
        if squeeze_batch:
            output = output[0]
        return output

    def predict_core(self, initial_state, Q):
        """Core prediction (same as predict for this predictor)."""
        return self.predict(initial_state, Q)

    def update_internal_state(self, Q0=None, s=None):
        """Update internal state (not used for this predictor)."""
        pass

    def reset(self):
        """Reset internal state."""
        pass

    @property
    def converged(self) -> bool:
        """Whether the last prediction converged."""
        return self._last_converged

    @property
    def last_stats(self) -> dict:
        """Statistics from last prediction."""
        return self._last_stats
