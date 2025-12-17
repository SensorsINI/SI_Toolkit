# predictor_backward_optimizer.py
#
# Wraps the BackwardPredictor for use in Brunton tests.
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
    - 'optimizer': Use optimizer-based BackwardPredictor (optimization only)
    - 'hybrid': Use network as initial guess, then optimize (future)
    
    Note: The optimizer only uses 2 control inputs (angular_control, translational_control).
    If the input controls have more features (e.g., including mu), only the first 2 are used.
    """
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary)

    # The optimizer only uses these 2 control inputs (indices 0 and 2 in standard ordering)
    # Standard ordering: [angular_control, mu, translational_control]
    # We need indices 0 (angular_control) and 2 (translational_control)
    OPTIMIZER_CONTROL_INDICES = [0, 2]  # angular_control, translational_control

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
        self._backward_predictor = None
        self._neural_predictor = None
        
        if self.forged_history_mode in ("optimizer", "hybrid"):
            # Lazy import to avoid TF import unless needed
            from utilities.BackwardPredictor import BackwardPredictor
            self._backward_predictor = BackwardPredictor()
            self._backward_predictor.configure(batch_size=batch_size, dt=abs(dt) if dt else None)
        
        if self.forged_history_mode in ("network", "hybrid"):
            # Initialize neural network predictor for network/hybrid modes
            self._init_neural_predictor(batch_size, dt, mode, disable_individual_compilation, model_name, path_to_model)
        
        # Store for diagnostics
        self._last_converged = True

    def _init_neural_predictor(self, batch_size, dt, mode, disable_individual_compilation, model_name=None, path_to_model=None):
        """Initialize the neural network predictor for network mode.
        
        Args:
            batch_size: Batch size for the predictor
            dt: Time step
            mode: Prediction mode
            disable_individual_compilation: Whether to disable individual compilation
            model_name: Name of the neural network model (from config_predictors.yml or kwargs)
            path_to_model: Path to the model directory (from config_predictors.yml or kwargs)
        """
        from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
        
        # Require model_name and path_to_model from config - no silent fallbacks
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

    def _extract_optimizer_controls(self, Q: np.ndarray) -> np.ndarray:
        """
        Extract the 2 control inputs that the optimizer uses from the full control array.
        
        Args:
            Q: Control array [..., control_dim] where control_dim >= 2
            
        Returns:
            Controls array [..., 2] with only angular_control and translational_control
        """
        control_dim = Q.shape[-1]
        if control_dim == 2:
            # Already correct dimension
            return Q
        elif control_dim == 3:
            # Standard case: [angular_control, mu, translational_control]
            # Extract indices 0 and 2
            return Q[..., self.OPTIMIZER_CONTROL_INDICES]
        else:
            # Fallback: take first 2
            return Q[..., :2]

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
            # Use neural network directly (same as B:Dense-...)
            return self._predict_network(initial_state, Q)
        elif self.forged_history_mode == "optimizer":
            # Use optimizer-based backward predictor
            return self._predict_optimizer(initial_state, Q)
        elif self.forged_history_mode == "hybrid":
            # Use neural network as initial guess, then refine with optimizer
            return self._predict_hybrid(initial_state, Q)
        else:
            raise ValueError(f"Unknown forged_history_mode: {self.forged_history_mode}")

    def _predict_network(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Predict using neural network only (same as B:Dense-...)."""
        if self._neural_predictor is None:
            raise RuntimeError("Neural predictor not initialized. Set FORGED_HISTORY_MODE='network'")
        return self._neural_predictor.predict(initial_state, Q)

    def _predict_optimizer(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Predict using optimizer-based backward predictor (sequential, not batch)."""
        if self._backward_predictor is None:
            raise RuntimeError("Backward predictor not initialized. Set FORGED_HISTORY_MODE='optimizer'")
        
        # Handle dimensions
        if initial_state.ndim == 1:
            initial_state = initial_state[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        if Q.ndim == 2:
            Q = Q[None, :, :]
        
        # Extract the 2 controls the optimizer uses
        Q_optimizer = self._extract_optimizer_controls(Q)
            
        batch_size = initial_state.shape[0]
        horizon = Q_optimizer.shape[1]
        state_dim = initial_state.shape[1]
        
        # Track convergence statistics
        num_converged = 0
        num_failed = 0
        
        # Process each trajectory sequentially (optimizer doesn't support batch)
        outputs = []
        iterator = trange(batch_size, desc="BP optimizer", leave=False) if batch_size > 1 else range(batch_size)
        
        for b in iterator:
            state_b = initial_state[b]
            controls_b = Q_optimizer[b]  # [horizon, 2]
            
            # Run optimizer-based backward prediction
            past_states = self._backward_predictor.predict(state_b, controls_b)
            
            if past_states is None:
                # Prediction failed - return NaN trajectory
                self._last_converged = False
                num_failed += 1
                past_states = np.full((horizon, state_dim), np.nan, dtype=np.float32)
            else:
                self._last_converged = True
                num_converged += 1
            
            # Prepend initial state to match neural predictor output format
            # Output: [initial_state, past_states...] = [horizon+1, state_dim]
            trajectory = np.concatenate([state_b[None, :], past_states], axis=0)
            outputs.append(trajectory)
        
        # Print summary if there were failures
        if num_failed > 0:
            print(f"[BP optimizer] Converged: {num_converged}/{batch_size}, Failed: {num_failed}")
        
        output = np.stack(outputs, axis=0)  # [batch_size, horizon+1, state_dim]
        
        if squeeze_batch:
            output = output[0]
            
        return output

    def _predict_hybrid(self, initial_state: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Predict using neural network as initial guess, then refine with optimizer.
        
        This combines the best of both worlds:
        - Neural network provides a fast initial trajectory estimate
        - Optimizer refines it to satisfy physics constraints
        """
        if self._neural_predictor is None:
            raise RuntimeError("Neural predictor not initialized for hybrid mode")
        if self._backward_predictor is None:
            raise RuntimeError("Backward predictor not initialized for hybrid mode")
        
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
        
        # Step 1: Get neural network prediction (uses full controls including mu)
        neural_output = self._neural_predictor.predict(initial_state, Q)
        # neural_output: [batch, horizon+1, state_dim] - includes initial state at index 0
        
        # Extract the 2 controls the optimizer uses
        Q_optimizer = self._extract_optimizer_controls(Q)
        
        # Track convergence
        num_converged = 0
        num_failed = 0
        
        # Process each trajectory
        outputs = []
        iterator = trange(batch_size, desc="BP hybrid", leave=False) if batch_size > 1 else range(batch_size)
        
        for b in iterator:
            state_b = initial_state[b]
            controls_b = Q_optimizer[b]  # [horizon, 2]
            
            # Extract neural prediction for this batch (exclude initial state)
            # neural_output[b] is [horizon+1, state_dim] with initial state at [0]
            neural_traj_b = neural_output[b, 1:, :]  # [horizon, state_dim] oldest→newest past states
            
            # Run optimizer with neural prediction as initial guess
            past_states = self._backward_predictor.predict(state_b, controls_b, X_init=neural_traj_b)
            
            if past_states is None:
                # Refinement failed - fall back to neural prediction
                self._last_converged = False
                num_failed += 1
                # Use neural prediction as fallback
                past_states = neural_traj_b
            else:
                self._last_converged = True
                num_converged += 1
            
            # Prepend initial state to match output format
            trajectory = np.concatenate([state_b[None, :], past_states], axis=0)
            outputs.append(trajectory)
        
        # Print summary
        if batch_size > 1:
            print(f"[BP hybrid] Refined: {num_converged}/{batch_size}, Fallback to neural: {num_failed}")
        
        output = np.stack(outputs, axis=0)  # [batch_size, horizon+1, state_dim]
        
        if squeeze_batch:
            output = output[0]
            
        return output

    def predict_core(self, initial_state, Q):
        """Core prediction (same as predict for this predictor)."""
        return self.predict(initial_state, Q)

    def update_internal_state(self, Q0=None, s=None):
        """Update internal state (feeds history buffers)."""
        if s is not None and Q0 is not None:
            # Flatten control if needed
            if Q0.ndim > 2:
                Q0 = Q0[:, 0, :]  # Take first timestep
            for b in range(s.shape[0] if s.ndim > 1 else 1):
                state_b = s[b] if s.ndim > 1 else s
                ctrl_b = Q0[b] if Q0.ndim > 1 else Q0
                self._backward_predictor.update(ctrl_b, state_b)

    def reset(self):
        """Reset internal state."""
        self._backward_predictor.previous_control_inputs = []
        self._backward_predictor.previous_measured_states = []
        self._backward_predictor.counter = 0
        self._backward_predictor._warmed_once = False

    @property
    def converged(self) -> bool:
        """Whether the last prediction converged."""
        return self._last_converged







