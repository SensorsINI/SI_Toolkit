# predictor_data_integrator.py
"""
Data Integrator Predictor - Tests correctness of training data derivatives.

This predictor takes pre-computed derivatives (D_* columns from training data)
as "external inputs" and integrates them to produce state trajectories.

For forward prediction:  x(t+1) = x(t) + dt * D_x(t)
For backward prediction: x(t-1) = x(t) + dt * D_x(t-1)  [with shifted controls]

This is useful for Brunton plot verification of training data quality.
If data is correctly prepared, integrating the recorded derivatives should
reconstruct the original trajectories perfectly.

Usage in Brunton test (config_testing.yml):
    predictors_specifications_testing: ['DI']  # or 'B:DI' for backward

The predictor reads derivative_features from config_training.yml outputs
and integrates them to reconstruct state trajectories.
"""

import os
import numpy as np
from typing import Optional, List

from SI_Toolkit.Predictors import template_predictor
from SI_Toolkit.computation_library import NumpyLibrary, TensorFlowLibrary, PyTorchLibrary
from SI_Toolkit.Predictors.autoregression import check_dimensions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class predictor_data_integrator(template_predictor):
    """
    Integrates pre-computed derivatives to verify training data correctness.
    
    Usage in Brunton test:
    - Configure with derivative_features matching D_* columns in your training data
    - The predictor will integrate these derivatives to produce state predictions
    - If training data is correct, predictions should match ground truth perfectly
    
    External inputs (Q) = derivatives from dataset (D_* columns)
    Initial state (s) = current state values
    Output = integrated trajectory
    """
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)
    
    def __init__(
        self,
        dt: float = 0.02,
        batch_size: int = 1,
        computation_library=None,
        derivative_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
        state_features: Optional[List[str]] = None,
        disable_individual_compilation: bool = False,
        variable_parameters=None,
        **kwargs
    ):
        """
        Initialize the data integrator predictor.
        
        Args:
            dt: Time step for integration. Positive for forward, negative for backward.
            batch_size: Batch size for predictions.
            computation_library: Computation library to use (Numpy, TF, or PyTorch).
            derivative_features: List of derivative feature names (e.g., ['D_pose_x', 'D_pose_y', ...]).
                These must exist as columns in your test dataset.
                If None, uses defaults from config_training.yml outputs.
            output_features: List of output state feature names (e.g., ['pose_x', 'pose_y', ...]).
                Derived from derivative_features by stripping 'D_' prefix if not provided.
            state_features: Initial state feature names. If None, uses output_features.
                This determines what columns are read from dataset as initial state.
            disable_individual_compilation: If True, skip compilation optimization.
            variable_parameters: Variable parameters (for API compatibility).
            
        Note on backward prediction:
            For backward prediction (negative dt), the same derivative columns (D_x) are used.
            The key relationships are:
            - Forward:  x(t+1) = x(t) + dt * D_x(t)
            - Backward: x(t-1) = x(t) - dt * D_x(t-1)
            The time alignment of derivatives is handled by the Brunton test.
        """
        super().__init__(batch_size=batch_size)
        
        self.dt = dt
        self.lib = computation_library if computation_library is not None else NumpyLibrary()
        
        # Set up feature mappings
        self._setup_features(derivative_features, output_features, state_features)
        
        # Build indices for gathering and scattering
        self._build_indices()
        
        self.output = None
        
        mode_str = "backward" if dt < 0 else "forward"
        print(f"[DataIntegrator] Mode: {mode_str}, dt={dt:.4f}")
        print(f"[DataIntegrator] External inputs (derivatives): {list(self.predictor_external_input_features)}")
        print(f"[DataIntegrator] Initial state: {list(self.predictor_initial_input_features)}")
        print(f"[DataIntegrator] Outputs: {list(self.predictor_output_features)}")
        
    def _setup_features(
        self,
        derivative_features: Optional[List[str]],
        output_features: Optional[List[str]],
        state_features: Optional[List[str]]
    ):
        """Set up input/output feature mappings."""
        
        # Default derivative features (from typical config_training.yml)
        # Use regular D_x columns for both forward and backward prediction.
        # The Brunton test's get_prediction handles index alignment internally.
        if derivative_features is None:
            derivative_features = [
                "D_pose_theta_cos",
                "D_pose_theta_sin",
                "D_pose_x",
                "D_pose_y",
                "D_linear_vel_x",
                "D_angular_vel_z",
                "D_slip_angle",
                "D_steering_angle",
            ]
        
        # Store base derivative names for deriving state/output features
        base_derivative_features = derivative_features.copy() if isinstance(derivative_features, list) else list(derivative_features)
        
        self.derivative_features = np.array(derivative_features)
        self.num_derivatives = len(derivative_features)
        
        # Derive output features by stripping D_ prefix and _-1 suffix from derivative names
        # Output features represent the state we're predicting (e.g., pose_x, not pose_x_-1)
        if output_features is None:
            output_features = []
            for f in base_derivative_features:
                out = f[2:] if f.startswith('D_') else f  # Strip 'D_' prefix
                if out.endswith('_-1'):
                    out = out[:-3]  # Strip '_-1' suffix
                output_features.append(out)
        self.output_feature_names = np.array(output_features)
        
        # State features for initial input - use base names (no _-1 suffix)
        # The initial state is where we START from, not a shifted version
        if state_features is None:
            state_features = output_features.copy() if isinstance(output_features, list) else list(output_features)
        
        # Override predictor features for Brunton test compatibility
        # These control what columns get extracted from the dataset
        
        # External inputs = derivatives (what we integrate)
        self.predictor_external_input_features = self.derivative_features
        
        # Initial input = state features we're tracking
        self.predictor_initial_input_features = np.array(state_features)
        
        # Output features = what we predict (same as initial for state integration)
        self.predictor_output_features = self.output_feature_names
        
        # Update num_states and num_control_inputs for API compatibility
        self.num_states = len(self.predictor_initial_input_features)
        self.num_control_inputs = len(self.predictor_external_input_features)
        
    def _build_indices(self):
        """Build index mappings between features."""
        
        # Map from output features to initial state indices (for gathering initial values)
        initial_features = self.predictor_initial_input_features
        
        self.output_to_initial_indices = []
        self.valid_output_mask = []  # Track which outputs have valid initial state
        
        for out_feat in self.output_feature_names:
            try:
                idx = np.where(initial_features == out_feat)[0][0]
                self.output_to_initial_indices.append(idx)
                self.valid_output_mask.append(True)
            except IndexError:
                # Feature not in initial state - use 0 as placeholder
                self.output_to_initial_indices.append(0)
                self.valid_output_mask.append(False)
        
        self.output_to_initial_indices = np.array(self.output_to_initial_indices)
        self.valid_output_mask = np.array(self.valid_output_mask)
        
    def predict(self, initial_state, Q) -> np.ndarray:
        """
        Predict trajectory by integrating derivatives.
        
        Args:
            initial_state: Initial state [batch_size x state_dim] or [state_dim]
            Q: Derivative inputs [batch_size x horizon x derivative_dim] or similar
               Each Q[:, t, :] contains derivatives D_x(t) to integrate.
               
        Returns:
            output: Integrated states [batch_size x (horizon+1) x output_dim]
                    First timestep is initial state, then integrated predictions.
        """
        # Convert inputs to numpy for simplicity (this predictor is for testing, not speed)
        if hasattr(initial_state, 'numpy'):
            initial_state = initial_state.numpy()
        initial_state = np.asarray(initial_state, dtype=np.float32)
        
        if hasattr(Q, 'numpy'):
            Q = Q.numpy()
        Q = np.asarray(Q, dtype=np.float32)
        
        # Ensure correct dimensions
        if initial_state.ndim == 1:
            initial_state = initial_state[np.newaxis, :]
        
        if Q.ndim == 2:
            Q = Q[np.newaxis, :, :]
        elif Q.ndim == 1:
            Q = Q[np.newaxis, np.newaxis, :]
        
        # Get dimensions
        batch_size = Q.shape[0]
        horizon = Q.shape[1]
        
        # Run integration
        output = self._integrate(initial_state, Q, batch_size, horizon)
        
        self.output = output
        return self.output
    
    def _integrate(self, initial_state, derivatives, batch_size, horizon):
        """
        Perform Euler integration over horizon steps.
        
        x(t+1) = x(t) + dt * D_x(t)
        
        Args:
            initial_state: [batch_size x state_dim]
            derivatives: [batch_size x horizon x derivative_dim]
            batch_size: Number of parallel predictions
            horizon: Number of integration steps
            
        Returns:
            outputs: [batch_size x (horizon+1) x output_dim]
        """
        num_outputs = len(self.output_feature_names)
        
        # Extract initial values for outputs from initial state
        # Use output_to_initial_indices to map from initial_state to output order
        if np.all(self.valid_output_mask):
            # All outputs have corresponding initial state features
            current_state = initial_state[:, self.output_to_initial_indices].copy()
        else:
            # Some outputs missing - initialize to zeros and fill valid ones
            current_state = np.zeros((batch_size, num_outputs), dtype=np.float32)
            for i, (idx, valid) in enumerate(zip(self.output_to_initial_indices, self.valid_output_mask)):
                if valid:
                    current_state[:, i] = initial_state[:, idx]
        
        # Pre-allocate output array: [batch_size, horizon+1, num_outputs]
        outputs = np.zeros((batch_size, horizon + 1, num_outputs), dtype=np.float32)
        outputs[:, 0, :] = current_state
        
        # Integration loop
        for t in range(horizon):
            # Get derivatives at this timestep
            d_state = derivatives[:, t, :]  # [batch_size x derivative_dim]
            
            # Euler integration: x(t+1) = x(t) + dt * dx/dt
            current_state = current_state + self.dt * d_state
            
            outputs[:, t + 1, :] = current_state
        
        return outputs
    
    def update_internal_state(self, Q=None, s=None):
        """No internal state to update for this predictor."""
        pass
    
    def reset(self):
        """Reset predictor state."""
        self.output = None


class predictor_data_integrator_flexible(predictor_data_integrator):
    """
    Flexible version that auto-configures from training config.
    
    Reads derivative features from config_training.yml outputs
    and sets up integration accordingly.
    
    The predictor expects:
    - External inputs (Q): D_* columns from dataset matching training outputs
    - Initial state (s): State columns matching the integrated features (D_* with prefix stripped)
    
    Usage in Brunton test:
        predictors_specifications_testing: ['DI']      # Forward integration
        predictors_specifications_testing: ['B:DI']    # Backward integration
        
    This will verify that integrating the D_* columns reproduces the state trajectory,
    confirming that the training data derivatives are computed correctly.
    
    Configuration in config_predictors.yml:
        data_integrator_default:
            predictor_type: "data_integrator"
            computation_library_name: "Numpy"
            config_path: "SI_Toolkit_ASF/config_training.yml"
            config_section: "training_nn_physical_model"  # Section containing D_* outputs
    """
    
    def __init__(
        self,
        dt: float = 0.02,
        batch_size: int = 1,
        computation_library=None,
        config_path: str = 'SI_Toolkit_ASF/config_training.yml',
        config_section: str = 'training_nn_physical_model',
        use_shifted_features: bool = False,
        disable_individual_compilation: bool = False,
        variable_parameters=None,
        **kwargs
    ):
        """
        Initialize from training config.
        
        Args:
            dt: Time step for integration (negative for backward prediction).
            batch_size: Batch size.
            computation_library: Computation library.
            config_path: Path to config_training.yml.
            config_section: Section in config to load from. Use 'training_nn_physical_model'
                for D_* derivative features, or 'training_default' for other setups.
            use_shifted_features: If True, use _-1 shifted derivatives for backward prediction.
                This appends '_-1' to derivative column names (e.g., D_pose_x_-1).
            disable_individual_compilation: Skip compilation.
            variable_parameters: Variable parameters (for API compatibility).
        """
        # Load config to get derivative features
        from SI_Toolkit.load_and_normalize import load_yaml
        
        derivative_features = None
        state_features = None
        
        try:
            config = load_yaml(config_path)
            
            # Try specified config section first, then fall back to training_default
            training_config = config.get(config_section, {})
            if not training_config:
                print(f"[DataIntegrator] Config section '{config_section}' not found, trying 'training_default'")
                training_config = config.get('training_default', {})
            
            outputs = training_config.get('outputs', [])
            
            # Filter to only D_ features
            derivative_features = [f for f in outputs if f.startswith('D_')]
            
            # For Brunton testing, we need the integrated state features as initial state.
            # Strip both 'D_' prefix and '_-1' suffix to get base state names.
            # E.g., 'D_pose_x_-1' -> 'pose_x'
            if derivative_features:
                state_features = []
                for f in derivative_features:
                    base = f[2:]  # Strip 'D_' prefix
                    if base.endswith('_-1'):
                        base = base[:-3]  # Strip '_-1' suffix
                    state_features.append(base)
            
            if not derivative_features:
                print(f"[DataIntegrator] Warning: No D_* features in '{config_section}' outputs.")
                print(f"[DataIntegrator] Available outputs: {outputs}")
                print(f"[DataIntegrator] Using default derivative features.")
                derivative_features = None
            else:
                print(f"[DataIntegrator] Loaded {len(derivative_features)} derivative features from '{config_section}':")
                print(f"[DataIntegrator]   Base derivatives: {derivative_features}")
                if use_shifted_features:
                    print(f"[DataIntegrator]   Will use shifted (_-1) versions for backward prediction")
                print(f"[DataIntegrator]   States: {state_features}")
                
        except Exception as e:
            print(f"[DataIntegrator] Could not load config from {config_path}: {e}")
            print("[DataIntegrator] Using default derivative features.")
        
        super().__init__(
            dt=dt,
            batch_size=batch_size,
            computation_library=computation_library,
            derivative_features=derivative_features,
            state_features=state_features,
            use_shifted_features=use_shifted_features,
            disable_individual_compilation=disable_individual_compilation,
            variable_parameters=variable_parameters,
            **kwargs
        )


if __name__ == '__main__':
    """Test the data integrator predictor."""
    
    print("=" * 60)
    print("Testing predictor_data_integrator")
    print("=" * 60)
    
    # Create predictor with simple test features
    derivative_features = ['D_x', 'D_y']
    state_features = ['x', 'y']
    
    predictor = predictor_data_integrator(
        dt=0.1,
        batch_size=2,
        derivative_features=derivative_features,
        output_features=['x', 'y'],
        state_features=state_features,
    )
    
    # Test data: 
    # Initial state: x=0, y=0
    # Derivatives: constant dx/dt=1, dy/dt=2
    # After 5 steps with dt=0.1: x=0.5, y=1.0
    
    initial_state = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)  # batch_size=2
    derivatives = np.ones((2, 5, 2), dtype=np.float32)  # horizon=5
    derivatives[:, :, 0] = 1.0  # dx/dt = 1
    derivatives[:, :, 1] = 2.0  # dy/dt = 2
    
    output = predictor.predict(initial_state, derivatives)
    
    print(f"\nInitial state:\n{initial_state}")
    print(f"\nDerivatives (constant): dx/dt=1, dy/dt=2")
    print(f"\nPredicted trajectory (dt=0.1, horizon=5):")
    print(f"Batch 0:\n{output[0]}")
    print(f"Batch 1:\n{output[1]}")
    
    # Verify
    expected_final_batch0 = np.array([0.5, 1.0])  # 0 + 5*0.1*1, 0 + 5*0.1*2
    expected_final_batch1 = np.array([1.5, 2.0])  # 1 + 5*0.1*1, 1 + 5*0.1*2
    
    print(f"\nExpected final (batch 0): {expected_final_batch0}")
    print(f"Expected final (batch 1): {expected_final_batch1}")
    print(f"Actual final (batch 0): {output[0, -1, :]}")
    print(f"Actual final (batch 1): {output[1, -1, :]}")
    
    assert np.allclose(output[0, -1, :], expected_final_batch0), "Batch 0 mismatch!"
    assert np.allclose(output[1, -1, :], expected_final_batch1), "Batch 1 mismatch!"
    
    print("\n✓ Basic integration test passed!")
    
    # Test backward prediction (negative dt)
    print("\n" + "=" * 60)
    print("Testing backward prediction (negative dt)")
    print("=" * 60)
    
    predictor_backward = predictor_data_integrator(
        dt=-0.1,  # Negative dt for backward prediction
        batch_size=1,
        derivative_features=derivative_features,
        output_features=['x', 'y'],
        state_features=state_features,
    )
    
    # Start at x=0.5, y=1.0, with same derivatives
    # Going backward should recover x=0, y=0
    initial_state_backward = np.array([[0.5, 1.0]], dtype=np.float32)
    derivatives_backward = np.ones((1, 5, 2), dtype=np.float32)
    derivatives_backward[:, :, 0] = 1.0
    derivatives_backward[:, :, 1] = 2.0
    
    output_backward = predictor_backward.predict(initial_state_backward, derivatives_backward)
    
    print(f"\nInitial state: {initial_state_backward}")
    print(f"Backward trajectory:\n{output_backward[0]}")
    print(f"\nExpected final (going back): [0, 0]")
    print(f"Actual final: {output_backward[0, -1, :]}")
    
    assert np.allclose(output_backward[0, -1, :], [0.0, 0.0], atol=1e-6), "Backward mismatch!"
    
    print("\n✓ Backward prediction test passed!")
    print("\n✓ All tests passed!")

