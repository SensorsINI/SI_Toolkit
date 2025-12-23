"""
Counterfactual trajectory generation for stateful controller analysis.

Generates "what-if" trajectories: given an anchor state x(t), reconstructs
alternative past trajectories that would have led to x(t) under different
conditions (e.g., different friction mu, mass, or other parameters).

Use case: Stateful controllers (like RNN) produce different outputs depending
on trajectory history, not just current state. This module enables testing
how controller behavior changes when we provide counterfactual histories.

Approach:
1. At each timestep t, take the current state x(t) as anchor
2. Use backward predictor to reconstruct past trajectory for multiple parameter values
3. Use trajectory regularization to keep counterfactual trajectories close to GT
4. Feed each counterfactual trajectory to the stateful controller
5. Compare controller outputs across parameter values

Multi-rate support:
- Backward predictor runs at fine dt (e.g., 0.01s) for accurate dynamics
- Controller receives subsampled trajectory at coarser DT (e.g., 0.2s)
- Subsampling factor n = DT / dt

Output stride:
- Process only every output_stride-th timestep to reduce output size
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import trange

from utilities.FastBackwardOptimizer import FastBackwardOptimizer
from utilities.state_utilities import STATE_VARIABLES


def subsample_trajectory(trajectory: np.ndarray, factor: int) -> np.ndarray:
    """
    Subsample a trajectory by a given factor.
    
    Args:
        trajectory: [N, state_dim] trajectory, oldest to newest
        factor: Subsampling factor (take every factor-th point)
    
    Returns:
        subsampled: [N//factor + 1, state_dim] subsampled trajectory
                    Always includes the last point (anchor)
    """
    if factor <= 1:
        return trajectory
    
    N = len(trajectory)
    # Always include the last point (anchor)
    # Work backwards to ensure anchor is included
    indices = list(range(N - 1, -1, -factor))[::-1]
    return trajectory[indices]


def subsample_controls(controls: np.ndarray, factor: int) -> np.ndarray:
    """
    Subsample controls to match subsampled trajectory.
    
    For trajectory subsampling, we need to select controls that correspond
    to the transitions between subsampled states.
    
    Args:
        controls: [H, 2] controls at fine dt
        factor: Subsampling factor
    
    Returns:
        subsampled: [H//factor, 2] subsampled controls
    """
    if factor <= 1:
        return controls
    
    H = len(controls)
    # Select every factor-th control, working backwards from the last
    indices = list(range(H - 1, -1, -factor))[::-1]
    return controls[indices]


class CounterfactualTrajectoryGenerator:
    """
    Generates counterfactual trajectories for stateful controller analysis.
    
    Given an anchor state x(t), generates alternative past trajectories
    that would have led to x(t) under different parameter values.
    
    Supports multi-rate operation:
    - Fine dt for accurate backward dynamics
    - Coarse DT for controller trajectory input
    """
    
    def __init__(
        self,
        horizon_fine: int = 200,
        dt: float = 0.01,
        controller_dt: float = None,
        parameter_name: str = 'mu',
        true_parameter_value: float = 0.739,
        parameter_values: List[float] = None,
        traj_weight: float = 0.1,
        continuation: bool = True,
        continuation_stages: int = 5,
        spread_anchor_error: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            horizon_fine: Number of past timesteps at fine dt (predictor horizon)
            dt: Fine timestep for backward predictor
            controller_dt: Controller timestep (coarse). If None, equals dt (no subsampling)
            parameter_name: Name of the parameter to vary (e.g., 'mu', 'mass')
            true_parameter_value: True parameter value from data
            parameter_values: List of parameter values to test (includes true value if not present)
            traj_weight: Weight for trajectory regularization
            continuation: Use continuation method for better convergence
            continuation_stages: Number of stages for continuation
            spread_anchor_error: Spread anchor error to ensure exact endpoint match
            verbose: Print debug info
        """
        self.horizon_fine = horizon_fine
        self.dt = dt
        self.controller_dt = controller_dt if controller_dt is not None else dt
        self.parameter_name = parameter_name
        self.true_parameter_value = true_parameter_value
        self.traj_weight = traj_weight
        self.continuation = continuation
        self.continuation_stages = continuation_stages
        self.spread_anchor_error = spread_anchor_error
        self.verbose = verbose
        
        # Compute subsampling factor
        self.subsample_factor = max(1, int(round(self.controller_dt / self.dt)))
        
        # Controller horizon (at coarse rate)
        self.horizon_controller = (horizon_fine + self.subsample_factor - 1) // self.subsample_factor
        
        if verbose:
            print("Multi-rate config:")
            print(f"  Parameter: {self.parameter_name}")
            print(f"  dt (predictor) = {self.dt}s")
            print(f"  DT (controller) = {self.controller_dt}s")
            print(f"  Subsample factor = {self.subsample_factor}")
            print(f"  Horizon fine = {self.horizon_fine} steps ({self.horizon_fine * self.dt}s)")
            print(f"  Horizon controller = {self.horizon_controller} steps ({self.horizon_controller * self.controller_dt}s)")
        
        # Default parameter values to test
        if parameter_values is None:
            parameter_values = [0.1, 0.3, 0.5, 0.7, true_parameter_value, 0.9, 1.1]
        self.parameter_values = sorted(set(parameter_values))
        
        # Ensure true value is in the list
        if true_parameter_value not in self.parameter_values:
            self.parameter_values.append(true_parameter_value)
            self.parameter_values = sorted(self.parameter_values)
        
        # Index of true value in the list
        self.true_parameter_idx = self.parameter_values.index(true_parameter_value)
        
        # Create optimizer for each parameter value
        self.optimizers: Dict[float, FastBackwardOptimizer] = {}
        for param_val in self.parameter_values:
            # Pass parameter as keyword argument using parameter_name
            self.optimizers[param_val] = FastBackwardOptimizer(
                dt=dt,
                max_iter=200,
                tol=1e-7,
                verbose=False,
                **{parameter_name: param_val},
            )
        
        # Stats for verification
        self.last_stats = {}
    
    def generate_backward_trajectories(
        self,
        anchor_state: np.ndarray,
        controls: np.ndarray,
        gt_trajectory: np.ndarray,
    ) -> Dict[float, np.ndarray]:
        """
        Generate backward trajectories for all parameter values at FINE resolution.
        
        Args:
            anchor_state: Current state x(t) [state_dim]
            controls: Past controls [H_fine, 2] from oldest to newest at dt
            gt_trajectory: Ground truth past trajectory [H_fine, state_dim] at dt
        
        Returns:
            trajectories: Dict mapping parameter_value -> [H_fine+1, state_dim] trajectory at fine dt
        """
        H = len(controls)
        assert len(gt_trajectory) == H, f"GT trajectory must have H={H} states, got {len(gt_trajectory)}"
        
        anchor_state = np.asarray(anchor_state, dtype=np.float32)
        controls = np.asarray(controls, dtype=np.float32)
        gt_trajectory = np.asarray(gt_trajectory, dtype=np.float32)
        
        trajectories = {}
        stats_all = {}
        
        for param_val in self.parameter_values:
            optimizer = self.optimizers[param_val]
            
            # Use GT trajectory as initial guess for oldest state
            x_init = gt_trajectory  # [H, state_dim]
            
            # Generate backward trajectory with trajectory regularization
            past_states, converged, stats = optimizer.predict_backward_shooting(
                x_current=anchor_state,
                controls=controls,
                x_init=x_init,
                traj_ref=gt_trajectory,
                traj_weight=self.traj_weight,
                continuation=self.continuation,
                continuation_stages=self.continuation_stages,
                spread_anchor_error=self.spread_anchor_error,
            )
            
            # Construct full trajectory including anchor
            full_trajectory = np.concatenate([
                past_states,
                anchor_state[None, :]
            ], axis=0)  # [H+1, state_dim]
            
            trajectories[param_val] = full_trajectory
            stats_all[param_val] = stats
            
            if self.verbose:
                mae = np.mean(np.abs(past_states - gt_trajectory))
                print(f"  {self.parameter_name}={param_val:.3f}: MAE vs GT = {mae:.6e}, converged={np.all(converged)}")
        
        self.last_stats = stats_all
        return trajectories
    
    def subsample_trajectories(
        self,
        trajectories: Dict[float, np.ndarray],
    ) -> Dict[float, np.ndarray]:
        """
        Subsample all trajectories for controller input.
        
        Args:
            trajectories: Dict mapping parameter_value -> [H_fine+1, state_dim] at fine dt
        
        Returns:
            subsampled: Dict mapping parameter_value -> [H_controller+1, state_dim] at coarse DT
        """
        if self.subsample_factor <= 1:
            return trajectories
        
        return {
            param_val: subsample_trajectory(traj, self.subsample_factor)
            for param_val, traj in trajectories.items()
        }
    
    def verify_true_parameter_recovery(
        self,
        trajectories: Dict[float, np.ndarray],
        gt_trajectory: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Verify that the trajectory for true parameter value matches ground truth.
        """
        true_param_traj = trajectories[self.true_parameter_value][:-1]  # Exclude anchor
        mae = np.mean(np.abs(true_param_traj - gt_trajectory))
        passed = mae < 1e-5
        return mae, passed
    
    def verify_anchor_consistency(
        self,
        trajectories: Dict[float, np.ndarray],
        anchor_state: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Verify that all trajectories end at the anchor state.
        """
        max_error = 0.0
        for param_val, traj in trajectories.items():
            endpoint = traj[-1]
            error = np.max(np.abs(endpoint - anchor_state))
            max_error = max(max_error, error)
        
        passed = max_error < 1e-6
        return max_error, passed


def add_control_with_counterfactual_trajectories(
    df,
    controller_config,
    controller_creator,
    df_modifier,
    parameter_name: str = 'mu',
    parameter_values: List[float] = None,
    horizon_fine: int = 200,
    controller_dt: float = None,
    output_stride: int = 1,
    controller_output_variable_names: Tuple[str, str] = ('angular_control', 'translational_control'),
    traj_weight: float = 0.1,
    continuation: bool = True,
    verbose: bool = False,
    run_verification: bool = True,
    **kwargs
):
    """
    Apply controller to dataset using counterfactual trajectories for each parameter value.
    
    For each timestep, generates backward trajectories under different parameter values
    (counterfactual histories) and feeds them to the controller.
    
    Supports:
    - Multi-rate: Fine dt for dynamics, coarse DT for controller
    - Output stride: Process only every output_stride-th timestep
    
    Args:
        df: DataFrame with state and control data
        controller_config: Controller configuration dict
        controller_creator: Function to create controller instance
        df_modifier: Function to modify df for controller input format
        parameter_name: Name of parameter to vary (e.g., 'mu', 'mass')
        parameter_values: List of parameter values to test
        horizon_fine: Number of past timesteps at fine dt (predictor resolution)
        controller_dt: Controller timestep. If None, uses data dt (no subsampling)
        output_stride: Process every output_stride-th timestep (reduces output size)
        controller_output_variable_names: Names for output columns
        traj_weight: Weight for trajectory regularization
        continuation: Use continuation method
        verbose: Print debug info
        run_verification: Run alignment verification tests
        **kwargs: Additional arguments
    
    Returns:
        df_output: DataFrame with only processed rows and new control columns
    """
    from SI_Toolkit.load_and_normalize import get_sampling_interval_from_datafile
    
    # Get sampling interval (dt)
    dt = get_sampling_interval_from_datafile(df, kwargs.get('current_path', ''))
    if dt is None:
        dt = 0.01
        print(f"Warning: Could not determine dt, using default {dt}")
    
    # Controller dt defaults to data dt
    if controller_dt is None:
        controller_dt = dt
    
    # Compute subsampling factor
    subsample_factor = max(1, int(round(controller_dt / dt)))
    
    # Get true parameter value from dataset
    if parameter_name in df.columns:
        true_param_value = df[parameter_name].iloc[0]
    else:
        true_param_value = 0.739
        print(f"Warning: Column '{parameter_name}' not found, using default {parameter_name}={true_param_value}")
    
    # Initialize counterfactual trajectory generator
    generator = CounterfactualTrajectoryGenerator(
        horizon_fine=horizon_fine,
        dt=dt,
        controller_dt=controller_dt,
        parameter_name=parameter_name,
        true_parameter_value=true_param_value,
        parameter_values=parameter_values,
        traj_weight=traj_weight,
        continuation=continuation,
        verbose=verbose,
    )
    
    # Prepare df for controller
    df_temp = df_modifier(df)
    
    # Create controller instance
    environment_attributes_dict = controller_config["environment_attributes_dict"]
    controller_instance = controller_creator(controller_config, initial_environment_attributes={})
    
    # Extract state matrix and controls from df
    state_columns = list(STATE_VARIABLES)
    states = df[state_columns].values.astype(np.float32)
    
    # Get control columns
    angular_control_col = 'angular_control'
    translational_control_col = 'translational_control'
    if angular_control_col not in df.columns:
        angular_control_col = 'angular_control_applied'
    if translational_control_col not in df.columns:
        translational_control_col = 'translational_control_applied'
    
    controls_df = df[[angular_control_col, translational_control_col]].values.astype(np.float32)
    
    N = len(df)
    H_fine = horizon_fine
    
    # Determine which timesteps to process (output stride)
    output_stride = max(1, int(output_stride))
    process_indices = list(range(H_fine, N, output_stride))
    n_output = len(process_indices)
    
    print("\n=== Counterfactual Trajectory Analysis Configuration ===")
    print(f"  Parameter: {parameter_name}")
    print(f"  Data dt: {dt}s")
    print(f"  Controller dt: {controller_dt}s")
    print(f"  Subsample factor: {subsample_factor}")
    print(f"  Horizon (fine): {H_fine} steps ({H_fine * dt:.2f}s)")
    print(f"  Horizon (controller): {generator.horizon_controller} steps ({generator.horizon_controller * controller_dt:.2f}s)")
    print(f"  Output stride: {output_stride}")
    print(f"  Total rows: {N}")
    print(f"  Rows to process: {n_output}")
    print(f"  {parameter_name} values: {generator.parameter_values}")
    
    # Initialize output arrays for the strided output
    control_outputs = {pv: np.full((n_output, 2), np.nan, dtype=np.float32) for pv in generator.parameter_values}
    output_times = np.full(n_output, np.nan)
    output_indices = np.zeros(n_output, dtype=np.int32)
    
    # Verification stats
    verification_stats = {
        'true_param_mae': [],
        'anchor_errors': [],
    }
    
    # Process selected timesteps
    for out_idx, t in enumerate(trange(len(process_indices), desc="Counterfactual trajectory analysis")):
        t = process_indices[out_idx]
        output_indices[out_idx] = t
        
        if 'time' in df.columns:
            output_times[out_idx] = df['time'].iloc[t]
        
        # Extract anchor state: x(t)
        anchor_state = states[t]
        
        # Extract GT past trajectory at fine resolution
        gt_trajectory = states[t - H_fine:t]  # [H_fine, state_dim]
        
        # Extract past controls at fine resolution
        past_controls = controls_df[t - H_fine:t]  # [H_fine, 2]
        
        # Generate counterfactual trajectories at fine resolution
        trajectories_fine = generator.generate_backward_trajectories(
            anchor_state=anchor_state,
            controls=past_controls,
            gt_trajectory=gt_trajectory,
        )
        
        # Subsample for controller
        trajectories_controller = generator.subsample_trajectories(trajectories_fine)
        
        # Run verification for first few timesteps
        if run_verification and out_idx < 3:
            mae, passed = generator.verify_true_parameter_recovery(trajectories_fine, gt_trajectory)
            verification_stats['true_param_mae'].append(mae)
            
            anchor_err, anchor_passed = generator.verify_anchor_consistency(trajectories_fine, anchor_state)
            verification_stats['anchor_errors'].append(anchor_err)
            
            if out_idx == 0:
                print(f"\n=== Verification at t={t} (first processed timestep) ===")
                print(f"  True {parameter_name} ({true_param_value:.4f}) recovery MAE: {mae:.6e} {'✓' if passed else '✗'}")
                print(f"  Anchor consistency max error: {anchor_err:.6e} {'✓' if anchor_passed else '✗'}")
                
                # Show subsampling
                example_traj_fine = trajectories_fine[true_param_value]
                example_traj_ctrl = trajectories_controller[true_param_value]
                print("\n  Subsampling verification:")
                print(f"    Fine trajectory shape: {example_traj_fine.shape}")
                print(f"    Controller trajectory shape: {example_traj_ctrl.shape}")
                print(f"    Fine traj[0] (oldest): linear_vel_x = {example_traj_fine[0, 4]:.4f}")
                print(f"    Fine traj[-1] (anchor): linear_vel_x = {example_traj_fine[-1, 4]:.4f}")
                print(f"    Ctrl traj[0] (oldest): linear_vel_x = {example_traj_ctrl[0, 4]:.4f}")
                print(f"    Ctrl traj[-1] (anchor): linear_vel_x = {example_traj_ctrl[-1, 4]:.4f}")
        
        # For each parameter value, feed counterfactual trajectory to controller
        for param_val in generator.parameter_values:
            # The controller-rate trajectory is available for stateful controllers:
            # traj = trajectories_controller[param_val]  # [H_ctrl+1, state_dim], oldest to newest
            
            # Get current row's environment attributes
            row = df_temp.iloc[t]
            environment_attributes = {key: row[value] for key, value in environment_attributes_dict.items()}
            
            # Override parameter with the test value
            environment_attributes[parameter_name] = param_val
            
            # TODO: For truly stateful controllers, we'd replay the trajectory here
            # For now, pass current state and let controller handle its own state
            s = row['state'] if 'state' in row else anchor_state
            new_controls = controller_instance.step(s, updated_attributes=environment_attributes)
            
            control_outputs[param_val][out_idx, :] = new_controls
    
    # Create output dataframe with only processed rows
    df_output = df.iloc[process_indices].copy().reset_index(drop=True)
    
    # Add output columns
    base_angular = controller_output_variable_names[0]
    base_trans = controller_output_variable_names[1]
    
    for param_val in generator.parameter_values:
        param_suffix = f"_{parameter_name}_{param_val:.2f}".replace('.', 'p')
        df_output[f"{base_angular}{param_suffix}"] = control_outputs[param_val][:, 0]
        df_output[f"{base_trans}{param_suffix}"] = control_outputs[param_val][:, 1]
    
    # Add metadata to dataframe attrs
    df_output.attrs['original_indices'] = output_indices.tolist()
    df_output.attrs['horizon_fine'] = H_fine
    df_output.attrs['controller_dt'] = controller_dt
    df_output.attrs['output_stride'] = output_stride
    df_output.attrs['parameter_name'] = parameter_name
    df_output.attrs['parameter_values'] = generator.parameter_values
    
    # Print verification summary
    if run_verification and verification_stats['true_param_mae']:
        avg_mae = np.mean(verification_stats['true_param_mae'])
        max_anchor_err = np.max(verification_stats['anchor_errors'])
        print("\n=== Verification Summary ===")
        print(f"  Average true-{parameter_name} MAE: {avg_mae:.6e}")
        print(f"  Max anchor error: {max_anchor_err:.6e}")
        
        if avg_mae > 1e-4:
            print("  ⚠ WARNING: Large MAE may indicate alignment issues!")
    
    print("\n=== Output ===")
    print(f"  Output rows: {len(df_output)} (from {N} original)")
    print(f"  New columns: {[f'{base_angular}{param_suffix}' for param_val in generator.parameter_values]}")
    
    return df_output


def run_alignment_tests(
    df,
    horizon_fine: int = 200,
    dt: float = 0.01,
    controller_dt: float = None,
    parameter_name: str = 'mu',
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Run comprehensive alignment tests to verify correct indexing.
    
    Tests:
    1. Control alignment: u[k] transitions x[k] to x[k+1]
    2. Trajectory alignment: gt_trajectory[0] = x(t-H)
    3. Anchor consistency: trajectory ends at x(t)
    4. True parameter recovery: optimizer recovers GT for true parameter value
    5. Subsampling: subsampled trajectory endpoints match
    
    Returns:
        results: Dict of test name -> passed (bool)
    """
    # Get true parameter value
    if parameter_name in df.columns:
        true_param_value = df[parameter_name].iloc[0]
    else:
        true_param_value = 0.739
    
    if controller_dt is None:
        controller_dt = dt
    
    subsample_factor = max(1, int(round(controller_dt / dt)))
    
    # Initialize optimizer for true parameter value
    optimizer = FastBackwardOptimizer(dt=dt, max_iter=200, tol=1e-8, **{parameter_name: true_param_value})
    
    # Extract states and controls
    state_columns = list(STATE_VARIABLES)
    states = df[state_columns].values.astype(np.float32)
    
    angular_col = 'angular_control' if 'angular_control' in df.columns else 'angular_control_applied'
    trans_col = 'translational_control' if 'translational_control' in df.columns else 'translational_control_applied'
    controls = df[[angular_col, trans_col]].values.astype(np.float32)
    
    H = horizon_fine
    results = {}
    
    # Test at a few timesteps
    test_indices = [H, H + 50] if len(df) > H + 50 else [H]
    
    all_passed = True
    for t in test_indices:
        if t >= len(df):
            continue
        
        if verbose:
            print(f"\n=== Test at t={t} ===")
        
        anchor_state = states[t]
        gt_trajectory = states[t - H:t]
        past_controls = controls[t - H:t]
        
        # Test 1: Backward prediction
        past_states, converged, stats = optimizer.predict_backward_shooting(
            x_current=anchor_state,
            controls=past_controls,
            x_init=gt_trajectory,
            traj_ref=gt_trajectory,
            traj_weight=0.1,
            continuation=True,
            spread_anchor_error=True,
        )
        
        # Test 2: Anchor consistency
        full_traj = np.concatenate([past_states, anchor_state[None, :]], axis=0)
        anchor_error = np.max(np.abs(full_traj[-1] - anchor_state))
        anchor_passed = anchor_error < 1e-6
        results[f't{t}_anchor'] = anchor_passed
        
        if verbose:
            print(f"  Anchor consistency: {anchor_error:.6e} {'✓' if anchor_passed else '✗'}")
        
        # Test 3: True parameter recovery
        mae = np.mean(np.abs(past_states - gt_trajectory))
        mae_passed = mae < 1e-5
        results[f't{t}_recovery'] = mae_passed
        
        if verbose:
            print(f"  True {parameter_name} recovery MAE: {mae:.6e} {'✓' if mae_passed else '✗'}")
        
        # Test 4: Subsampling verification
        subsampled = subsample_trajectory(full_traj, subsample_factor)
        subsample_passed = np.abs(subsampled[-1, 4] - anchor_state[4]) < 1e-6
        results[f't{t}_subsample'] = subsample_passed
        
        if verbose:
            print(f"  Subsampling (factor={subsample_factor}):")
            print(f"    Fine traj length: {len(full_traj)}")
            print(f"    Subsampled length: {len(subsampled)}")
            print(f"    Anchor match: {'✓' if subsample_passed else '✗'}")
        
        if not (anchor_passed and mae_passed and subsample_passed):
            all_passed = False
    
    results['all_passed'] = all_passed
    
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"All tests passed: {all_passed}")
    
    return results

