import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from Control_Toolkit.others.globals_and_utils import get_controller_name, import_controller_by_name

from scipy.integrate import nquad
from scipy.stats import qmc
from math import ceil

from scipy.signal import savgol_filter
import numdifftools as nd
from typing import Any, Dict, List, Tuple
import warnings
from copy import deepcopy


def controller_creator(controller_config, initial_environment_attributes):
    controller_name = controller_config['controller_name']
    optimizer_name = controller_config.get('optimizer_name', None)

    environment_name = controller_config['environment_name']
    action_space = controller_config['action_space']


    controller_name, _ = get_controller_name(
        controller_name=controller_name
    )

    controller_class = import_controller_by_name(controller_name)

    # Initialize a controller instance
    controller_instance = controller_class(
        environment_name=environment_name,
        initial_environment_attributes=initial_environment_attributes,
        control_limits=(action_space.low, action_space.high),
    )

    # Configure the controller
    if hasattr(controller_instance, 'has_optimizer') and controller_instance.has_optimizer:
        controller_instance.configure(optimizer_name)
    else:
        controller_instance.configure()

    return controller_instance


def df_modifier(df):
    return df


def add_control_along_trajectories(
        df,
        controller_config,
        controller_creator=controller_creator,
        df_modifier=df_modifier,
        controller_output_variable_name='Q_calculated',
        integration_method='monte_carlo',
        integration_num_evals=64,
        save_output_only=False,
        **kwargs
):
    """
    Calculates controller output for a single sequence and returns the Q sequence.

    :param df: DataFrame containing trajectory data.
    :param controller_config: Controller configuration dictionary.
    :param controller_output_variable_name: Base name for controller output columns.
    :param integration_method: Method for integration.
    :param integration_num_evals: Number of evaluations for integration.
    :return: List of Q values for the sequence.
    """

    environment_attributes_dict = controller_config['environment_attributes_dict']

    if not isinstance(controller_output_variable_name, list):
        names_of_variables_to_save = [controller_output_variable_name]
    else:
        names_of_variables_to_save = controller_output_variable_name

    # Keep a copy of the "base" output names (before regular_grid expands them).
    # For regular_grid we expect the controller output dimension to match this base list length.
    base_output_names_pre_grid = list(names_of_variables_to_save)


    # Process random sampling features
    df, environment_attributes_dict = process_random_sampling(df, environment_attributes_dict)

    # Process regular grid evaluation features (e.g. mu_regular_grid_0.3_1.1_0.05)
    # This generates multiple controller outputs per row (one per grid point).
    environment_attributes_dict, regular_grid_specs, names_of_variables_to_save = process_regular_grid(
        df=df,
        environment_attributes_dict=environment_attributes_dict,
        output_variable_names=names_of_variables_to_save,
    )

    df_original = df.copy()
    df = df_modifier(df)

    # Get integration features and their ranges
    integration_features, feature_ranges, environment_attributes_dict = get_integration_features(df, environment_attributes_dict)

    # Get derivative features
    differentiation_features, environment_attributes_dict, names_of_variables_to_save = get_differentiation_features(
                                                                                         environment_attributes_dict,
                                                                                         names_of_variables_to_save)

    if regular_grid_specs and (integration_features or differentiation_features):
        raise ValueError("Cannot use regular_grid features together with integrate/differentiate features.")

    # Build initial environment attributes:
    # - for normal (column-based) attributes use df[value].iloc[0]
    # - for regular_grid specs, seed with the first grid value
    initial_environment_attributes = {}
    for key, value in environment_attributes_dict.items():
        if value in df.columns:
            initial_environment_attributes[key] = df[value].iloc[0]
        else:
            initial_environment_attributes[key] = np.nan
    for spec in regular_grid_specs:
        initial_environment_attributes[spec["controller_key"]] = float(spec["grid_values"][0])

    controller_instance = controller_creator(controller_config, initial_environment_attributes)

    Q_sequence = []
    state_components = controller_config['state_components']
    # Optional evaluation order for regular_grid:
    # - by_row (legacy): for each row, evaluate all grid values
    # - by_grid: for each grid value, run through the whole trajectory (better for stateful controllers / MPC warm-start)
    regular_grid_eval_order = str(kwargs.get('regular_grid_eval_order', 'by_row')).lower()
    regular_grid_reset_between_values = bool(kwargs.get('regular_grid_reset_between_values', True))

    if regular_grid_specs and regular_grid_eval_order not in ('by_row', 'by_grid'):
        raise ValueError(
            f"Invalid regular_grid_eval_order='{regular_grid_eval_order}'. "
            f"Supported: 'by_row' or 'by_grid'."
        )

    bar_total = len(df)
    if regular_grid_specs and regular_grid_eval_order == 'by_grid':
        # We do len(df) controller steps per grid value.
        bar_total = len(df) * len(regular_grid_specs[0]["grid_values"])

    bar = tqdm(
        total=bar_total,
        desc='Processing Sequence',
        leave=True,
        bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
        dynamic_ncols=True
    )
    try:
        # Reset or reinitialize the controller if necessary
        if hasattr(controller_instance, 'reset'):
            controller_instance.reset()

        if integration_features and differentiation_features:
            raise ValueError("Cannot integrate and differentiate at the same time.")

        if integration_features:
            # Integration always runs by_row (one result per row).
            for idx, row in df.iterrows():
                s = row[state_components]
                time_step = row['time']
                environment_attributes = {
                    key: row[value] for key, value in environment_attributes_dict.items()
                    if value in row.index
                }
                Q = integration(
                    controller=controller_instance,
                    s=s,
                    time=time_step,
                    environment_attributes=environment_attributes,
                    features=integration_features,
                    feature_ranges=feature_ranges,
                    method=integration_method,
                    num_evals=integration_num_evals,
                )
                Q_sequence.append(Q)
                bar.update(1)

        elif differentiation_features:
            # Differentiation always runs by_row (one result per row).
            for idx, row in df.iterrows():
                s = row[state_components]
                time_step = row['time']
                environment_attributes = {
                    key: row[value] for key, value in environment_attributes_dict.items()
                    if value in row.index
                }
                jacobian, control = differentiation(
                    controller=controller_instance,
                    s=s,
                    time=time_step,
                    environment_attributes=environment_attributes,
                    differentiation_features=differentiation_features,
                )
                jacobian_flat = jacobian.flatten()
                control_flat = control.flatten()
                Q = np.concatenate((jacobian_flat, control_flat))
                Q_sequence.append(Q)
                bar.update(1)

        else:
            if regular_grid_specs:
                if len(regular_grid_specs) != 1:
                    raise ValueError(
                        f"Only one regular_grid feature is supported at a time, got {len(regular_grid_specs)}."
                    )
                spec = regular_grid_specs[0]
                grid_key = spec["controller_key"]
                grid_values = spec["grid_values"]

                base_output_dim = len(base_output_names_pre_grid)
                # Preallocate output matrix for regular_grid:
                # shape = (num_rows, base_output_dim * num_grid_values)
                Q_mat = np.zeros((len(df), base_output_dim * len(grid_values)), dtype=float)

                if regular_grid_eval_order == 'by_row':
                    for ridx, row in enumerate(df.itertuples(index=False)):
                        # Use df columns by name (slower), but keep logic close to legacy path.
                        row_s = getattr(row, state_components)
                        row_t = getattr(row, 'time')
                        environment_attributes = {}
                        for key, value in environment_attributes_dict.items():
                            if hasattr(row, value):
                                environment_attributes[key] = getattr(row, value)

                        outputs = []
                        for gv in grid_values:
                            updated_attributes = dict(environment_attributes)
                            updated_attributes[grid_key] = float(gv)
                            out = np.atleast_1d(controller_instance.step(
                                s=row_s,
                                time=row_t,
                                updated_attributes=updated_attributes,
                            )).astype(float)
                            outputs.append(out)
                        Q_row = np.concatenate(outputs, axis=0)
                        Q_mat[ridx, :] = Q_row
                        Q_sequence.append(Q_row)
                        bar.update(1)

                else:  # by_grid
                    # Run entire trajectory per grid value, then fill the corresponding block of columns.
                    for g_i, gv in enumerate(grid_values):
                        if regular_grid_reset_between_values and hasattr(controller_instance, 'reset'):
                            controller_instance.reset()

                        col_start = g_i * base_output_dim
                        col_end = (g_i + 1) * base_output_dim

                        for ridx, row in enumerate(df.itertuples(index=False)):
                            row_s = getattr(row, state_components)
                            row_t = getattr(row, 'time')
                            environment_attributes = {}
                            for key, value in environment_attributes_dict.items():
                                if hasattr(row, value):
                                    environment_attributes[key] = getattr(row, value)
                            environment_attributes[grid_key] = float(gv)

                            out = np.atleast_1d(controller_instance.step(
                                s=row_s,
                                time=row_t,
                                updated_attributes=environment_attributes,
                            )).astype(float)

                            if out.shape[0] != base_output_dim:
                                raise ValueError(
                                    f"Controller output dimension mismatch for regular_grid. "
                                    f"Expected {base_output_dim} (from controller_output_variable_name), got {out.shape[0]}."
                                )
                            Q_mat[ridx, col_start:col_end] = out
                            bar.update(1)

                    Q_sequence = Q_mat  # override list accumulation; we already built the full matrix

            else:
                for idx, row in df.iterrows():
                    s = row[state_components]
                    time_step = row['time']
                    environment_attributes = {}
                    for key, value in environment_attributes_dict.items():
                        if value in row.index:
                            environment_attributes[key] = row[value]

                    Q = np.atleast_1d(controller_instance.step(
                        s=s,
                        time=time_step,
                        updated_attributes=environment_attributes,
                    ))
                    Q_sequence.append(Q)
                    bar.update(1)

    except Exception as e:
        print(f"Error in processing sequence: {e}")
    finally:
        bar.close()

    Q_sequence = np.atleast_2d(Q_sequence)
    if save_output_only:
        return pd.DataFrame(Q_sequence, columns=names_of_variables_to_save)
    else:
        df_original[names_of_variables_to_save] = Q_sequence

    return df_original


def process_random_sampling(df, environment_attributes_dict):
    """
    Process random sampling for features specified in environment_attributes_dict.
    Allows specifying custom ranges and an optional step for discrete sampling within the feature names.

    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: Updated df and environment_attributes_dict
    """
    # Updated regex pattern explanation:
    #   (.+)                     : Captures the feature name (any characters)
    #   _random_uniform_         : Literal string to denote random uniform sampling
    #   ([-+]?\d*\.?\d+)         : Captures the lower bound (supports optional sign and decimals)
    #   _                       : Separator between lower and upper bound
    #   ([-+]?\d*\.?\d+)         : Captures the upper bound
    #   (?:_([-+]?\d*\.?\d+))?    : Optionally captures the step value for discrete sampling
    #   _?                      : Optional trailing underscore for flexibility in naming
    pattern = re.compile(
        r'^(.+)_random_uniform_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)(?:_([-+]?\d*\.?\d+))?_?$'
    )
    num_rows = len(df)

    for key, value in environment_attributes_dict.items():
        match = pattern.match(value)
        if match:
            feature = match.group(1)
            # Check if min and max are provided in the name
            if len(match.groups()) >= 3:
                try:
                    feature_min = float(match.group(2))
                    feature_max = float(match.group(3))
                except ValueError:
                    raise ValueError(f"Invalid range values in feature name: {value}")
            else:
                # If no range is provided, use the min and max from the DataFrame
                feature_min = df[feature].min()
                feature_max = df[feature].max()

            # Generate random uniform samples within the specified range
            new_feature_name = f"{feature}_random_uniform"

            # Check for an optional step value for discrete sampling
            if match.group(4) is not None:
                try:
                    step = float(match.group(4))
                except ValueError:
                    raise ValueError(f"Invalid step value in feature name: {value}")
                if step <= 0:
                    raise ValueError(f"Step value must be positive in feature name: {value}")
                # np.arange is used to generate evenly spaced discrete values.
                # Adding a small epsilon (step/10) ensures inclusion of the upper bound despite floating point precision issues.
                discrete_values = np.arange(feature_min, feature_max + step / 10, step)
                # Randomly assign each row a value from the discrete set (sampling with replacement)
                df[new_feature_name] = np.random.choice(discrete_values, num_rows, replace=True)
            else:
                # Generate continuous random samples when no step is provided
                df[new_feature_name] = np.random.uniform(feature_min, feature_max, num_rows)

            # Update the environment_attributes_dict with the new feature name
            environment_attributes_dict[key] = new_feature_name

    return df, environment_attributes_dict


def _format_grid_value_for_column(value: float) -> str:
    """
    Format a float into a stable column suffix, e.g. 0.3 -> '0p3', 1.1 -> '1p1'.
    """
    s = f"{float(value):.6f}".rstrip('0').rstrip('.')
    return s.replace('.', 'p').replace('-', 'm')


def process_regular_grid(df, environment_attributes_dict, output_variable_names):
    """
    Process regular grid specs in environment_attributes_dict.

    Example:
        environment_attributes_dict["mu"] = "mu_regular_grid_0.3_1.1_0.5"

    Behavior:
    - Does NOT add new columns to df.
    - Returns:
        - updated environment_attributes_dict (grid spec value replaced with the base feature name)
        - regular_grid_specs: list of dicts with controller_key, feature, grid_values
        - updated output_variable_names (expanded with suffixes for each grid value)
    """
    pattern = re.compile(r'^(.+)_regular_grid_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_?$')
    regular_grid_specs = []

    for key, value in list(environment_attributes_dict.items()):
        match = pattern.match(value)
        if not match:
            continue

        feature = match.group(1)
        try:
            vmin = float(match.group(2))
            vmax = float(match.group(3))
            step = float(match.group(4))
        except ValueError:
            raise ValueError(f"Invalid regular_grid spec: {value}")
        if step <= 0:
            raise ValueError(f"regular_grid step must be positive in spec: {value}")
        if vmax < vmin:
            raise ValueError(f"regular_grid max must be >= min in spec: {value}")

        # Include upper bound if it lands on the grid; epsilon helps floating precision.
        grid_values = np.arange(vmin, vmax + step / 10, step, dtype=float)
        if grid_values.size == 0:
            raise ValueError(f"regular_grid produced empty grid for spec: {value}")

        regular_grid_specs.append(
            {
                "controller_key": key,  # name used by controller, e.g. 'mu'
                "feature": feature,      # base feature name, e.g. 'mu'
                "grid_values": grid_values,
            }
        )

        # Replace the spec with the base feature name (may or may not exist as a df column).
        environment_attributes_dict[key] = feature

    if regular_grid_specs:
        # Expand output names: one set of outputs per grid value.
        if len(regular_grid_specs) != 1:
            raise ValueError(
                f"Only one regular_grid feature is supported at a time, got {len(regular_grid_specs)}."
            )

        spec = regular_grid_specs[0]
        feature = spec["feature"]
        expanded = []
        for gv in spec["grid_values"]:
            gv_suffix = _format_grid_value_for_column(gv)
            for base in output_variable_names:
                expanded.append(f"{base}_{feature}_{gv_suffix}")
        output_variable_names = expanded

    return environment_attributes_dict, regular_grid_specs, output_variable_names


def get_integration_features(df, environment_attributes_dict):
    """
    Identify features to integrate over from environment_attributes_dict.
    Allows specifying custom ranges within the feature names.

    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: integration_features, feature_ranges, updated environment_attributes_dict
    """
    # Regex pattern to capture feature name and optional min and max values
    pattern = re.compile(r'^(.+)_integrate_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_?$')
    integration_features = []
    feature_ranges = {}

    for key, value in environment_attributes_dict.items():
        match = pattern.match(value)
        if match:
            feature = match.group(1)
            # Check if min and max are provided in the name
            if len(match.groups()) == 3:
                try:
                    feature_min = float(match.group(2))
                    feature_max = float(match.group(3))
                except ValueError:
                    raise ValueError(f"Invalid range values in feature name: {value}")
            else:
                # If no range is provided, use the min and max from the DataFrame
                feature_min = df[feature].min()
                feature_max = df[feature].max()

            integration_features.append(feature)
            feature_ranges[feature] = (feature_min, feature_max)

            # Update the environment_attributes_dict to use the feature name
            environment_attributes_dict[key] = feature

    return integration_features, feature_ranges, environment_attributes_dict



def get_differentiation_features(environment_attributes_dict, output_variable_names):
    """
    Identify features to differentiate over from environment_attributes_dict.

    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: differentiation_features, updated environment_attributes_dict
    """
    # Regex pattern to capture feature name and optional min and max values
    pattern = re.compile(r'^(.+)_differentiate_$')
    differentiation_features = []

    for key, value in environment_attributes_dict.items():
        match = pattern.match(value)
        if match:
            feature = match.group(1)

            differentiation_features.append(feature)

            # Update the environment_attributes_dict to use the feature name
            environment_attributes_dict[key] = feature

    if differentiation_features:
        # Construct the column names
        names_of_derivatives_to_save = [
            f"{output_name}_d{diff_feature}"
            for output_name in output_variable_names
            for diff_feature in differentiation_features
        ]
        names_of_controls_to_save = [
            f"{output_name[1:]}_d{diff_feature}"
            for output_name in output_variable_names
            for diff_feature in differentiation_features
        ]
        names_of_variables_to_save = names_of_derivatives_to_save + names_of_controls_to_save
    else:
        names_of_variables_to_save = output_variable_names

    return differentiation_features, environment_attributes_dict, names_of_variables_to_save



def integration(controller, s, time, environment_attributes, features, feature_ranges, method='nquad', num_evals=100):
    """
    Integrate the controller output over multiple feature ranges to obtain the average control
    using either Regular Grid (nquad) or Advanced Monte Carlo methods.

    :param controller: The controller instance.
    :param s: State vector.
    :param time: Current time.
    :param environment_attributes: Current environment attributes.
    :param features: List of features to integrate over.
    :param feature_ranges: Dictionary mapping each feature to its (min, max) range.
    :param method: Integration method - 'nquad' or 'monte_carlo'.
    :param num_evals: Total number of function evaluations.
    :return: Average control value.
    """
    d = len(features)  # Dimensionality
    def integrand(*args):
        updated_attributes = environment_attributes.copy()
        for feature, value in zip(features, args):
            updated_attributes[feature] = value
        return controller.step(s=s, time=time, updated_attributes=updated_attributes)

    # Define the limits for each feature
    limits = [feature_ranges[feature] for feature in features]
    lower_bounds = np.array([limit[0] for limit in limits])
    upper_bounds = np.array([limit[1] for limit in limits])
    ranges = upper_bounds - lower_bounds
    volume = np.prod(ranges)

    if method == 'nquad':
        # Control the number of function evaluations by setting limits on subdivisions
        # Note: nquad does not allow exact control over function evaluations,
        # but we can limit the recursion depth and tolerances to approximate it.

        # Estimate the maximum number of function evaluations
        # nquad uses recursive calls, so it's not straightforward.
        # Here, we set a maximum number of subdivisions per dimension.
        # This is a heuristic approach.
        max_subdivisions = ceil(num_evals ** (1/d))  # Approximate

        opts = {'limit': max_subdivisions, 'epsabs': 1e-2, 'epsrel': 1e-2}

        # Perform the multi-dimensional integration using nquad
        integral, error = nquad(integrand, limits, opts=[opts]*d)

        average_control = integral / volume

    elif method == 'monte_carlo':
        # Advanced Monte Carlo Integration using Quasi-Monte Carlo (Sobol) and Stratified Sampling

        # Number of samples
        N = num_evals

        # Initialize Sobol sampler
        sampler = qmc.Sobol(d=d, scramble=True)

        # Generate Sobol sequence samples
        sample = sampler.random_base2(m=int(np.log2(N)))  # N must be a power of 2 for Sobol

        # If N is not a power of 2, adjust to the next power of 2
        if 2**int(np.log2(N)) != N:
            m = int(np.ceil(np.log2(N)))
            N_new = 2**m
            sample = sampler.random_base2(m=m)
            print(f"Adjusted number of samples from {N} to {N_new} for Sobol sequence.")
            N = N_new

        # Stratified Sampling: Divide each dimension into sqrt(N) strata
        # and sample uniformly within each stratum
        # Note: This increases complexity; here we use Sobol sequences which already provide low-discrepancy.

        # Scale samples to the feature ranges
        sample_scaled = lower_bounds + sample * ranges

        # Evaluate the integrand at all sampled points
        evaluations = np.array([integrand(*point) for point in sample_scaled])

        # Compute the integral as the average value times the volume
        integral = np.mean(evaluations) * volume
        average_control = integral / volume
        print(f"Monte Carlo Integration: Estimated average control = {average_control}, integral = {integral}, volume = {volume}")

    else:
        raise ValueError("Invalid integration method. Choose 'nquad' or 'monte_carlo'.")

    return np.atleast_1d(average_control)


def differentiation(
    controller: Any,
    s: Any,
    time: float,
    environment_attributes: Dict[str, Any],
    differentiation_features: List[str],
    method: str = 'savgol',
    step_size: float = 0.5e-3,
    window_length: int = None,
    polyorder: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differentiate controller output with respect to multiple features using the specified method.
    Always returns a 2D NumPy array (Jacobian matrix) and central controller outputs.

    :param controller: The controller instance.
    :param s: State vector.
    :param time: Current time.
    :param environment_attributes: Current environment attributes.
    :param differentiation_features: List of features to differentiate over.
    :param method: Differentiation method to use ('savgol' or 'nd').
    :param step_size: Step size for generating points or numerical differentiation.
    :param window_length: (For 'savgol' method) The length of the filter window (must be odd and >= polyorder + 2).
    :param polyorder: (For 'savgol' method) The order of the polynomial used to fit the samples.
    :return:
        - jacobian: A (output_dim, num_features) array of derivatives.
        - central_outputs: A (output_dim, num_features) array of controller outputs at the central points.
    """

    # Set default parameters based on method
    if method == 'savgol':
        if window_length is None:
            window_length = 5
        if polyorder is None:
            polyorder = 1
        # Validate window_length and polyorder
        if window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")
        if window_length < polyorder + 2:
            raise ValueError("window_length must be at least polyorder + 2.")
    elif method == 'nd':
        pass
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods are 'savgol' and 'nd'.")

    # Retrieve current values, set to np.nan if feature is missing, and issue a warning
    current_values = []
    for feature in differentiation_features:
        if feature in environment_attributes:
            current_values.append(environment_attributes[feature])
        else:
            warnings.warn(
                f"Feature '{feature}' not found in environment_attributes. Setting its value to np.nan.",
                UserWarning
            )
            current_values.append(np.nan)

    # Convert to NumPy array for consistency
    current_values = np.array(current_values, dtype=float)

    # Initialize lists to store partial derivatives and central outputs
    partial_derivatives = []
    central_outputs = []

    output_dim = None

    # Iterate over each feature and compute derivatives
    for feature, value in zip(differentiation_features, current_values):
        derivative, central_output = differentiate_feature(
            controller=controller,
            s=s,
            time=time,
            environment_attributes=environment_attributes,
            feature=feature,
            value=value,
            method=method,
            step_size=step_size,
            window_length=window_length if method == 'savgol' else None,
            polyorder=polyorder if method == 'savgol' else None
        )

        # Determine output_dim if not already set
        if output_dim is None and not np.isnan(derivative).all():
            output_dim = derivative.shape[0]

        partial_derivatives.append(derivative)
        central_outputs.append(central_output)

    # Convert lists to NumPy arrays
    jacobian = np.column_stack(partial_derivatives)  # Shape: (output_dim, num_features)
    central_outputs = np.column_stack(central_outputs)  # Shape: (output_dim, num_features)

    # Ensure the Jacobian is 2D
    jacobian = np.atleast_2d(jacobian)
    central_outputs = np.atleast_2d(central_outputs)

    return jacobian, central_outputs



def differentiate_feature(
    controller: Any,
    s: Any,
    time: float,
    environment_attributes: Dict[str, Any],
    feature: str,
    value: float,
    method: str,
    step_size: float,
    window_length: int,
    polyorder: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differentiate controller output with respect to a single feature.

    :return:
        - derivative: (output_dim,) array of derivatives for this feature.
        - central_output: (output_dim,) array of controller outputs at the central point.
    """
    if np.isnan(value):
        # Evaluate controller to determine output_dim if not already known
        sample_output = controller.step(s=s, time=time, updated_attributes=environment_attributes)
        sample_output = np.atleast_1d(sample_output).astype(float)
        output_dim = sample_output.shape[0]
        nan_array = np.full((output_dim,), np.nan)
        return nan_array, nan_array

    # Define function to evaluate controller output at a given feature value
    def func(x: float):
        updated_attributes = deepcopy(environment_attributes)
        updated_attributes[feature] = x
        output = controller.step(s=s, time=time, updated_attributes=updated_attributes)
        return np.atleast_1d(output).astype(float)

    if method == 'savgol':
        half_window = (window_length - 1) // 2
        offsets = np.arange(-half_window, half_window + 1) * step_size
        test_values = value + offsets

        # Evaluate controller outputs at test values
        outputs = [func(test_value) for test_value in test_values]
        outputs = np.vstack(outputs)  # Shape: (window_length, output_dim)
        # print(f'feature: {feature}, test_values{test_values}, outputs: {outputs}')

        output_dim = outputs.shape[1]

        # Compute the derivative using Savitzky–Golay filter for each output dimension
        derivatives = savgol_filter(
            outputs,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=step_size,
            axis=0,  # Compute derivative along the window axis
            mode='constant',
        )

        # Extract the central derivative and central output
        central_derivative = derivatives[half_window, :]  # Shape: (output_dim,)
        central_output = outputs[half_window, :]         # Shape: (output_dim,)
    elif method == 'nd':
        derivative_func = nd.Derivative(func, step=step_size, method='central', order=2)
        central_derivative = derivative_func(value)  # Shape: (output_dim,)
        central_output = func(value)                # Shape: (output_dim,)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods are 'savgol' and 'nd'.")

    return central_derivative, central_output