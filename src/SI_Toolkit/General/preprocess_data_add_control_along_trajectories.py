import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from Control_Toolkit.others.globals_and_utils import get_controller_name, import_controller_by_name

from scipy.integrate import nquad
from scipy.stats import qmc
from math import ceil
from copy import deepcopy

import numdifftools as nd
from typing import Any, Dict, List
import warnings

def add_control_along_trajectories(
        df,
        controller_config,
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

    controller_name = controller_config['controller_name']
    optimizer_name = controller_config.get('optimizer_name', None)

    environment_name = controller_config['environment_name']
    action_space = controller_config['action_space']
    state_components = controller_config['state_components']
    environment_attributes_dict = controller_config['environment_attributes_dict']

    if not isinstance(controller_output_variable_name, list):
        names_of_variables_to_save = [controller_output_variable_name]
    else:
        names_of_variables_to_save = controller_output_variable_name

    # Process random sampling features
    df, environment_attributes_dict = process_random_sampling(df, environment_attributes_dict)

    # Get integration features and their ranges
    integration_features, feature_ranges, environment_attributes_dict = get_integration_features(df, environment_attributes_dict)

    # Get derivative features
    differentiation_features, environment_attributes_dict, names_of_variables_to_save = get_differentiation_features(
                                                                                         environment_attributes_dict,
                                                                                         names_of_variables_to_save)

    initial_environment_attributes = {key: df[value].iloc[0] for key, value in environment_attributes_dict.items()}

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

    Q_sequence = []
    bar = tqdm(
        total=len(df),
        desc='Processing Sequence',
        leave=True,
        bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
        dynamic_ncols=True
    )

    try:
        # Reset or reinitialize the controller if necessary
        if hasattr(controller_instance, 'reset'):
            controller_instance.reset()

        for idx, row in df.iterrows():
            s = row[state_components].values
            time_step = row['time']
            environment_attributes = {key: row[value] for key, value in environment_attributes_dict.items()}

            if integration_features and differentiation_features:
                raise ValueError("Cannot integrate and differentiate at the same time.")

            if integration_features:
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
            elif differentiation_features:
                jacobian = differentiation(
                        controller=controller_instance,
                        s=s,
                        time=time_step,
                        environment_attributes=environment_attributes,
                        differentiation_features=differentiation_features,
                    )
                jacobian_flat = jacobian.flatten()
                Q = jacobian_flat


            else:
                Q = np.atleast_1d(controller_instance.step(
                    s=s,
                    time=time_step,
                    updated_attributes=environment_attributes,
                ))
            Q_sequence.append(Q)

            # Update progress bar
            bar.update(1)

    except Exception as e:
        print(f"Error in processing sequence: {e}")
    finally:
        bar.close()

    Q_sequence = np.atleast_2d(Q_sequence)
    if save_output_only:
        return pd.DataFrame(Q_sequence, columns=names_of_variables_to_save)
    else:
        df[names_of_variables_to_save] = Q_sequence

    return df


def process_random_sampling(df, environment_attributes_dict):
    """
    Process random sampling for features specified in environment_attributes_dict.
    Allows specifying custom ranges within the feature names.

    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: Updated df and environment_attributes_dict
    """
    # Regex pattern to capture feature name and optional min and max values
    pattern = re.compile(r'^(.+)_random_uniform_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_?$')
    num_rows = len(df)

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

            # Generate random uniform samples within the specified range
            new_feature_name = f"{feature}_random_uniform"
            df[new_feature_name] = np.random.uniform(feature_min, feature_max, num_rows)

            # Update the environment_attributes_dict with the new feature name
            environment_attributes_dict[key] = new_feature_name

    return df, environment_attributes_dict


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
        names_of_variables_to_save = [
            f"{output_name}_d{diff_feature}"
            for output_name in output_variable_names
            for diff_feature in differentiation_features
        ]
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



import numpy as np
import warnings
from typing import Any, Dict, List
import numdifftools as nd
from joblib import Parallel, delayed
from time import sleep

def differentiation(
        controller: Any,
        s: Any,
        time: float,
        environment_attributes: Dict[str, Any],
        differentiation_features: List[str],
        step_size: float = 1e-3,
) -> np.ndarray:
    """
    Differentiate controller output with respect to multiple features using numerical differentiation.
    Always returns a 2D NumPy array (Jacobian matrix).

    :param controller: The controller instance.
    :param s: State vector.
    :param time: Current time.
    :param environment_attributes: Current environment attributes.
    :param differentiation_features: List of features to differentiate over.
    :param step_size: Step size for numerical differentiation.
    :return:
        - If controller.step returns a scalar, returns a (1, num_features) array.
        - If controller.step returns a vector of length m, returns a (m, num_features) array.
    """

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

    # Initialize a list to store partial derivatives
    partial_derivatives = []

    # Compute partial derivatives sequentially
    for feature, value in zip(differentiation_features, current_values):
        print('*********')
        print('*********')
        print('*********')
        def func(x: float) -> np.ndarray:
            updated_attributes = {key: val if not isinstance(val, (list, dict)) else deepcopy(val)
                                  for key, val in environment_attributes.items()}

            updated_attributes[feature] = x
            output = controller.step(s=s, time=time, updated_attributes=updated_attributes)
            return np.atleast_1d(output).astype(float)

        for test_value in [value - step_size, value, value + step_size]:
            print(f"Feature: {feature}, Test Value: {test_value}, Output: {func(test_value)}")

        derivative_func = nd.Derivative(func, method='central')
        derivative = derivative_func(value)
        print(f"Feature: {feature}, Value: {value}, Derivative: {derivative}")
        partial_derivatives.append(derivative)
        print('*********')
        print('*********')
        print('*********')
        sleep(0.001)
    # Stack the partial derivatives to form the Jacobian matrix
    jacobian = np.column_stack(partial_derivatives)  # Shape: (output_dim, num_features)
    # Ensure the Jacobian is 2D
    if jacobian.ndim == 1:
        jacobian = jacobian.reshape(1, -1)

    return jacobian
