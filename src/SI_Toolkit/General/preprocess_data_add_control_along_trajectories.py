import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from Control_Toolkit.others.globals_and_utils import get_controller_name, import_controller_by_name

from scipy.integrate import nquad
from scipy.stats import qmc
from math import ceil

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

    # Process random sampling features
    df, environment_attributes_dict = process_random_sampling(df, environment_attributes_dict)

    # Get integration features and their ranges
    integration_features, feature_ranges, environment_attributes_dict = get_integration_features(df, environment_attributes_dict)

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
            else:
                Q = float(controller_instance.step(
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

    if save_output_only:
        return pd.DataFrame({controller_output_variable_name: Q_sequence})
    else:
        df[controller_output_variable_name] = Q_sequence

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



import re

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
    # Define the integration order based on features
    integration_order = features
    d = len(integration_order)  # Dimensionality

    def integrand(*args):
        updated_attributes = environment_attributes.copy()
        for feature, value in zip(integration_order, args):
            updated_attributes[feature] = value
        return controller.step(s=s, time=time, updated_attributes=updated_attributes)

    # Define the limits for each feature
    limits = [feature_ranges[feature] for feature in integration_order]
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

    return average_control
