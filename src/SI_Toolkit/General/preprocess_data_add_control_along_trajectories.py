import os
import re
import logging
from math import ceil
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import nquad
from scipy.stats import qmc
from tqdm import tqdm

from Control_Toolkit.others.globals_and_utils import get_controller_name, import_controller_by_name
import concurrent.futures


# Adjust the paths as needed
MAIN_LOG_FILE = 'progress_main.log'
WORKER_LOG_DIR = 'worker_logs'

# Create worker log directory if it doesn't exist
os.makedirs(WORKER_LOG_DIR, exist_ok=True)

# Configure the main logger
logging.basicConfig(
    filename=MAIN_LOG_FILE,
    filemode='a',
    format='%(asctime)s - MAIN - %(levelname)s - %(message)s',
    level=logging.INFO
)
main_logger = logging.getLogger('Main')


class TqdmToLogger(object):
    """
    Output stream for TQDM which will write to the logger.
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf):
        buf = buf.strip('\r\n')
        if buf:
            self.logger.log(self.level, buf)

    def flush(self):
        pass


def add_control_along_trajectories(
        df,
        controller_config,
        controller_output_variable_name='Q_calculated',
        num_samples=10,
        parallel=False,
        integration_method='monte_carlo',
        intergration_num_evals=64,
        **kwargs
):
    """
    Adds controller output and its uncertainty to the trajectory data with per-worker progress bars.

    :param df: DataFrame containing trajectory data.
    :param controller_config: Controller configuration dictionary.
    :param controller_output_variable_name: Base name for controller output columns.
    :param num_samples: Number of stochastic evaluations per trajectory step.
    :param parallel: Whether to use parallel processing.
    :return: DataFrame with added controller outputs and uncertainty measures.
    """

    main_logger.info("Starting add_control_along_trajectories function.")

    controller_name = controller_config['controller_name']
    optimizer_name = controller_config.get('optimizer_name', None)

    environment_name = controller_config['environment_name']
    action_space = controller_config['action_space']
    state_components = controller_config['state_components']
    environment_attributes_dict = controller_config['environment_attributes_dict']

    # Process random sampling features
    df, environment_attributes_dict = process_random_sampling(df, environment_attributes_dict)
    main_logger.info("Processed random sampling features.")

    # Get integration features and their ranges
    integration_features, feature_ranges, environment_attributes_dict = get_integration_features(df, environment_attributes_dict)
    main_logger.info("Retrieved integration features and their ranges.")

    initial_environment_attributes = {key: df[value].iloc[0] for key, value in environment_attributes_dict.items()}

    # Determine the number of workers
    if num_samples is None:
        num_samples = 1

    if parallel:
        num_workers = min(cpu_count(), num_samples)
        main_logger.info(f"Number of workers: {num_workers}")
    else:
        num_workers = 1  # Single worker for sequential processing
        main_logger.info("Running in sequential mode with a single worker.")

    controller_name, _ = get_controller_name(
        controller_name=controller_name
    )

    # Calculate number of sequences per worker
    base_sequences_per_worker = num_samples // num_workers
    remainder = num_samples % num_workers
    sequences_per_worker = [base_sequences_per_worker] * num_workers
    for i in range(remainder):
        sequences_per_worker[i] += 1  # Distribute the remainder

    main_logger.info(f"Sequences per worker: {sequences_per_worker}")

    # Prepare worker tasks
    tasks = []
    for worker_idx in range(num_workers):
        task = (
            worker_idx,  # Pass worker index
            sequences_per_worker[worker_idx],
            df.copy(),  # Ensure each worker has its own copy
            controller_config,
            environment_attributes_dict.copy(),
            integration_features,
            feature_ranges,
            state_components,
            controller_output_variable_name,
            integration_method,
            intergration_num_evals,
        )
        tasks.append(task)

    results = []

    if parallel:
        main_logger.info("Starting parallel processing.")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(worker_process_sequences, task): task[0] for task in tasks
            }
            main_logger.info(f"Submitted {len(futures)} tasks to the executor.")

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                worker_idx = futures[future]
                try:
                    worker_result = future.result()
                    results.append(worker_result)
                    main_logger.info(f"Worker {worker_idx + 1} completed successfully.")
                except Exception as e:
                    main_logger.error(f"Worker {worker_idx + 1} generated an exception: {e}")

    else:
        main_logger.info("Starting sequential processing.")
        # Initialize a single tqdm progress bar for the single worker with custom bar format
        worker_idx = 0
        bar = tqdm(
            total=sequences_per_worker[0] * len(df),  # Number of sequences * time steps
            desc='Worker 1',
            position=0,
            leave=True,
            bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
            dynamic_ncols=True
        )

        try:
            worker_result = worker_process_sequences(tasks[0], bar)
            results.append(worker_result)
            main_logger.info(f"Worker {worker_idx + 1} completed successfully.")
        except Exception as e:
            main_logger.error(f"Worker {worker_idx + 1} generated an exception: {e}")
        finally:
            # Update and close the progress bar
            bar.update(sequences_per_worker[0] * len(df) - bar.n)
            bar.close()

    # Aggregate all Q sequences
    all_Q_sequences = []
    for worker_Q_sequences in results:
        all_Q_sequences.extend(worker_Q_sequences)  # Each worker returns a list of Q sequences

    if len(all_Q_sequences) == 0:
        main_logger.error("No Q sequences were generated. Check for errors in worker processing.")
        raise ValueError("No Q sequences were generated. Check for errors in worker processing.")

    main_logger.info("Aggregated all Q sequences.")

    # Convert to numpy array for efficient computation
    # Shape: (num_samples, num_time_steps)
    Q_array = np.array(all_Q_sequences)

    # Compute statistics across the samples for each time step
    mean_Q = np.mean(Q_array, axis=0)
    std_Q = np.std(Q_array, axis=0)
    conf_low, conf_high = stats.t.interval(
        0.95,
        len(all_Q_sequences) - 1,
        loc=mean_Q,
        scale=stats.sem(Q_array, axis=0)
    )

    main_logger.info("Computed statistics across Q sequences.")

    # Add the statistics to the DataFrame
    df[f'{controller_output_variable_name}_mean'] = mean_Q
    df[f'{controller_output_variable_name}_std'] = std_Q
    df[f'{controller_output_variable_name}_conf_low'] = conf_low
    df[f'{controller_output_variable_name}_conf_high'] = conf_high

    main_logger.info("Added statistics to the DataFrame.")

    main_logger.info("Completed add_control_along_trajectories function.")

    return df


def worker_process_sequences(task, progress_bar=None):
    """
    Worker function to process multiple full sequences with progress updates and logging.

    :param task: Tuple containing (
        worker_idx,
        num_sequences,
        df,
        controller_config,
        environment_attributes_dict,
        integration_features,
        feature_ranges,
        state_components,
        controller_output_variable_name,
        integration_method,
        intergration_num_evals,
    )
    :param progress_bar: tqdm progress bar instance (only used in sequential mode).
    :return: List of Q sequences (each sequence is a list of Q values per time step)
    """
    worker_idx = task[0]
    try:
        (
            worker_idx,
            num_sequences,
            df,
            controller_config,
            environment_attributes_dict,
            integration_features,
            feature_ranges,
            state_components,
            controller_output_variable_name,
            integration_method,
            intergration_num_evals,
        ) = task

        # Set up worker logger
        worker_log_file = os.path.join(WORKER_LOG_DIR, f'progress_worker_{worker_idx + 1}.log')
        worker_logger = setup_worker_logger(worker_idx, worker_log_file)

        worker_logger.info(f"Worker {worker_idx + 1} starting. Processing {num_sequences} sequences.")

        controller_name = controller_config['controller_name']
        optimizer_name = controller_config.get('optimizer_name', None)
        environment_name = controller_config['environment_name']
        action_space = controller_config['action_space']

        initial_environment_attributes = {key: df[value].iloc[0] for key, value in environment_attributes_dict.items()}

        controller_class = import_controller_by_name(controller_name)

        # Initialize a controller instance for this worker
        controller_instance = controller_class(
            environment_name=environment_name,
            initial_environment_attributes=initial_environment_attributes,
            control_limits=(action_space.low, action_space.high),
        )

        # Configure the controller
        if hasattr(controller_instance, 'has_optimizer') and controller_instance.has_optimizer:
            controller_instance.configure(optimizer_name)
            worker_logger.info(f"Controller {controller_name} configured with optimizer {optimizer_name}.")
        else:
            controller_instance.configure()
            worker_logger.info(f"Controller {controller_name} configured without optimizer.")

        Q_sequences = []

        steps_completed = 0
        total_steps = num_sequences * len(df)

        if progress_bar is None:
            # Use TqdmToLogger to redirect tqdm output to the logger
            tqdm_logger = TqdmToLogger(worker_logger, level=logging.INFO)
            progress_bar = tqdm(
                total=total_steps,
                desc=f'Worker {worker_idx + 1}',
                file=tqdm_logger,
                mininterval=1,
                ascii=True
            )

        for seq_idx in range(num_sequences):
            Q_sequence = []
            # Reset or reinitialize the controller if necessary
            if hasattr(controller_instance, 'reset'):
                controller_instance.reset()
                worker_logger.debug(f"Controller reset for sequence {seq_idx + 1}/{num_sequences}.")

            worker_logger.info(f"Worker {worker_idx + 1}, Sequence {seq_idx + 1}/{num_sequences} started.")

            for idx, row in df.iterrows():
                s = row[state_components].values
                time_step = row['time']
                environment_attributes = {key: row[key] for key in environment_attributes_dict.keys()}

                if integration_features:
                    Q = integration(
                        controller=controller_instance,
                        s=s,
                        time=time_step,
                        environment_attributes=environment_attributes,
                        features=integration_features,
                        feature_ranges=feature_ranges,
                        method=integration_method,
                        num_evals=intergration_num_evals,
                    )
                else:
                    Q = float(controller_instance.step(
                        s=s,
                        time=time_step,
                        updated_attributes=environment_attributes,
                    ))
                Q_sequence.append(Q)

                # Update progress
                steps_completed += 1
                progress_bar.update(1)

                # Log progress every 10 steps to reduce log verbosity
                if idx % 10 == 0 or idx == len(df) - 1:
                    worker_logger.debug(f"Worker {worker_idx + 1}, Sequence {seq_idx + 1}/{num_sequences}, Step {idx + 1}/{len(df)} completed.")

            Q_sequences.append(Q_sequence)
            worker_logger.info(f"Worker {worker_idx + 1}, Sequence {seq_idx + 1}/{num_sequences} completed.")

        progress_bar.close()
        worker_logger.info(f"Worker {worker_idx + 1} finished processing {num_sequences} sequences.")

        return Q_sequences

    except Exception as e:
        # Log the error and return an empty list for this worker
        try:
            worker_logger = setup_worker_logger(worker_idx, os.path.join(WORKER_LOG_DIR, f'progress_worker_{worker_idx + 1}.log'))
            worker_logger.error(f"Error in worker {worker_idx + 1} processing {num_sequences} sequences: {e}")
        except Exception as log_exception:
            main_logger.error(f"Failed to log error for worker {worker_idx + 1}: {log_exception}")
        return []


def setup_worker_logger(worker_idx, log_file):
    """
    Sets up a logger for a worker.

    :param worker_idx: Index of the worker.
    :param log_file: Path to the log file for the worker.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(f'Worker_{worker_idx + 1}')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers to the logger if it's already set up
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - WORKER - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def process_random_sampling(df, environment_attributes_dict):
    """
    Process random sampling for features specified in environment_attributes_dict.
    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: Updated df and environment_attributes_dict
    """
    pattern = re.compile(r'(.+)_random_uniform_$')
    num_rows = len(df)
    for key, value in environment_attributes_dict.items():
        match = pattern.match(value)
        if match:
            feature = match.group(1)
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            df[feature + '_random_uniform'] = np.random.uniform(feature_min, feature_max, num_rows)
            environment_attributes_dict[key] = feature + '_random_uniform'
    return df, environment_attributes_dict


def get_integration_features(df, environment_attributes_dict):
    """
    Identify features to integrate over from environment_attributes_dict.
    :param df: DataFrame containing the data
    :param environment_attributes_dict: dictionary of environment attributes
    :return: integration_features, feature_ranges, updated environment_attributes_dict
    """
    pattern = re.compile(r'(.+)_integrate_$')
    integration_features = []
    feature_ranges = {}
    for key, value in environment_attributes_dict.items():
        match = pattern.match(value)
        if match:
            feature = match.group(1)
            integration_features.append(feature)
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            feature_ranges[feature] = (feature_min, feature_max)
            environment_attributes_dict[key] = feature  # Update to use the feature name
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

    else:
        raise ValueError("Invalid integration method. Choose 'nquad' or 'monte_carlo'.")

    return average_control
