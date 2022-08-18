import tensorflow as tf

"""
In the below functions normalizing_array is required 
to have columns in the same order as the features of (de)normalized_array
and rows in the order
0 -> mean
1 -> std
2 -> max
3 -> min
"""


def get_normalization_function_tf(
        normalization_info,
        variables_names,
        normalization_type='minmax_sym'
):

    normalizing_array = tf.convert_to_tensor(
        normalization_info[variables_names], dtype=tf.float32)

    if normalization_type == 'gaussian':
        a = 1.0 / normalizing_array[1, :]
        b = - normalizing_array[0, :] / normalizing_array[1, :]

    elif normalization_type == 'minmax_pos':
        a = 1.0 / (normalizing_array[2, :] - normalizing_array[3, :])
        b = - normalizing_array[3, :] / (normalizing_array[2, :] - normalizing_array[3, :])

    elif normalization_type == 'minmax_sym':
        a = 2.0 / (normalizing_array[2, :] - normalizing_array[3, :])
        b = -1.0 + 2.0 * (-normalizing_array[3, :] / (normalizing_array[2, :] - normalizing_array[3, :]))
    else:
        raise NameError('{} is not recognized as a normalization type'.format(normalization_type))

    a = tf.convert_to_tensor(a, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)

    def normalize_tf(denormalized_array):
        normalized_array = a * denormalized_array + b
        return normalized_array

    return normalize_tf


def get_denormalization_function_tf(
                                 normalization_info,
                                 variables_names,
                                 normalization_type='minmax_sym'):

    denormalizing_array = tf.convert_to_tensor(
        normalization_info[variables_names], dtype=tf.float32)

    if normalization_type == 'gaussian':
        A = denormalizing_array[1, :]
        B = denormalizing_array[0, :]

    elif normalization_type == 'minmax_pos':
        A = (denormalizing_array[2, :] - denormalizing_array[3, :])
        B = denormalizing_array[3, :]

    elif normalization_type == 'minmax_sym':
        A = ((denormalizing_array[2, :] - denormalizing_array[3, :]) / 2.0)
        B = ((denormalizing_array[2, :] - denormalizing_array[3, :]) / 2.0) + denormalizing_array[3, :]
    else:
        raise NameError('{} is not recognized as a normalization type'.format(normalization_type))

    A = tf.convert_to_tensor(A, dtype=tf.float32)
    B = tf.convert_to_tensor(B, dtype=tf.float32)

    def denormalize_tf(normalized_array):
        denormalized_array = A * normalized_array + B
        return denormalized_array

    return denormalize_tf


def get_scaling_function_for_output_of_differential_network(
                                 normalization_info,
                                 network_outputs,
                                 dt,
                                 normalization_type='minmax_sym'):

    DIFF_NET_STATE_VARIABLES = [x[2:] for x in network_outputs]  # Outputs without D_ -> to make possible comparison with inputs
    denormalizing_derivatives = tf.convert_to_tensor(normalization_info[network_outputs], dtype=tf.float32)
    normalizing_variables = tf.convert_to_tensor(normalization_info[DIFF_NET_STATE_VARIABLES], dtype=tf.float32)

    # # Augmentation matrix allows network not to include some derivatives - probably now not useful and in a false place
    # augmentation_matrix = np.zeros(shape=(len(inputs), len(outputs)))
    # for i in range(len(inputs)):
    #     if inputs[i] in DIFF_NET_STATE_VARIABLES:
    #         augmentation_matrix[i, DIFF_NET_STATE_VARIABLES.index(inputs[i])] = 1
    #
    # augmentation_matrix = tf.convert_to_tensor(augmentation_matrix, dtype=tf.float32)

    # Normalization constants
    if normalization_type == 'gaussian':
        a = 1.0 / normalizing_variables[1, :]
        b = - normalizing_variables[0, :] / normalizing_variables[1, :]

    elif normalization_type == 'minmax_pos':
        a = 1.0 / (normalizing_variables[2, :] - normalizing_variables[3, :])
        b = - normalizing_variables[3, :] / (normalizing_variables[2, :] - normalizing_variables[3, :])

    elif normalization_type == 'minmax_sym':
        a = 2.0 / (normalizing_variables[2, :] - normalizing_variables[3, :])
        b = -1.0 + 2.0 * (-normalizing_variables[3, :] / (normalizing_variables[2, :] - normalizing_variables[3, :]))
    else:
        raise NameError('{} is not recognized as a normalization type'.format(normalization_type))

    # Denormalization constants
    if normalization_type == 'gaussian':
        C = denormalizing_derivatives[1, :]
        D = denormalizing_derivatives[0, :]

    elif normalization_type == 'minmax_pos':
        C = (denormalizing_derivatives[2, :] - denormalizing_derivatives[3, :])
        D = denormalizing_derivatives[3, :]

    elif normalization_type == 'minmax_sym':
        C = ((denormalizing_derivatives[2, :] - denormalizing_derivatives[3, :]) / 2.0)
        D = ((denormalizing_derivatives[2, :] - denormalizing_derivatives[3, :]) / 2.0) + denormalizing_derivatives[3, :]
    else:
        raise NameError('{} is not recognized as a normalization type'.format(normalization_type))

    # C = augmentation_matrix @ C
    # D = augmentation_matrix @ D

    p1 = a * C * dt
    p2 = a * D * dt - b

    p1 = tf.convert_to_tensor(p1, dtype=tf.float32)
    p2 = tf.convert_to_tensor(p2, dtype=tf.float32)

    def scale_output_of_differential_network(normalized_array):
        # scaled_array = p1 * (normalized_array @ augmentation_matrix) + p2
        scaled_array = p1 * normalized_array + p2
        return scaled_array

    return scale_output_of_differential_network
