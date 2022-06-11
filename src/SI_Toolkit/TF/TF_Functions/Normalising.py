import tensorflow as tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

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
