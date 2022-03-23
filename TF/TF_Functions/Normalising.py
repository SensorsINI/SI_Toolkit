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


@Compile
def normalize_tf(denormalized_array,
                 normalizing_array,
                 normalization_type='minmax_sym'
                 ):

    if normalization_type == 'gaussian':
        normalized_array = (denormalized_array - normalizing_array[0, :]) / normalizing_array[1, :]

    elif normalization_type == 'minmax_pos':
        normalized_array = (denormalized_array - normalizing_array[3, :]) / (
                    normalizing_array[2, :] - normalizing_array[3, :])


    elif normalization_type == 'minmax_sym':
        normalized_array = -1.0 + 2.0 * (
                (denormalized_array - normalizing_array[3, :]) / (normalizing_array[2, :] - normalizing_array[3, :])
        )

    return normalized_array


@Compile
def denormalize_tf(normalized_array,
                   denormalizing_array,
                   normalization_type='minmax_sym'):

    if normalization_type == 'gaussian':
        denormalized_array = normalized_array * denormalizing_array[1, :] + denormalizing_array[0, :]

    elif normalization_type == 'minmax_pos':
        denormalized_array = normalized_array * (denormalizing_array[2, :] - denormalizing_array[3, :]) + \
                                               denormalizing_array[3, :]
    elif normalization_type == 'minmax_sym':
        denormalized_array = ((normalized_array + 1.0) / 2.0) * \
                                               (denormalizing_array[2, :] - denormalizing_array[3, :]) \
                                               + denormalizing_array[3, :]

    return denormalized_array
