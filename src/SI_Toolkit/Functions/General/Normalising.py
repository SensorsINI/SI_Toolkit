"""
In the below functions normalizing_array is required 
to have columns in the same order as the features of (de)normalized_array
and rows in the order
0 -> mean
1 -> std
2 -> max
3 -> min
"""

from SI_Toolkit.computation_library import NumpyLibrary
import numpy as np
import os

def get_normalization_function(
        normalization_info,
        variables_names,
        lib=NumpyLibrary,
        normalization_type='minmax_sym',
        return_coeffs=False,
):
    if normalization_info is None:
        print('No normalization info provided! Normalization function is identity in this case!')

        def normalize(denormalized_array):
            return denormalized_array

        return normalize

    normalizing_array = lib.to_tensor(
        normalization_info[variables_names].values, dtype=lib.float32)

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

    a = lib.to_tensor(a, dtype=lib.float32)
    b = lib.to_tensor(b, dtype=lib.float32)

    def normalize(denormalized_array):
        normalized_array = a * denormalized_array + b
        return normalized_array

    if return_coeffs:
        return normalize, a, b
    else:
        return normalize


def get_denormalization_function(
                                 normalization_info,
                                 variables_names,
                                 lib=NumpyLibrary,
                                 normalization_type='minmax_sym',
                                 return_coeffs=False,
):
    if normalization_info is None:
        print('No normalization info provided! Denormalization function is identity in this case!')

        def denormalize(normalized_array):
            return normalized_array

        return denormalize

    denormalizing_array = lib.to_tensor(
        normalization_info[variables_names].values, dtype=lib.float32)

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

    A = lib.to_tensor(A, dtype=lib.float32)
    B = lib.to_tensor(B, dtype=lib.float32)

    def denormalize(normalized_array):
        denormalized_array = A * normalized_array + B
        return denormalized_array

    if return_coeffs:
        return denormalize, A, B
    else:
        return denormalize


def get_scaling_function_for_output_of_differential_network(
                                 normalization_info,
                                 network_outputs,
                                 dt,
                                 lib=NumpyLibrary,
                                 normalization_type='minmax_sym',
                                 return_coeffs=False,
):
    if normalization_info is None:
        print('No normalization info provided! scale_output_of_differential_network function is identity in this case!')

        def scale_output_of_differential_network(normalized_array):
            return normalized_array

        return scale_output_of_differential_network

    DIFF_NET_STATE_VARIABLES = [(x[2:] if x[:2] == 'D_' else x) for x in network_outputs]  # Outputs without D_ -> to make possible comparison with inputs
    denormalizing_derivatives = lib.to_tensor(normalization_info[network_outputs].values, dtype=lib.float32)
    normalizing_variables = lib.to_tensor(normalization_info[DIFF_NET_STATE_VARIABLES].values, dtype=lib.float32)

    # # Augmentation matrix allows network not to include some derivatives - probably now not useful and in a false place
    # augmentation_matrix = np.zeros(shape=(len(inputs), len(outputs)))
    # for i in range(len(inputs)):
    #     if inputs[i] in DIFF_NET_STATE_VARIABLES:
    #         augmentation_matrix[i, DIFF_NET_STATE_VARIABLES.index(inputs[i])] = 1
    #
    # augmentation_matrix = lib.to_tensor(augmentation_matrix, dtype=lib.float32)

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
    p2 = a * D * dt

    p1 = lib.to_tensor(p1, dtype=lib.float32)
    p2 = lib.to_tensor(p2, dtype=lib.float32)

    def scale_output_of_differential_network(normalized_array):
        # scaled_array = p1 * (normalized_array @ augmentation_matrix) + p2
        scaled_array = p1 * normalized_array + p2
        return scaled_array

    if return_coeffs:
        return scale_output_of_differential_network, p1, p2
    else:
        return scale_output_of_differential_network


def write_out_normalization_vectors(normalization_info, net_info):
    _,  a, b = get_normalization_function(normalization_info, net_info.inputs, return_coeffs=True)
    _, A, B = get_denormalization_function(normalization_info, net_info.outputs, return_coeffs=True)
    # with open(os.path.join(net_info.path_to_net, 'normalization_vectors.txt'), 'w') as f:
    #     f.write('Normalization coefficients for inputs\n')
    #     f.write('a = ' + str(a) + '\n')
    #     f.write('b = ' + str(b) + '\n')
    #     f.write('Denormalization coefficients for outputs\n')
    #     f.write('A = ' + str(A) + '\n')
    #     f.write('B = ' + str(B) + '\n')

    np.savetxt(os.path.join(net_info.path_to_net, "normalization_vec_a.csv"), a[np.newaxis, :], delimiter=",", fmt='%0.8f')
    np.savetxt(os.path.join(net_info.path_to_net, "normalization_vec_b.csv"), b[np.newaxis, :], delimiter=",", fmt='%0.8f')
    np.savetxt(os.path.join(net_info.path_to_net, "denormalization_vec_A.csv"), A[np.newaxis, :], delimiter=",", fmt='%0.8f')
    np.savetxt(os.path.join(net_info.path_to_net, "denormalization_vec_B.csv"), B[np.newaxis, :], delimiter=",", fmt='%0.8f')
