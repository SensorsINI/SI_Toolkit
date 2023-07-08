__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import os
import typing
import numpy as np
import torch
from torch import Tensor
from glob import glob
from torch.autograd.function import Function
import time
import math
import torch.nn as nn
import torch.nn.functional as F

def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_filepaths(dirpath: str, search_key: str):
    """
    :param dirpath: folder path to search
    :param search_key: key to search (e.g. search_key='*.wav')
    :return: A sorted list of file paths
    """
    filepaths = [y for x in os.walk(dirpath) for y in glob(os.path.join(x[0], search_key))]
    filepaths = sorted(filepaths)
    return filepaths


def get_filenames(filepaths: typing.List):
    return [os.path.basename(x) for x in filepaths]


def decode_ckpt_name(filename: str, num_metrics: int):
    metrics = filename.split('-')
    if len(metrics) == num_metrics + 1:
        metrics = metrics[1:]
    elif len(metrics) == num_metrics + 2:
        metrics = metrics[1:-1]
    else:
        raise ValueError('CKPT filename illegal!')
    metrics = '='.join(metrics).split('=')
    values = metrics[1::2]
    metrics = metrics[::2]
    dict_metrics = dict(zip(metrics, values))
    return dict_metrics


class GradPreserveRound(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


class GradPreserveFloor(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.floor(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


def quantize_tensor(x: Tensor, qi: int, qf: int, enable: int = 0, use_floor: bool = False) -> Tensor:
    """
    :param x: input tensor
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :param enable: if 0, return x
    :param use_floor: Whether use floor() instead of round()
    :return: tensor quantized to fixed-point precision
    """
    if enable == 0:
        return x
    else:
        power = torch.tensor(float(2. ** qf), dtype=torch.float32)
        clip_val = torch.tensor(float(2. ** (qi + qf - 1) - 1), dtype=torch.float32)
        if use_floor:
            value = GradPreserveFloor.apply(x * power)
        else:
            value = GradPreserveRound.apply(x * power)  # Round Half to Even

        value = torch.max(value, -clip_val)
        value = torch.min(value, clip_val)
        # value = torch.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
        value = torch.div(value, power)
        return value

def quantize_array(x, qi, qf, enable, unsigned=0):
    """
    :param x: input numpy array
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :param enable: whether enable the function
    :param unsigned: whether unsigned input
    :return: array quantized to fixed-point precision
    """
    if enable == 0:
        return x
    power = np.asarray(2. ** qf).astype(np.float64)
    if unsigned == 0:
        clip_val = float(2. ** (qi + qf - 1))
        value = np.round(x * power)
        value = np.clip(value, -clip_val, clip_val - 1)  # saturation arithmetic
        result = value / power
    else:
        clip_val = float(2. ** (qi + qf))
        value_rnd = np.round(x * power)
        value = np.clip(value_rnd, 0, clip_val - 1)  # saturation arithmetic
        result = value / power
    return result

# def quantize_rnn(net, qi, qf, en):
#     for name, param in net.named_parameters():
#         if 'rnn' in name:
#             param.data = quantizeTensor(param.data, qi, qf, en)
#     return net

def cast_fxp_to_hex(x, data_type):
    x = np.asarray(x, dtype=data_type)
    return x

def cast_fp_to_fxp(x, qi, qf, use_floor=False):
    """
    Converts a floating point number into a fixed point number

    :param x: input numpy array
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    power = 2. ** qf
    clip_val = 2. ** (qi + qf - 1)
    value = x * power
    if use_floor:
        value = np.floor(value)  # rounding
    else:
        value = np.around(value)  # rounding
    value = np.clip(value, -clip_val, clip_val - 1).astype(np.int64)  # saturation arithmetic
    return value


def pruneTensor(x, alpha):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    n_neuron = x.size(0)
    n_input = x.size(1)
    prune_prob_mask = torch.exp(
        -alpha * torch.unsqueeze(torch.arange(0, n_neuron), dim=1).repeat(1, n_input).float()).cuda()
    prune_rand_mask = torch.rand(n_neuron, n_input).cuda()
    prune_mask = prune_rand_mask.masked_fill_(prune_rand_mask > prune_prob_mask, 1)
    prune_mask = prune_mask.masked_fill_(prune_rand_mask <= prune_prob_mask, 0)
    _, indices = torch.sort(torch.abs(x), 0)
    # print("indices shape", indices.size())
    # print("prune_mask shape", prune_mask.size())
    # print("x shape", x.size())
    for j in range(0, n_input):
        x[indices[:, j], j] *= prune_mask[:, j]

    return x


def targetedDropout(x, gamma, alpha, epoch):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    torch.manual_seed(epoch)
    torch.cuda.manual_seed_all(epoch)

    n_elements = x.numel()
    drop_part = round(n_elements * gamma)
    weight_vec = x.view(-1)
    weight_vec_abs = torch.abs(weight_vec)
    sorted, indices = torch.sort(weight_vec_abs)
    # print(sorted)
    drop_indices = indices[0:drop_part]
    drop_rand_mask = torch.rand(drop_indices.size(0)).cuda()
    drop_mask = torch.ones(drop_indices.size(0)).cuda()
    drop_mask = drop_mask.masked_fill_(drop_rand_mask <= alpha, 0)
    weight_vec[drop_indices] *= drop_mask
    weight = torch.reshape(weight_vec, (x.size(0), x.size(1)))

    return weight


def alignedTargetedDropout(x, gamma, alpha, num_pe, epoch):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    n_rows = x.shape[0]
    n_cols = x.shape[1]

    # Split and shuffle weight matrix
    for i in range(0, num_pe):
        for j in range(0, n_cols):
            targetedDropout(x[np.arange(i, n_rows, num_pe), j], gamma, alpha, epoch)
    return x


def look_ahead_seq(seq_in, t_width=16, padding=0, batch_first=0):
    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)

    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    # int(torch.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i < seq_len - t_width:
            seq_block = seq[i:i + t_width, :, :]
        else:
            seq_block = seq[i:, :, :]
            seq_block_pad = torch.zeros([t_width - (seq_len - i), n_batch, n_feature], dtype=torch.float32).cuda()
            seq_block = torch.cat((seq_block, seq_block_pad), 0)
        new_seq.append(seq_block)
    new_seq = torch.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq


def look_around_seq(seq_in, t_width=16, padding=0, batch_first=0):
    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)

    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    # int(torch.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i >= seq_len - t_width:
            seq_block = seq[i - t_width:, :, :]
            seq_block_pad = torch.zeros([t_width - (seq_len - i) + 1, n_batch, n_feature], dtype=torch.float32).cuda()
            seq_block = torch.cat((seq_block, seq_block_pad), 0)
        elif i < t_width:
            seq_block = seq[0:i + 1 + t_width, :, :]
            seq_block_pad = torch.zeros([t_width - i, n_batch, n_feature], dtype=torch.float32).cuda()
            seq_block = torch.cat((seq_block, seq_block_pad), 0)
        else:
            seq_block = seq[i - t_width:i + 1 + t_width, :, :]
        # print(seq_block.size())
        new_seq.append(seq_block)
    new_seq = torch.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq


def get_temporal_sparsity(list_layer, seq_len, threshold):
    # Evaluate Sparsity
    num_zeros = 0
    num_elems = 0
    # print(seq_len.size())
    # Iterate through nnlayers
    for layer in list_layer:
        all_delta_vec = layer.transpose(0, 1)
        all_delta_vec = torch.abs(all_delta_vec)  # Take absolute values of all delta vector elements
        for i, delta_vec in enumerate(all_delta_vec):
            seq = delta_vec[:seq_len[i], :]
            zero_mask = seq < threshold
            num_zeros += torch.sum(zero_mask)
            num_elems += torch.numel(zero_mask)
    sparsity = float(num_zeros) / float(num_elems)
    return sparsity


def gen_xilinx_sdk_clib(dir_path, file_name, dict_params):
    """
    Generate C Library for NN parameters
    :param dir_path: dir path to export the file
    :param filename: name of the C Library
    :param dict_params:    Dictionary from which to save numpy arrays of NN parameters in fp
    :param qdict:    Dictionary from which to save properties of corresponding variables in pdict in tuples
        Elements of each tuple in qdict in order:
        - quantize
        - qi:        number of integer bits before the decimal point in fxp
        - qf:        number of integer bits after the decimal point in fxp
        - var_class: 0 - Constant | 1 - General Tensors | 2 - NN Parameter |
        - print_row_size
    :return:         mat_param_fxp - parameter in fixed-point numbers
    """

    # Create file
    try:
        f_h = open(dir_path + '/' + file_name + '.h', 'w')
    except IOError as err:
        print("I/O error({0}): {1}".format(err.errno, err.strerror))
    try:
        f_c = open(dir_path + '/' + file_name + '.c', 'w')
    except IOError as err:
        print("I/O error({0}): {1}".format(err.errno, err.strerror))

    def get_var_type(quantize, bit_width):
        if quantize:
            # Get variable type
            if 0 < bit_width <= 8:
                var_type = 'signed char'
                num_aligned_bytes = 8
            elif 8 < bit_width <= 16:
                var_type = 'short'
                num_aligned_bytes = 8
            elif 16 < bit_width <= 32:
                var_type = 'long'
                num_aligned_bytes = 8
            else:
                raise ValueError('Bit width of parameter must not be larger than 32...')
        else:
            var_type = 'float'
            num_aligned_bytes = 16
        return var_type, num_aligned_bytes

    # Write macro definitions
    macro_name = file_name.upper() + '_H'
    f_h.write('#ifndef ' + macro_name + '\n' + '#define ' + macro_name + '\n')
    f_c.write('#include "' + file_name + '.h"\n')

    # Iterate over params in dict
    pdict_out = {}
    for key, value in dict_params.items():
        # Get quantization precision of the variable
        temp_value, cast, qi, qf, var_class, print_row_size = value
        temp_value = np.asarray(temp_value)
        bit_width = qi + qf

        # Convert fp to fxp
        if cast:
            pdict_out[key] = cast_fp_to_fxp(temp_value, qi, qf)
        else:
            pdict_out[key] = temp_value

        # Get variable type
        var_type, num_aligned_bytes = get_var_type(cast, bit_width)

        if var_class == 'const':  # Constant
            f_h.write('#define ' + key.upper() + ' %d\n' % pdict_out[key])
        elif var_class == 'tensor':  # General Tensor
            if pdict_out[key].ndim == 1:
                pdict_out[key] = np.reshape(pdict_out[key].shape[0], -1)

            # Get param dimension
            size = pdict_out[key].size
            dim0 = pdict_out[key].shape[0]
            dim1 = pdict_out[key].shape[1]

            # Write variable declaration
            f_h.write('/*\n'
                      ' * Var Type: General Matrix \n'
                      ' * Var Name: %s' % key + '\n'
                                                ' * Bit Width: %d' % bit_width + '\n'
                                                                                 ' * Dimension: (%d, %d)' % (
                          dim0, dim1) + '\n'
                                        ' */\n'
                      )
            f_h.write('#define ' + key.upper() + '_NUM_ROWS %d\n' % dim0)
            f_h.write('#define ' + key.upper() + '_NUM_COLS %d\n' % dim1)
            f_h.write('#define ' + key.upper() + '_MAT_SIZE %d\n' % size)
            f_h.write('extern const %s %s[' % (var_type, key) + key.upper() + '_MAT_SIZE];\n')
            f_c.write('const %s %s[' % (var_type, key) + key.upper() + '_MAT_SIZE] = {\n')

            # Write parameters column-wisely
            for i in range(0, dim1):
                for j in range(0, dim0):
                    if var_type == 'float':
                        f_c.write('%.20f' % pdict_out[key][j][i])
                    else:
                        f_c.write('%d' % pdict_out[key][j][i])
                    if not (i >= dim1 - 1 and j >= dim0 - 1):
                        f_c.write(',')
                    if (i * dim0 + j) % print_row_size == print_row_size - 1:
                        f_c.write('\n')
            f_c.write('};\n\n')

        elif var_class == 'param':  # NN Parameter
            if pdict_out[key].ndim == 1:
                pdict_out[key] = np.reshape(pdict_out[key].shape[0], -1)

            # Get param dimension
            size = pdict_out[key].size
            dim0 = pdict_out[key].shape[0]
            dim1 = pdict_out[key].shape[1]

            # Write variable declaration
            f_h.write('/*\n'
                      ' * Var Type: NN Parameter Matrix \n'
                      ' * Var Name: %s' % key + '\n'
                                                ' * Bit Width: %d' % bit_width + '\n'
                                                                                 ' * Dimension: (%d, %d)' % (
                          dim0, dim1) + '\n'
                                        ' */\n'
                      )

            f_h.write('#define ' + key.upper() + '_SIZE %d\n' % size)
            f_h.write('extern const %s %s[' % (
                var_type, key) + key.upper() + '_SIZE] __attribute__ ((aligned (%d)));\n' % num_aligned_bytes)
            f_c.write('const %s %s[' % (var_type, key) + key.upper() + '_SIZE] = {\n')

            # Write parameters column-wisely
            for i in range(0, dim1):
                for j in range(0, dim0):
                    write_val = pdict_out[key][j][i]
                    if isinstance(write_val, float):
                        f_c.write('%f' % write_val)
                    else:
                        f_c.write('%d' % write_val)
                    if not (i == dim1 - 1 and j == dim0 - 1):
                        f_c.write(',')
                    if (i * dim0 + j) % print_row_size == print_row_size - 1:
                        f_c.write('\n')
            f_c.write('};\n\n')

    f_h.write('#endif')
    f_c.close()
    f_h.close()

    return pdict_out

# def gen_sim_file(dir_path, dict_params, qi, qf):
#     """
#     Generate Modelsim Simulation Files for NN parameters
#     :param dir_path: dir path to export files
#     :param dict_params:    Dictionary from which to save numpy arrays of NN parameters in fp
#     :param qi:       number of integer bits before the decimal point in fxp
#     :param qf:       number of integer bits after the decimal point in fxp
#     :return:         mat_param_fxp - parameter in fixed-point numbers
#     """
#
#     # Get unsigned variable type
#     bit_width = qi + qf
#     if 0 < bit_width <= 8:
#         var_type = 'uint8'
#     elif 8 < bit_width <= 16:
#         var_type = 'uint16'
#     elif 16 < bit_width <= 32:
#         var_type = 'uint32'
#     else:
#         raise ValueError('Bit width of parameter must not be larger than 32...')
#
#     # Iterate over params in dict
#     pdict_out = {}
#     for key, value in dict_params.items():
#         # Convert fp to fxp
#         pdict_out[key] = cast_fp_to_fxp(value, qi, qf)
#
#         # Get param dimension
#         dim0 = value.shape[0]
#         dim1 = value.shape[1]
#         # Create file
#         try:
#             f = open(dir_path + '/' + key + '.txt', 'w')
#         except IOError as err:
#             print("I/O error({0}): {1}".format(err.errno, err.strerror))
#
#         # Write parameters column-wisely
#         param_hex = cast_fxp_to_hex(pdict_out[key], var_type)
#         for i in range(0, dim1):
#             for j in range(0, dim0):
#                 if var_type == 'uint8':
#                     f.write('%2.2X\n' % param_hex[j][i])
#                 elif var_type == 'uint16':
#                     f.write('%4.4X\n' % param_hex[j][i])
#                 elif var_type == 'uint32':
#                     f.write('%8.8X\n' % param_hex[j][i])
#                 else:
#                     f.write('%X\n' % param_hex[j][i])
#         f.close()
#
#     return pdict_out


def gen_sim_file(dir_path, dict_params, qi, qf, row_blocks=1):
    """
    Generate Modelsim Simulation Files for NN parameters

    :param dict_params:    Dictionary from which to save numpy arrays of NN parameters in fp
    :param qi:       number of integer bits before the decimal point in fxp
    :param qf:       number of integer bits after the decimal point in fxp
    :return:         mat_param_fxp - parameter in fixed-point numbers
    """

    # Get unsigned variable type
    bit_width = qi + qf
    if 0 < bit_width <= 8:
        var_type = 'uint8'
    elif 8 < bit_width <= 16:
        var_type = 'uint16'
    elif 16 < bit_width <= 32:
        var_type = 'uint32'
    else:
        raise ValueError('Bit width of parameter must not be larger than 32...')

    # Iterate over params in dict
    pdict_out = {}
    for key, value in dict_params.items():
        # Convert fp to fxp
        pdict_out[key] = cast_fp_to_fxp(value, qi, qf, use_floor=False)

        # Get param dimension
        n_rows = value.shape[0]
        n_cols = value.shape[1]

        # Create file
        try:
            f = open(os.path.join(dir_path, key + '.txt'), 'w')
        except:
            raise RuntimeError("Failed to create file.")

        def write_hex(f, x, var_type):
            x_hex = cast_fxp_to_hex(x, var_type)
            if var_type == 'uint8':
                f.write('%2.2X' % x_hex)
            elif var_type == 'uint16':
                f.write('%4.4X' % x_hex)
            elif var_type == 'uint32':
                f.write('%8.8X' % x_hex)
            else:
                f.write('%X' % x_hex)

        # Write parameters column-wisely
        param_print = pdict_out[key]
        for i_col in range(0, n_cols):
            block_buffer = np.zeros(row_blocks, dtype=np.int32)
            i_block = 0
            for i_row in range(0, n_rows):
                block_buffer[i_block] = param_print[i_row][i_col]
                i_block = i_block + 1
                if i_block == row_blocks:
                    for block_elem in reversed(block_buffer):
                        write_hex(f, block_elem, var_type)
                    f.write(' //')
                    for block_elem in reversed(block_buffer):
                        f.write(' %d,' % block_elem)
                    i_block = 0
                    f.write('\n')
        f.close()

    return pdict_out

def hardsigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.25 * x) + 0.5
    x = F.threshold(-x, -1.0, -1.0)
    x = F.threshold(-x, 0.0, 0.0)
    return x


def get_layer_sparsity(x: nn.Module):
    num_nonzeros = 0
    num_elements = 0
    for name, param in x.named_parameters():
        print(name)
    for name, param in x.named_parameters():
        num_nonzeros += len(torch.nonzero(param.data))
        num_elements += param.data.nelement()
    return float(1 - num_nonzeros/num_elements)