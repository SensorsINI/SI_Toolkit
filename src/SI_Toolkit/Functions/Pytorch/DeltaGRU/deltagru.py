__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import torch
import torch.nn as nn
import torch.nn.functional as F
from SI_Toolkit.Functions.Pytorch.DeltaGRU.util import quantize_tensor
from torch.nn import Parameter
from typing import Tuple, Optional
from torch import Tensor


def get_temporal_sparsity(delta_vec_list):
    n_zeros = 0
    n_elems = 0
    for delta_vec in delta_vec_list:
        n_zeros += torch.sum(delta_vec == 0).float().item()
        n_elems += float(torch.numel(delta_vec))
    sp = n_zeros / n_elems
    # self.dict_stats['sparsity_delta_x'] = sparsity_delta_x
    # self.dict_stats['sparsity_delta_h'] = sparsity_delta_h
    # self.dict_stats['sparsity_delta'] = sparsity_delta
    # return self.dict_stats
    return sp


def hard_sigmoid(x, qi, qf, q_enable):
    """
    Hard sigmoid function: y(x) = 0.25*x + 0.5

    :param x: input tensor
    :param qi: number of bit before decimal points for quantization
    :param qf: number of bit after decimal points for quantization
    :param q_enable: If = 1, enable quantization
    :return: output of the hard sigmoid funtion
    """
    x = quantize_tensor(0.25 * x, qi, qf, q_enable) + 0.5
    x = torch.clamp(x, 0, 1)
    return x


class DeltaGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bias=True,
                 batch_first=False,
                 benchmark=False,
                 thx=0.0,
                 thh=0.0675,
                 qa=1,
                 qaf=1,
                 aqi=8,
                 aqf=8,
                 afqi=2,
                 afqf=6,
                 eval_sp=0,
                 debug=0,
                 gru_layer: nn.Module = None
                 ):
        super(DeltaGRU, self).__init__()

        # Set type of DeltaGRU Implementation
        self.deltagru_type = 'edgedrnn'  # Change it to 'cudnn' if you want to match the cuDNN version used in PyTorch GRU

        # Properties
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.x_p_size = max(input_size, hidden_size)
        self.num_layers = num_layers
        self.thx = thx
        self.thh = thh
        self.qa = qa
        self.qaf = qaf
        self.aqi = aqi
        self.aqf = aqf
        self.afqi = afqi
        self.afqf = afqf
        self.eval_sp = eval_sp
        self.debug = debug
        self.benchmark = benchmark

        # GRU Attributes
        self.num_gates = 3
        self.gate_size = self.num_gates * self.hidden_size
        self.num_states = 1

        # Layer List
        self.layer_list = nn.ModuleList()

        # Statistics
        self.abs_sum_delta_hid = torch.zeros(1)
        self.sp_dx = 0
        self.sp_dh = 0

        # Log
        self.list_log = []
        self.reset_log()

        # Instantiate parameters
        for l in range(num_layers):
            layer_input_size = self.input_size if l == 0 else self.hidden_size
            w_ih = Parameter(torch.Tensor(self.gate_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            b_ih = Parameter(torch.Tensor(self.gate_size))
            b_hh = Parameter(torch.Tensor(self.gate_size))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            param_names = ['weight_ih_l{}', 'weight_hh_l{}']
            if bias:
                param_names += ['bias_ih_l{}', 'bias_hh_l{}']
            param_names = [x.format(l) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)

        # Load GRU Parameters if available
        if gru_layer is not None:
            self.load_gru_param(gru_layer)

    def reset_log(self):
        setattr(self, 'list_log', [{} for i in range(self.num_layers)])

    def log_var_append(self, l, key, var):
        if not self.training and not self.benchmark:
            if key not in self.list_log[l].keys():
                self.list_log[l][key] = []
            self.list_log[l][key].append(var.detach().cpu())

    def log_var(self, l, key, var):
        if not self.training and not self.benchmark:
            if key not in self.list_log[l].keys():
                self.list_log[l][key] = []
            self.list_log[l][key] = var.detach().cpu()
    # def reset_log_sparsity(self):
    #     setattr(self, log, [dict.fromkeys(self.vars, []) for i in range(self.num_layers)])

    def set_eval_sparsity(self, x):
        self.eval_sp = x
        for i in range(self.num_layers):
            self.layer_list[i].set_eval_sparsity(self.eval_sp)

    def deltagru_forward(self, input: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], l: int):
        """
        :param input: input tensor (seq_len, batch_size, feature_size)
        :param state: (x_p, h_0, h_p, ma_0, mb_0)
        :param l: index of the current layer
        :return: (output, state)
        """
        # Get Weights (Biases are already initialized in the state)
        weight_ih = getattr(self, 'weight_ih_l{}'.format(l))
        weight_hh = getattr(self, 'weight_hh_l{}'.format(l))
        # print(weight_ih)

        # Get Inputs
        inputs = input.unbind(0)

        # Initial States
        x_p_len = self.input_size if l == 0 else self.hidden_size
        x_p = quantize_tensor(state[0][:, :x_p_len], self.aqi, self.aqf, self.qa)
        x_p_next = quantize_tensor(torch.zeros_like(state[0]), self.aqi, self.aqf, self.qa)
        h = quantize_tensor(state[1], self.aqi, self.aqf, self.qa)
        h_p = quantize_tensor(state[2], self.aqi, self.aqf, self.qa)
        ma = state[3]  # Contain biases
        mb = state[4]  # Contain biases
        one = torch.ones_like(h, device=h.device)

        # Iterate through timesteps
        output = []
        seq_len = len(inputs)
        for i in range(seq_len):
            # Get Current Input
            x = inputs[i]

            # Get raw delta vectors
            delta_x = x - x_p
            delta_h = h - h_p
            self.log_var_append(l, 'pa_x', x)
            self.log_var_append(l, 'pa_h_t-1', h)
            self.log_var_append(l, 'pa_x_p', x_p)
            self.log_var_append(l, 'pa_h_p', h_p)

            # Zero-out elements of delta vectors below the threshold
            delta_x_abs = torch.abs(delta_x)
            delta_x = delta_x.masked_fill_(delta_x_abs < self.thx, 0)
            delta_h_abs = torch.abs(delta_h)
            delta_h = delta_h.masked_fill(delta_h_abs < self.thh, 0)

            # Update previous state vectors memory on indices that had above-threshold change
            x_p = torch.where(delta_x_abs >= self.thx, x, x_p)
            x_p_next[:, :x_p_len] = x_p
            h_p = torch.where(delta_h_abs >= self.thh, h, h_p)
            h_p_next = h_p

            # Calculate Partial Sums
            psum_x = F.linear(delta_x, weight_ih, ma)
            psum_h = F.linear(delta_h, weight_hh)
            psum_x_chunks = psum_x.chunk(3, dim=1)
            psum_h_chunks = psum_h.chunk(3, dim=1)

            # Get Delta Memory Vectors for the next timestep
            m_r = psum_x_chunks[0] + psum_h_chunks[0]  # Reset Gate
            m_u = psum_x_chunks[1] + psum_h_chunks[1]  # Update Gate
            m_cx = psum_x_chunks[2]  # Cell State w.r.t. input
            m_ch = psum_h_chunks[2] + mb  # Cell State w.r.t. hidden
            ma = torch.cat((m_r, m_u, m_cx), -1)
            mb = m_ch

            # Calculate Reset Gate
            pre_r = quantize_tensor(m_r, self.aqi, self.aqf, self.qa)
            r = torch.sigmoid(pre_r)
            r = quantize_tensor(r, self.afqi, self.afqf, self.qaf)

            # Calculate Update Gate
            pre_u = quantize_tensor(m_u, self.aqi, self.aqf, self.qa)
            u = torch.sigmoid(pre_u)
            u = quantize_tensor(u, self.afqi, self.afqf, self.qaf)

            # Calculate Cell State
            q_m_ch = quantize_tensor(m_ch, self.aqi, self.aqf, self.qa)
            r_times_q_m_ch = torch.mul(r, q_m_ch)
            m_c = m_cx + r_times_q_m_ch
            pre_c = quantize_tensor(m_c, self.aqi, self.aqf, self.qa)
            c = torch.tanh(pre_c)
            c = quantize_tensor(c, self.afqi, self.afqf, self.qaf)

            # Calculate GRU Output
            one_minus_u = one - u
            a = quantize_tensor(torch.mul(one_minus_u, c), self.aqi, self.aqf, self.qa)
            b = quantize_tensor(torch.mul(u, h), self.aqi, self.aqf, self.qa)
            a_plus_b = a + b
            h = quantize_tensor(a_plus_b, self.aqi, self.aqf, self.qa)

            # Collect Outputs
            output += [h]

            # Log
            self.log_var_append(l, 'pa_delta_x', delta_x)
            self.log_var_append(l, 'pa_delta_h', delta_h)
            self.log_var_append(l, 'pa_x_p_next', x_p_next)
            self.log_var_append(l, 'pa_h_p_next', h_p_next)
            self.log_var_append(l, 'pacc_m_r', m_r)
            self.log_var_append(l, 'pacc_m_u', m_u)
            self.log_var_append(l, 'pacc_m_cx', m_cx)
            self.log_var_append(l, 'pacc_m_ch', m_ch)
            self.log_var_append(l, 'pa_q_m_ch', q_m_ch)
            self.log_var_append(l, 'pacc_m_c', m_c)
            self.log_var_append(l, 'pacc_ma', ma)
            self.log_var_append(l, 'pacc_mb', mb)
            self.log_var_append(l, 'pa_pre_r', pre_r)
            self.log_var_append(l, 'pa_pre_u', pre_u)
            self.log_var_append(l, 'pa_pre_c', pre_c)
            self.log_var_append(l, 'paa_r_times_q_m_ch', r_times_q_m_ch)
            self.log_var_append(l, 'pa_r', r)
            self.log_var_append(l, 'pa_u', u)
            self.log_var_append(l, 'pa_c', c)
            self.log_var_append(l, 'pa_a', a)
            self.log_var_append(l, 'pa_b', b)
            self.log_var_append(l, 'pa_one_minus_u', one_minus_u)
            self.log_var_append(l, 'pa_a_plus_b', a_plus_b)
            self.log_var_append(l, 'pa_h', h)

        if not self.training and not self.benchmark:
            for key in self.list_log[l]:
                self.list_log[l][key] = torch.stack(self.list_log[l][key], dim=0)

            # Log parameters
            self.log_var(l, 'weight_ih', weight_ih)
            self.log_var(l, 'weight_hh', weight_hh)
            bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
            bias_hh = getattr(self, 'bias_hh_l{}'.format(l))
            self.log_var(l, 'bias_ih', bias_ih)
            self.log_var(l, 'bias_hh', bias_hh)

        output = torch.stack(output)
        state = (x_p_next, h, h_p_next, ma, mb)

        return output, state

    def forward(self, input: Tensor, state: Optional[Tuple] = None, quantize: int = 0, feature_lengths=None):
        # Reset Log
        if not self.training:
            self.reset_log()

        # Reshape input if necessary
        x = input
        if self.batch_first:
            x = input.transpose(0, 1)
        setattr(self, 'batch_size', int(x.size(1)))

        x = quantize_tensor(x, self.aqi, self.aqf, self.qa)

        # Set Runtime Quantization
        # self.set_qa_rt()

        # Initializers
        self.init_input = torch.zeros(self.batch_size, self.x_p_size,
                                 dtype=x.dtype, device=x.device)
        self.init_state = torch.zeros(self.batch_size, self.hidden_size,
                                 dtype=x.dtype, device=x.device)
        self.init_gates = torch.zeros(self.batch_size, self.gate_size,
                                 dtype=x.dtype, device=x.device)
        # Initialize States
        if state is None:  # state = (x_p, h_0, h_p, ma_0, mb_0)
            state = self.initialze_state()
        else:
            for l in range(self.num_layers):
                state.append((state[0][l], state[1][l], state[2][l], state[3][l], state[4][l]))

        # Iterate through layers
        layer_state = []
        for l in range(self.num_layers):
            # Forward Propation of Layer
            x, state_next = self.deltagru_forward(x, state[l], l)
            layer_state += [list(state_next)]

        # Concat layer states
        state_next = tuple([torch.stack([layer_state[i_layer][i_state] for i_layer in range(self.num_layers)])
                            for i_state in range(self.num_states)])

        # Get Statistics
        # if not self.training:
        #     self.get_temporal_sparsity()
        #     # self.get_stats()

        if self.batch_first:
            x = x.transpose(0, 1)



        return x, state_next

    def initialze_state(self):
        state = []
        for l in range(self.num_layers):
            bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
            bias_hh = getattr(self, 'bias_hh_l{}'.format(l))
            if self.deltagru_type == 'edgedrnn':  # To be compatible with EdgeDRNN
                init_ma = self.init_gates + bias_ih  # Only use bias_ih. bias_hh becomes a placeholder.
                init_mb = self.init_state
            else:  # To match cuDNN GRU, use both bias_ih and bias_hh
                bias_ih_chunks = bias_ih.chunk(self.num_gates)
                bias_hh_chunks = bias_hh.chunk(self.num_gates)
                init_ma = self.init_gates + torch.cat(
                    (bias_ih_chunks[0] + bias_hh_chunks[0], bias_ih_chunks[1] + bias_hh_chunks[1], bias_ih_chunks[2]))
                init_mb = self.init_state + bias_hh_chunks[2]
            state.append((self.init_input, self.init_state, self.init_state, init_ma, init_mb))
        return state

    def process_biases(self):
        # In default, we use the DeltaGRU equations in the EdgeDRNN paper
        with torch.no_grad():
            if self.deltagru_type == 'edgedrnn':
                for l in range(self.num_layers):
                    bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
                    bias_hh = getattr(self, 'bias_hh_l{}'.format(l))
                    bias_ih_chunks = bias_ih.chunk(self.num_gates)
                    bias_hh_chunks = bias_hh.chunk(self.num_gates)
                    bias_ih = torch.cat(
                        (bias_ih_chunks[0] + bias_hh_chunks[0], bias_ih_chunks[1] + bias_hh_chunks[1],
                         bias_ih_chunks[2]))
                    bias_hh = bias_hh.masked_fill_(bias_hh != 0, 0)