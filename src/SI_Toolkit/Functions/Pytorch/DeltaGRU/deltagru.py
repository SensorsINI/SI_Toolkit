import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from SI_Toolkit.Functions.Pytorch.DeltaGRU.util import quantize_tensor, hardsigmoid
from typing import Tuple, Optional

class DeltaGRU(nn.GRU):
    def __init__(self,
                 input_size=16,
                 hidden_size=256,
                 num_layers=2,
                 batch_first=False,
                 thx=0,
                 thh=0,
                 qa=0,
                 aqi=8,
                 aqf=8,
                 qw=0,
                 wqi=1,
                 wqf=7,
                 nqi=2,
                 nqf=4,
                 bw_acc=32,
                 use_hardsigmoid=0,
                 use_hardtanh=0,
                 debug=0):
        super(DeltaGRU, self).__init__(input_size, hidden_size, num_layers)

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.th_x = thx
        self.th_h = thh
        self.qa = qa
        self.aqi = aqi
        self.aqf = aqf
        self.qw = qw
        self.wqi = wqi
        self.wqf = wqf
        self.nqi = nqi
        self.nqf = nqf
        self.bw_acc = bw_acc
        self.debug = debug
        self.weight_ih_height = 3 * self.hidden_size  # Wih has 4 weight matrices stacked vertically
        self.weight_ih_width = self.input_size
        self.weight_hh_width = self.hidden_size
        self.use_hardsigmoid = use_hardsigmoid
        self.use_hardtanh = use_hardtanh
        self.x_p_length = max(self.input_size, self.hidden_size)
        self.num_gates = 3

        # Bias_nh
        self.bias_nh = torch.zeros(1, requires_grad=False)

        # Statistics
        self.abs_sum_delta_hid = torch.zeros(1)
        self.sp_dx = 0
        self.sp_dh = 0

        # Log for Debug
        self.list_log = []
        self.reset_log()

        # Debug
        self.set_debug(self.debug)

    def reset_log(self):
        setattr(self, 'list_log', [{} for i in range(self.num_layers)])

    def log_var_append(self, l, key, var):
        if not self.training and self.debug:
            if key not in self.list_log[l].keys():
                self.list_log[l][key] = []
            self.list_log[l][key].append(var.detach().cpu())

    def log_var(self, l, key, var):
        if not self.training and self.debug:
            if key not in self.list_log[l].keys():
                self.list_log[l][key] = []
            self.list_log[l][key] = var.detach().cpu()

    def set_debug(self, value):
        setattr(self, "debug", value)
        self.statistics = {
            "num_dx_zeros": 0,
            "num_dx_numel": 0,
            "num_dh_zeros": 0,
            "num_dh_numel": 0
        }

    def add_to_debug(self, x, i_layer, name):
        if self.debug and not self.training:
            if isinstance(x, Tensor):
                variable = np.squeeze(x.cpu().numpy())
            else:
                variable = np.squeeze(np.asarray(x))
            variable_name = '_'.join(['l' + str(i_layer), name])
            if variable_name not in self.statistics.keys():
                self.statistics[variable_name] = []
            self.statistics[variable_name].append(variable)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'l0' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param[:self.hidden_size, :])
                    nn.init.xavier_uniform_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.xavier_uniform_(param[2 * self.hidden_size:3 * self.hidden_size, :])
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
            else:
                if 'weight' in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
            if 'bias' in name:
                nn.init.constant_(param, 0)
        print("--------------------------------------------------------------------")

    def get_temporal_sparsity(self):
        temporal_sparsity = {}
        if self.debug and not self.training:
            temporal_sparsity["SP_T_DX"] = float(self.statistics["num_dx_zeros"] / self.statistics["num_dx_numel"])
            temporal_sparsity["SP_T_DH"] = float(self.statistics["num_dh_zeros"] / self.statistics["num_dh_numel"])
            temporal_sparsity["SP_T_DV"] = float((self.statistics["num_dx_zeros"] + self.statistics["num_dh_zeros"]) /
                                                 (self.statistics["num_dx_numel"] + self.statistics["num_dh_numel"]))
        self.statistics.update(temporal_sparsity)
        return temporal_sparsity

    def initialize_state(self, x: Tensor):
        """
        Initialize DeltaGRU States Inputs (please refer to the DeltaGRU formulations)
        :param x:       x(t), Input Tensor
        :return: state, which is a tuple of 5 Tensors:
            (1) Tensor x_p_0:   x(t-1), Input Tensor
            (2) Tensor h_0:     h(t-1), Hidden state
            (3) Tensor h_p_0:   h(t-2), Hidden state
            (4) Tensor dm_ch_0: dm_ch(t-1), Delta Memory for next gate hidden MxV
            (5) Tensor dm_0:    dm(t-1), Delta Memory
        """
        # Get Batch Size
        batch_size = x.size(0) if self.batch_first else x.size(1)

        # Generate zero state if external state not provided
        x_p_0 = torch.zeros(self.num_layers, batch_size, self.x_p_length,
                            dtype=x.dtype, device=x.device)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        h_p_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        dm_nh_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        dm_0 = torch.zeros(self.num_layers, batch_size, self.weight_ih_height, dtype=x.dtype, device=x.device)
        for l in range(self.num_layers):
            bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
            dm_0[l, :, :self.hidden_size] = dm_0[l, :, :self.hidden_size] + bias_ih[:self.hidden_size]
            dm_0[l, :, self.hidden_size:2 * self.hidden_size] = dm_0[l, :, self.hidden_size:2 * self.hidden_size] + bias_ih[self.hidden_size:2 * self.hidden_size]
            dm_0[l, :, 2 * self.hidden_size:3 * self.hidden_size] = dm_0[l, :, 2 * self.hidden_size:3 * self.hidden_size] + bias_ih[2 * self.hidden_size:3 * self.hidden_size]
            # dm_nh_0[l, :, :] = dm_nh_0[l, :, :] + self.bias_nh.to(x.device)
            # dm_nh_0[l, :, :] = torch.zeros_like()
        state = (x_p_0, h_0, h_p_0, dm_nh_0, dm_0)
        return state

    def layer_forward(self, input: Tensor, idx_layer: int, qa: int, state: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        # Get States
        x_p_0, h_0, h_p_0, dm_nh_0, dm_0 = state
        x_p_0 = x_p_0[idx_layer]
        h_0 = h_0[idx_layer]
        h_p_0 = h_p_0[idx_layer]
        dm_nh_0 = dm_nh_0[idx_layer]
        dm_0 = dm_0[idx_layer]

        # Get Layer Parameters
        weight_ih = getattr(self, 'weight_ih_l{}'.format(idx_layer))
        weight_hh = getattr(self, 'weight_hh_l{}'.format(idx_layer))

        # Get Feature Dimension
        input_size = input.size(-1)
        batch_size = input.size(1)

        # Quantize threshold
        th_x = quantize_tensor(torch.tensor(self.th_x, dtype=input.dtype), self.aqi, self.aqf, qa)
        th_h = quantize_tensor(torch.tensor(self.th_h, dtype=input.dtype), self.aqi, self.aqf, qa)

        # Get Layer Inputs
        inputs = quantize_tensor(input, self.aqi, self.aqf, qa)
        inputs = inputs.unbind(0)

        # Collect Layer Outputs
        output = []

        # Regularizer
        reg = torch.zeros(1, dtype=input.dtype, device=input.device).squeeze()

        # Iterate through time steps
        x_p_out = torch.zeros(batch_size, self.x_p_length,
                              dtype=input.dtype, device=input.device)
        x_p = quantize_tensor(x_p_0[:, :input_size], self.aqi, self.aqf, qa)
        x_prev_out_size = torch.zeros_like(x_p)
        x_prev_out = quantize_tensor(x_prev_out_size, self.aqi, self.aqf, qa)
        h = quantize_tensor(h_0, self.aqi, self.aqf, qa)
        h_p = quantize_tensor(h_p_0, self.aqi, self.aqf, qa)
        dm_nh = quantize_tensor(dm_nh_0, self.aqi, self.aqf, qa)
        dm = dm_0
        l1_norm_delta_h = torch.zeros(1, dtype=input.dtype)  # Intialize L1 Norm of delta h

        # Iterate through timesteps
        seq_len = len(inputs)
        for t in range(seq_len):
            # Get current input vectors
            x = inputs[t]

            # Get Delta Vectors
            delta_x = x - x_p
            delta_h = h - h_p
            self.log_var_append(idx_layer, 'pa_x', x)
            self.log_var_append(idx_layer, 'pa_h_t-1', h)
            self.log_var_append(idx_layer, 'pa_x_p', x_p)
            self.log_var_append(idx_layer, 'pa_h_p', h_p)

            # Zero-out elements of delta vector below the threshold
            delta_x_abs = torch.abs(delta_x)
            delta_x = delta_x.masked_fill(delta_x_abs < th_x, 0)
            delta_h_abs = torch.abs(delta_h)
            delta_h = delta_h.masked_fill(delta_h_abs < th_h, 0)

            reg += torch.sum(torch.abs(delta_h))

            # if not self.training and self.debug:
            if self.debug and not self.training:
                zero_mask_delta_x = torch.as_tensor(delta_x == 0, dtype=x.dtype)
                zero_mask_delta_h = torch.as_tensor(delta_h == 0, dtype=x.dtype)
                self.statistics["num_dx_zeros"] += torch.sum(zero_mask_delta_x)
                self.statistics["num_dh_zeros"] += torch.sum(zero_mask_delta_h)
                self.statistics["num_dx_numel"] += torch.numel(delta_x)
                self.statistics["num_dh_numel"] += torch.numel(delta_h)

            # Update previous state vectors memory on indices that had above-threshold change
            x_p = torch.where(delta_x_abs >= self.th_x, x, x_p)
            x_prev_out[:, :input.size(-1)] = x_p
            h_p = torch.where(delta_h_abs >= self.th_h, h, h_p)

            # Get l1 norm of delta_h
            l1_norm_delta_h += torch.sum(torch.abs(delta_h.cpu()))

            # Run forward pass for one time step
            mac_x = torch.mm(delta_x, weight_ih.t()) + dm
            mac_h = torch.mm(delta_h, weight_hh.t())
            mac_x_chunks = mac_x.chunk(3, dim=1)
            mac_h_chunks = mac_h.chunk(3, dim=1)
            dm_r = mac_x_chunks[0] + mac_h_chunks[0]
            dm_z = mac_x_chunks[1] + mac_h_chunks[1]
            dm_n = mac_x_chunks[2]
            dm_nh = mac_h_chunks[2] + dm_nh
            dm = torch.cat((dm_r, dm_z, dm_n), 1)

            pre_act_r = quantize_tensor(dm_r, self.aqi, self.aqf, qa)
            pre_act_z = quantize_tensor(dm_z, self.aqi, self.aqf, qa)

            # Compute reset (r) and update (z) gates
            gate_r = hardsigmoid(pre_act_r) if self.use_hardsigmoid else torch.sigmoid(pre_act_r)
            gate_z = hardsigmoid(pre_act_z) if self.use_hardsigmoid else torch.sigmoid(pre_act_z)
            q_r = quantize_tensor(gate_r, self.nqi, self.nqf, qa)
            q_z = quantize_tensor(gate_z, self.nqi, self.nqf, qa)

            # Compute next gate (n)
            pre_act_nh = quantize_tensor(dm_nh, self.aqi, self.aqf, qa)
            pre_act_n = quantize_tensor(dm_n + torch.mul(q_r, pre_act_nh), self.aqi, self.aqf, qa)
            gate_n = F.hardtanh(pre_act_n) if self.use_hardtanh else torch.tanh(pre_act_n)
            q_n = quantize_tensor(gate_n, self.nqi, self.nqf, qa)

            # Compute candidate memory
            one_minus_u = torch.ones_like(q_z, device=q_z.device) - q_z
            a = quantize_tensor(torch.mul(one_minus_u, q_n), self.aqi, self.aqf, qa)
            b = quantize_tensor(torch.mul(q_z, h), self.aqi, self.aqf, self.qa)
            h = a + b
            h = quantize_tensor(h, self.aqi, self.aqf, qa)

            # Append current DeltaLSTM hidden output to the list
            output += [h]

            # Log
            self.log_var_append(idx_layer, 'pa_delta_x', delta_x)
            self.log_var_append(idx_layer, 'pa_delta_h', delta_h)
            self.log_var_append(idx_layer, 'pa_x_p_next', x_p)
            self.log_var_append(idx_layer, 'pa_h_p_next', h_p)
            self.log_var_append(idx_layer, 'pacc_m_r', dm_r)
            self.log_var_append(idx_layer, 'pacc_m_u', dm_z)
            self.log_var_append(idx_layer, 'pacc_m_cx', dm_n)
            self.log_var_append(idx_layer, 'pacc_m_ch', dm_nh)
            self.log_var_append(idx_layer, 'pa_q_m_ch', pre_act_nh)
            self.log_var_append(idx_layer, 'pacc_ma', dm)
            self.log_var_append(idx_layer, 'pacc_mb', dm_n)
            self.log_var_append(idx_layer, 'pa_pre_r', pre_act_r)
            self.log_var_append(idx_layer, 'pa_pre_u', pre_act_z)
            self.log_var_append(idx_layer, 'pa_pre_c', pre_act_n)
            # self.log_var_append(idx_layer, 'paa_r_times_q_m_ch', r_times_q_m_ch)
            self.log_var_append(idx_layer, 'pa_r', gate_r)
            self.log_var_append(idx_layer, 'pa_u', gate_z)
            self.log_var_append(idx_layer, 'pa_c', gate_n)
            self.log_var_append(idx_layer, 'pa_a', a)
            self.log_var_append(idx_layer, 'pa_b', b)
            self.log_var_append(idx_layer, 'pa_one_minus_u', one_minus_u)
            # self.log_var_append(idx_layer, 'pa_a_plus_b', a_plus_b)
            self.log_var_append(idx_layer, 'pa_h', h)

        if not self.training and self.debug:
            for key in self.list_log[idx_layer]:
                self.list_log[idx_layer][key] = torch.stack(self.list_log[idx_layer][key], dim=0)

            # Log parameters
            self.log_var(idx_layer, 'weight_ih', weight_ih)
            self.log_var(idx_layer, 'weight_hh', weight_hh)
            bias_ih = getattr(self, 'bias_ih_l{}'.format(idx_layer))
            bias_hh = getattr(self, 'bias_hh_l{}'.format(idx_layer))
            self.log_var(idx_layer, 'bias_ih', bias_ih)
            self.log_var(idx_layer, 'bias_hh', bias_hh)

        output = torch.stack(output)
        x_p_out[:, :input_size] = x_p
        state_next = (x_p_out, h, h_p, dm_nh, dm)
        return output, state_next, reg

    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = None):
        # Reset Log
        if not self.training:
            self.reset_log()

        # Quantize
        qa = 0 if self.training else self.qa

        # Initialize State
        if state is None:
            state = self.initialize_state(input)

        # Reshape input if necessary
        x = input
        if self.batch_first:  # Force x shape: (Time, Batch, Feature)
            x = x.transpose(0, 1)
        setattr(self, 'batch_size', int(x.size()[1]))
        x = quantize_tensor(x, self.aqi, self.aqf, qa)

        # Iterate through layers
        reg = torch.zeros(1, dtype=x.dtype, device=input.device).squeeze()
        x_p_n = []
        h_n = []
        h_p_n = []
        dm_nh_n = []
        dm_n = []
        for l in range(self.num_layers):
            x, (x_p_n_l, h_n_l, h_p_n_l, dm_nh_n_l, dm_n_l), reg_l = self.layer_forward(x, l, qa, state)
            x_p_n.append(x_p_n_l)
            h_n.append(h_n_l)
            h_p_n.append(h_p_n_l)
            dm_nh_n.append(dm_nh_n_l)
            dm_n.append(dm_n_l)
            reg += reg_l
        x_p_n = torch.stack(x_p_n)  # Next h(t-1)
        h_n = torch.stack(h_n)  # Next h(t-1)
        h_p_n = torch.stack(h_p_n)  # Next h(t-2)
        dm_nh_n = torch.stack(dm_nh_n)  # Next M_nh(t-1)
        dm_n = torch.stack(dm_n)    # Next M(t-1)
        state_next = (x_p_n, h_n, h_p_n, dm_nh_n, dm_n)

        # Debug
        if self.debug and not self.training:
            self.statistics["sparsity_dx"] = float(self.statistics["num_dx_zeros"] / self.statistics["num_dx_numel"])
            self.statistics["sparsity_dh"] = float(self.statistics["num_dh_zeros"] / self.statistics["num_dh_numel"])
            self.statistics["sparsity_to"] = float((self.statistics["num_dx_zeros"] + self.statistics["num_dh_zeros"]) /
                                                   (self.statistics["num_dx_numel"] + self.statistics["num_dh_numel"]))

        if self.batch_first:
            x = x.transpose(0, 1)
        return x, state_next, reg

    def process_biases(self):
        # In default, we use the DeltaGRU equations in the EdgeDRNN paper
        with torch.no_grad():
            for l in range(self.num_layers):
                bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
                bias_hh = getattr(self, 'bias_hh_l{}'.format(l))
                bias_ih_chunks = bias_ih.chunk(self.num_gates)
                bias_hh_chunks = bias_hh.chunk(self.num_gates)
                bias_ih = torch.cat(
                    (bias_ih_chunks[0] + bias_hh_chunks[0], bias_ih_chunks[1] + bias_hh_chunks[1],
                     bias_ih_chunks[2]))
                # self.bias_nh = bias_hh_chunks[2].clone()
                # self.bias_nh = torch.zeros_like(self.bias_nh)
                # self.bias_nh.requires_grad = False
                bias_hh = bias_hh.masked_fill_(bias_hh != 0, 0)
                bias_hh.requires_grad = False
