"""
DeltaGRU algorithm by Chang Gao.
Do not modify this file!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
from torch.autograd.function import Function
# from modules.util import quantize_tensor
from SI_Toolkit.Functions.Pytorch.DeltaGRU.util import quantize_tensor


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


class DeltaGRUCell(nn.Module):
    def __init__(self,
                 n_inp,
                 n_hid,
                 th_x=0.25,
                 th_h=0.25,
                 qa=0,
                 aqi=8,
                 aqf=8,
                 nqi=2,
                 nqf=4,
                 eval_sp=0,
                 debug=0,
                 use_cuda=1):
        super(DeltaGRUCell, self).__init__()

        # Properties
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.th_x = torch.tensor(th_x).float()
        self.th_h = torch.tensor(th_h).float()
        self.aqi = torch.tensor(aqi).int()
        self.aqf = torch.tensor(aqf).int()
        self.nqi = torch.tensor(nqi).int()
        self.nqf = torch.tensor(nqf).int()
        self.eval_sp = eval_sp
        self.qa = torch.tensor(qa).int()
        self.use_cuda = use_cuda

        # Whether use GPU
        if self.use_cuda:
            self.th_x = self.th_x.cuda()
            self.th_h = self.th_h.cuda()

        # Network parameters
        self.weight_ih = torch.nn.Parameter(torch.empty(3 * n_hid, n_inp))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * n_hid, n_hid))
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * n_hid))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * n_hid))

        # Statistics
        self.delta_inp = []
        self.delta_hid = []

        # Regularizer
        # self.abs_delta_hid = torch.zeros(1).float()

        # Debug
        self.debug = debug
        if self.debug:
            self.dict_debug = {}
            self.dict_debug['x'] = []
            self.dict_debug['h'] = []
            self.dict_debug['dx'] = []
            self.dict_debug['dh'] = []
            self.dict_debug['q_mem_r'] = []
            self.dict_debug['mem_r'] = []
            self.dict_debug['q_mem_u'] = []
            self.dict_debug['mem_u'] = []
            self.dict_debug['mem_cx'] = []
            self.dict_debug['q_mem_ch'] = []
            self.dict_debug['mem_ch'] = []
            self.dict_debug['q_acc_c'] = []
            self.dict_debug['acc_c'] = []
            self.dict_debug['r'] = []
            self.dict_debug['u'] = []
            self.dict_debug['c'] = []
            self.dict_debug['a'] = []
            self.dict_debug['b'] = []
            self.dict_debug['one_minus_u'] = []
            self.dict_debug['stat_cnt_idx_dx'] = []
            self.dict_debug['stat_cnt_idx_dh'] = []

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.constant_(self.bias_ih, 0)
        nn.init.constant_(self.bias_hh, 0)

    def timestep(self, delta_inp, delta_hid, hidden, mem_x_prev, mem_ch_prev, one):
        # print("QA: ", self.qa)
        acc_x_curr = F.linear(delta_inp, self.weight_ih, mem_x_prev)
        acc_h_curr = F.linear(delta_hid, self.weight_hh)
        acc_x_chunks = acc_x_curr.chunk(3, dim=1)
        acc_h_chunks = acc_h_curr.chunk(3, dim=1)
        # print("acc_x_chunks: Max: %3.4f | Min: %3.4f" % (torch.max(acc_x_curr), torch.min(acc_x_curr)))
        # print("acc_h_chunks: Max: %3.4f | Min: %3.4f" % (torch.max(acc_h_curr), torch.min(acc_h_curr)))

        mem_r_curr = acc_x_chunks[0] + acc_h_chunks[0]
        mem_u_curr = acc_x_chunks[1] + acc_h_chunks[1]
        mem_cx_curr = acc_x_chunks[2]
        mem_ch_curr = acc_h_chunks[2] + mem_ch_prev

        # Quantize Accumulation
        # mem_all = torch.cat((mem_r_curr, mem_u_curr, mem_cx_curr, mem_ch_curr), 1)
        # mem_all_max = torch.max(mem_all)
        # mem_all_min = torch.min(mem_all)
        # print("Max: ", mem_all_max)
        # print("Min: ", mem_all_min)
        # mem_r_curr = quantize_tensor(mem_r_curr, self.aqi+1, self.aqf+7, self.qa)
        # mem_u_curr = quantize_tensor(mem_u_curr, self.aqi+1, self.aqf+7, self.qa)
        # mem_cx_curr = quantize_tensor(mem_cx_curr, self.aqi+1, self.aqf+7, self.qa)
        # mem_ch_curr = quantize_tensor(mem_ch_curr, self.aqi+1, self.aqf+7, self.qa)
        mem_x_curr = torch.cat((mem_r_curr, mem_u_curr, mem_cx_curr), 1)


        # print("Max: %f | Min: %f" % (torch.max(mem_x_curr), torch.min(mem_x_curr)))

        # mem_x_curr = quantize_tensor(mem_x_curr, self.aqi+1, self.aqf+7, self.qa)

        # Quantize Delta Memories
        q_mem_r_curr = quantize_tensor(mem_r_curr, self.aqi, self.aqf, self.qa)
        q_mem_u_curr = quantize_tensor(mem_u_curr, self.aqi, self.aqf, self.qa)

        # Calculate reset gate and update gate
        # r = hard_sigmoid(q_mem_r_curr, self.aqi, self.aqf, self.qa)
        # u = hard_sigmoid(q_mem_u_curr, self.aqi, self.aqf, self.qa)

        r = torch.sigmoid(q_mem_r_curr)
        u = torch.sigmoid(q_mem_u_curr)
        r = quantize_tensor(r, self.nqi, self.nqf, 1)
        u = quantize_tensor(u, self.nqi, self.nqf, 1)

        # Quantize mem_ch_curr
        q_mem_ch_curr = quantize_tensor(mem_ch_curr, self.aqi, self.aqf, self.qa)

        # Calculate candidate accumulation
        acc_c = mem_cx_curr + torch.mul(r, q_mem_ch_curr)
        q_acc_c = quantize_tensor(acc_c, self.aqi, self.aqf, self.qa)

        # Calculate candidate
        # c = F.hardtanh(q_acc_c)
        c = torch.tanh(q_acc_c)
        c = quantize_tensor(c, self.nqi, self.nqf, 1)

        # Quantize candidate
        # if self.use_cuda:
        #     one_minus_u = (torch.ones_like(u).cuda() - u)
        # else:
        #     one_minus_u = (torch.ones_like(u) - u)
        one_minus_u = one - u
        a = quantize_tensor(torch.mul(one_minus_u, c), self.aqi, self.aqf, self.qa)
        b = quantize_tensor(torch.mul(u, hidden), self.aqi, self.aqf, self.qa)
        next_hidden = a + b

        # Debug
        if self.debug:
            self.dict_debug['q_mem_r'].append(torch.squeeze(q_mem_r_curr))
            self.dict_debug['mem_r'].append(torch.squeeze(mem_r_curr))
            self.dict_debug['q_mem_u'].append(torch.squeeze(q_mem_u_curr))
            self.dict_debug['mem_u'].append(torch.squeeze(mem_u_curr))
            self.dict_debug['mem_cx'].append(torch.squeeze(mem_cx_curr))
            self.dict_debug['q_mem_ch'].append(torch.squeeze(q_mem_ch_curr))
            self.dict_debug['mem_ch'].append(torch.squeeze(mem_ch_curr))
            self.dict_debug['acc_c'].append(torch.squeeze(acc_c))
            self.dict_debug['q_acc_c'].append(torch.squeeze(q_acc_c))
            self.dict_debug['r'].append(torch.squeeze(r))
            self.dict_debug['u'].append(torch.squeeze(u))
            self.dict_debug['c'].append(torch.squeeze(c))
            self.dict_debug['a'].append(torch.squeeze(a))
            self.dict_debug['b'].append(torch.squeeze(b))
            self.dict_debug['one_minus_u'].append(torch.squeeze(one_minus_u))

        return next_hidden, mem_x_curr, mem_ch_curr

    def delta_forward(self, x, h_0=None):

        # Get Input Tensor Dimensions
        max_seq_len = x.size()[0]
        n_batch = x.size()[1]

        # Initialize result accumulator
        # Quantize candidate
        # if self.use_cuda:
        #     one_minus_u = (torch.ones_like(u).cuda() - u)
        # else:
        #     one_minus_u = (torch.ones_like(u) - u)

        if h_0 is None:
            hidden = torch.zeros(n_batch, self.n_hid).float()  # Initialize hidden state
        else:
            hidden = h_0.float()
        input_prev = torch.zeros(n_batch, self.n_inp).float()  # Initialize previous input state
        hidden_prev = torch.zeros(n_batch, self.n_hid).float()  # Initialize previous hidden state
        # mem_x = torch.unsqueeze(self.bias_ih, dim=0).repeat(n_batch, 1).float()  # Initialize mem_x
        # mem_ch = torch.zeros(n_batch, self.n_hid).float()  # Initialize mem_ch
        bias_x_chunks = self.bias_ih.chunk(3)
        bias_h_chunks = self.bias_hh.chunk(3)
        init_mem_x = torch.cat(
            (bias_x_chunks[0] + bias_h_chunks[0], bias_x_chunks[1] + bias_h_chunks[1], bias_x_chunks[2]))
        init_mem_h = bias_h_chunks[2]
        mem_x = torch.unsqueeze(init_mem_x, dim=0).repeat(n_batch, 1).float()  # Initialize mem_x
        mem_ch = torch.unsqueeze(init_mem_h, dim=0).repeat(n_batch, 1).float()  # Initialize mem_x
        output = torch.zeros(max_seq_len, n_batch, self.n_hid).float()
        one = torch.ones_like(hidden_prev).float()
        self.abs_delta_hid = torch.zeros_like(hidden_prev).float()

        # Whether use GPU
        if self.use_cuda:
            hidden = hidden.cuda()
            input_prev = input_prev.cuda()
            hidden_prev = hidden_prev.cuda()
            mem_x = mem_x.cuda()
            mem_ch = mem_ch.cuda()
            output = output.float().cuda()
            one = one.cuda()
            self.abs_delta_hid = self.abs_delta_hid.cuda()

        # Quantize input X
        # x = quantize_tensor(x, self.aqi, self.aqf, 1)

        # Save history of delta vectors to evaluate sparsity
        self.delta_inp = []
        self.delta_hid = []

        # Debug
        if self.debug:
            self.dict_debug['stat_cnt_idx_dx'] = torch.zeros_like(input_prev).detach().squeeze().cpu()
            self.dict_debug['stat_cnt_idx_dh'] = torch.zeros_like(hidden_prev).detach().squeeze().cpu()

        # Iterate through time steps
        for i, input_curr in enumerate(x.chunk(max_seq_len, dim=0)):
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # Get current input vectors
            input_curr = torch.squeeze(input_curr, dim=0)
            hidden_curr = hidden

            # Get raw delta vectors
            delta_inp_t = input_curr - input_prev
            delta_hid_t = hidden_curr - hidden_prev

            # Zero-out elements of delta input vector below the threshold
            delta_inp_abs = torch.abs(delta_inp_t)
            delta_inp_t = delta_inp_t.masked_fill_(delta_inp_abs < self.th_x, 0)
            # delta_inp_t = torch.masked_fill(torch.logical_and(delta_inp_t < self.th_x, delta_inp_t > -self.th_x), 0)

            # Zero-out elements of delta hidden vector below the threshold
            delta_hid_abs = torch.abs(delta_hid_t)
            delta_hid_t = delta_hid_t.masked_fill_(delta_hid_abs < self.th_h, 0)

            # Get L1 Penalty
            self.abs_delta_hid += torch.abs(delta_hid_t)

            # Convert Delta Vector to Sparse Vector
            # delta_inp = delta_inp.to_sparse()
            # delta_hid = delta_hid.to_sparse()

            # Run forward pass for one time step
            hidden, mem_x, mem_ch = self.timestep(delta_inp_t,
                                                  delta_hid_t,
                                                  hidden,
                                                  mem_x,
                                                  mem_ch,
                                                  one)

            # Quantize hidden
            hidden = quantize_tensor(hidden, self.aqi, self.aqf, 1)

            # Evaluate Temporal Sparsity
            if self.eval_sp:
                self.delta_inp.append(delta_inp_t.clone().detach().cpu())
                self.delta_hid.append(delta_hid_t.clone().detach().cpu())

            # Debug
            if self.debug:
                self.dict_debug['x'].append(torch.squeeze(input_curr))
                self.dict_debug['h'].append(torch.squeeze(hidden))
                self.dict_debug['dx'].append(torch.squeeze(delta_inp_t))
                self.dict_debug['dh'].append(torch.squeeze(delta_hid_t))
                self.dict_debug['stat_cnt_idx_dx'] += delta_inp_t.clone().detach().masked_fill_(
                    delta_inp_abs >= self.th_x, 1).cpu().squeeze()
                self.dict_debug['stat_cnt_idx_dh'] += delta_hid_t.clone().detach().masked_fill_(
                    delta_hid_abs >= self.th_h, 1).cpu().squeeze()

            # Update previous input vector memory on indices that had above-threshold change
            input_prev = torch.where(delta_inp_abs >= self.th_x, input_curr, input_prev)

            # Update previous hidden vector memory on indices that had above-threshold change
            hidden_prev = torch.where(delta_hid_abs >= self.th_h, hidden_curr, hidden_prev)

            # Append current DeltaGRU hidden output to the list
            output[i, :, :] = hidden

        if self.eval_sp:
            self.delta_inp = torch.stack(self.delta_inp).detach().cpu()
            self.delta_hid = torch.stack(self.delta_hid).detach().cpu()

        # self.abs_delta_hid = self.abs_delta_hid.cpu()

        return output

    def set_quantize_act(self, x):
        self.qa = torch.tensor(x).int()

    def set_eval_sp(self, x):
        self.eval_sp = torch.tensor(x).int()

    def forward(self, x, h_0=None):
        """
        :param input: 3D-Input tensor of feature of all time steps with size (seq_len, n_batch, n_feature)
        :param feat_len: 1D-Tensor of size (n_batch) having sequence length of each sample in the batch
        :param show_sp: Whether to return sparsity of delta vectors
        :return:
            - output_seq: 3D-Tensor of all time steps of the rnn outputs with size (seq_len, n_batch, n_hid)
            - nz_dx: Number of nonzero elements in delta input vectors of all time steps
            - nz_dh Number of nonzero elements in delta hidden vectors of all time steps
            - abs_delta_hid: L1 mean penalty
            - sp_W: Weight sparsity
        """

        # Clear debug dictionary
        if self.debug:
            for key, value in self.dict_debug.items():
                self.dict_debug[key] = []

        output = self.delta_forward(x, h_0)

        # Store data for debugging
        if self.debug:
            for key, value in self.dict_debug.items():
                if 'stat' not in key:
                    self.dict_debug[key] = torch.stack(value, dim=0)

        return output


class DeltaGRU(nn.Module):
    def __init__(self,
                 n_inp,
                 n_hid,
                 num_layers,
                 th_x=0.25,
                 th_h=0.25,
                 qa=0,
                 aqi=8,
                 aqf=8,
                 nqi=2,
                 nqf=4,
                 eval_sp=0,
                 debug=0,
                 use_cuda=None):
        super(DeltaGRU, self).__init__()

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        # Properties
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.num_layers = num_layers
        self.th_x = th_x
        self.th_h = th_h
        self.aqi = aqi
        self.aqf = aqf
        self.nqi = nqi
        self.nqf = nqf
        self.eval_sp = eval_sp
        self.qa = qa
        self.debug = debug
        self.use_cuda = use_cuda
        self.layer_list = nn.ModuleList()

        # Statistics
        self.abs_sum_delta_hid = torch.zeros(1)
        self.sp_dx = 0
        self.sp_dh = 0

        # Debug
        self.list_rnn_debug = []

        # Instantiate DeltaGRU layers
        for i in range(self.num_layers):
            if i == 0:
                layer = DeltaGRUCell(n_inp=self.n_inp,
                                     n_hid=self.n_hid,
                                     th_x=self.th_x,
                                     th_h=self.th_h,
                                     qa=qa,
                                     aqi=self.aqi,
                                     aqf=self.aqf,
                                     nqi=self.nqi,
                                     nqf=self.nqf,
                                     eval_sp=self.eval_sp,
                                     debug=self.debug,
                                     use_cuda=self.use_cuda)
            else:
                layer = DeltaGRUCell(n_inp=self.n_hid,
                                     n_hid=self.n_hid,
                                     th_x=self.th_x,
                                     th_h=self.th_h,
                                     qa=qa,
                                     aqi=self.aqi,
                                     aqf=self.aqf,
                                     nqi=self.nqi,
                                     nqf=self.nqf,
                                     eval_sp=self.eval_sp,
                                     debug=self.debug,
                                     use_cuda=self.use_cuda)
            self.layer_list.append(layer)

    def set_quantize_act(self, x):
        self.qa = x
        for i in range(self.num_layers):
            self.layer_list[i].set_quantize_act(self.qa)

    def set_eval_sparsity(self, x):
        self.eval_sp = x
        for i in range(self.num_layers):
            self.layer_list[i].set_eval_sparsity(self.eval_sp)

    def forward(self, x, h=None):
        if x.dim() != 3:
            raise ValueError("The input vector x must be 3-dimensional (len, batch, n_feat)")
        self.list_rnn_debug = []
        # Propagate through layers
        self.sp_dx = 0
        self.sp_dh = 0
        delta_inp = []
        delta_hid = []
        self.abs_sum_delta_hid = 0
        for i, rnn in enumerate(self.layer_list):
            if h is None:
                x = rnn(x)
            else:
                x = rnn(x, h[i])
            if self.debug:
                self.list_rnn_debug.append(rnn.dict_debug)
            self.abs_sum_delta_hid += rnn.abs_delta_hid
            delta_inp.append(rnn.delta_inp)
            delta_hid.append(rnn.delta_hid)
        self.abs_sum_delta_hid = torch.sum(self.abs_sum_delta_hid)

        if self.eval_sp:
            self.sp_dx = get_temporal_sparsity(delta_inp)
            self.sp_dh = get_temporal_sparsity(delta_hid)

        return x, x[-1, :, :]
