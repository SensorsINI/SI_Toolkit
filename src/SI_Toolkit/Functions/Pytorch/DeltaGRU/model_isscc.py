from typing import Optional, Tuple

import torch.nn as nn
from utils import util
from nnmodels.layers.deltagru import DeltaGRU
from nnmodels.layers.deltarnn import DeltaRNN
from nnmodels.layers.qLSTM import qLSTM
from nnmodels.layers.qFC import qLinear
from nnmodels.layers.dLSTM import dLSTM


class Model(nn.Module):
    def __init__(self, args, config):
        super(Model, self).__init__()
        # Attributes
        # try:
        self.inp_size = config['model']['input_size']
        self.n_classes = config['model']['n_classes']
        self.hid_type = config['model']['hid_type']
        self.hid_size = config['model']['hid_size']
        self.hid_layers = config['model']['hid_layers']
        self.hid_dropout = config['model']['hid_dropout']
        self.fc_type = config['model']['fc_type']
        self.fc_extra_size = config['model']['fc_extra_size']
        self.fc_dropout = config['model']['fc_dropout']
        self.qa = config['model']['qa']
        self.qw = config['model']['qw']
        self.qg = config['model']['qg']
        self.aqi = config['model']['aqi']
        self.aqf = config['model']['aqf']
        self.wqi = config['model']['wqi']
        self.wqf = config['model']['wqf']
        self.nqi = config['model']['nqi']
        self.nqf = config['model']['nqf']
        self.gqi = config['model']['gqi']
        self.gqf = config['model']['gqf']
        self.qa_fc_extra = config['model']['qa_fc_extra']
        self.aqi_fc_extra = config['model']['aqi_fc_extra']
        self.aqf_fc_extra = config['model']['aqf_fc_extra']
        self.qa_fc_final = config['model']['qa_fc_final']
        self.aqi_fc_final = config['model']['aqi_fc_final']
        self.aqf_fc_final = config['model']['aqf_fc_final']
        self.th_x = util.quantize_array(config['model']['th_x'], self.aqi, self.aqf, self.qa)
        self.th_h = util.quantize_array(config['model']['th_h'], self.aqi, self.aqf, self.qa)
        self.benchmark = config['model']['benchmark']
        self.eval_sp = config['model']['eval_sp']
        self.use_cuda = args.use_cuda
        self.debug = config['model']['debug']
        self.hardsigmoid = config['model']['hardsigmoid']
        self.hardtanh = config['model']['hardtanh']
        self.adapt_hw = 0
        self.get_stats = config['model']['drnn_stats']
            # optional_args = ['num_array', 'num_array_pe', 'act_latency',
            #                  'act_interval', 'op_freq']
            # for i in optional_args:
            #     if i in args:
            #         setattr(self, i, args[i])
        if 'num_array' in args:
            self.num_array = args.num_array
        if 'num_array_pe' in args:
            self.num_array_pe = args.num_array_pe
        if 'act_latency' in args:
            self.act_latency = args.act_latency
        if 'act_interval' in args:
            self.act_interval = args.act_interval
        if 'op_freq' in args:
            self.op_freq = args.op_freq
        # except AttributeError:
        #     raise AttributeError("Missing compulsory arguments.")

        # Debug
        self.list_debug = []

        # Input Dropout
        self.input_dropout = nn.Dropout(p=args.inp_dropout)

        # Instantiate RNN layers
        if self.hid_type == 'LSTM':
            self.cla = nn.LSTM(input_size=self.inp_size,
                               hidden_size=self.hid_size,
                               num_layers=self.hid_layers,
                               bias=True,
                               bidirectional=False,
                               dropout=self.hid_dropout)
        elif self.hid_type == 'GRU':
            self.cla = nn.GRU(input_size=self.inp_size,
                              hidden_size=self.hid_size,
                              num_layers=self.hid_layers,
                              bias=True,
                              bidirectional=False,
                              dropout=self.hid_dropout)
        elif self.hid_type == 'DeltaGRU':
            self.cla = DeltaGRU(n_inp=self.inp_size,
                                n_hid=self.hid_size,
                                num_layers=self.hid_layers,
                                th_x=self.th_x,
                                th_h=self.th_h,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                nqi=self.nqi,
                                nqf=self.nqf,
                                qa=self.qa,
                                eval_sp=self.eval_sp,
                                debug=self.debug,
                                use_cuda=self.use_cuda, )
        elif self.hid_type == 'myLSTM':
            self.cla = DeltaRNN(mode='LSTM',
                                input_size=self.inp_size,
                                hidden_size=self.hid_size,
                                num_layers=self.hid_layers,
                                th_x=self.th_x,
                                th_h=self.th_h,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                nqi=self.nqi,
                                nqf=self.nqf
                                )
        elif self.hid_type == 'DeltaLSTM':
            self.cla = DeltaRNN(mode='DeltaLSTM',
                                input_size=self.inp_size,
                                hidden_size=self.hid_size,
                                num_layers=self.hid_layers,
                                th_x=self.th_x,
                                th_h=self.th_h,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                nqi=self.nqi,
                                nqf=self.nqf,
                                debug=self.debug,
                                num_array=self.num_array,
                                num_array_pe=self.num_array_pe,
                                act_latency=self.act_latency,
                                act_interval=self.act_interval,
                                op_freq=self.op_freq,
                                hardtanh=self.hardtanh,
                                hardsigmoid=self.hardsigmoid,
                                adapt_hw=self.adapt_hw,
                                get_stats=self.get_stats
                                )
        elif self.hid_type == 'qLSTM':
            self.cla = qLSTM(n_inp=self.inp_size,
                             n_hid=self.hid_size,
                             num_layers=self.hid_layers,
                             dropout=self.hid_dropout,
                             qa=self.qa,
                             qw=self.qw,
                             qg=self.qg,
                             aqi=self.aqi,
                             aqf=self.aqf,
                             wqi=self.wqi,
                             wqf=self.wqf,
                             gqi=self.gqi,
                             gqf=self.gqf,
                             eval_sp=self.eval_sp,
                             cuda=self.use_cuda,
                             debug=self.debug)
        elif self.hid_type == 'dLSTM':
            self.cla = dLSTM(n_inp=self.inp_size,
                             n_hid=self.hid_size,
                             num_layers=self.hid_layers,
                             # dropout=self.cla_dropout,
                             th_x=self.th_x,
                             th_h=self.th_h,
                             qa=self.qa,
                             qw=self.qw,
                             qg=self.qg,
                             aqi=self.aqi,
                             aqf=self.aqf,
                             wqi=self.wqi,
                             wqf=self.wqf,
                             gqi=self.gqi,
                             gqf=self.gqf,
                             eval_sp=self.eval_sp,
                             cuda=self.use_cuda,
                             debug=self.debug)
        elif self.hid_type == 'FC':
            list_fc = []
            inp_fc_layer = nn.Sequential(
                nn.Linear(in_features=self.inp_size, out_features=self.hid_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.hid_dropout)
            )
            list_fc.append(inp_fc_layer)
            hid_fc_layer = nn.Sequential(
                nn.Linear(in_features=self.hid_size, out_features=self.hid_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.hid_dropout)
            )
            for i in range(0, self.hid_layers-1):
                list_fc.append(hid_fc_layer)
            self.cla = nn.Sequential(*list_fc)
        if self.fc_type == 'FC':
            if self.fc_extra_size != 0:
                self.fc_extra = nn.Sequential(
                    nn.Linear(in_features=self.hid_size, out_features=self.fc_extra_size, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=self.fc_dropout)
                )
                self.fc_final = nn.Linear(in_features=self.fc_extra_size, out_features=self.n_classes, bias=True)
            else:
                self.fc_final = nn.Linear(in_features=self.hid_size, out_features=self.n_classes, bias=True)
        elif self.fc_type == 'qFC':
            if self.fc_extra_size != 0:
                self.fc_extra = nn.Sequential(
                    qLinear(in_features=self.hid_size,
                            out_features=self.fc_extra_size,
                            bias=True,
                            qa=self.qa_fc_extra,
                            qw=self.qw,
                            qg=self.qg,
                            aqi=self.aqi_fc_extra,
                            aqf=self.aqf_fc_extra,
                            wqi=self.wqi,
                            wqf=self.wqf,
                            gqi=self.gqi,
                            gqf=self.gqf),
                    nn.ReLU(),
                    nn.Dropout(p=self.fc_dropout)
                )
                self.fc_final = nn.Sequential(
                    qLinear(in_features=self.fc_extra_size,
                            out_features=self.n_classes,
                            bias=True,
                            qa=self.qa_fc_final,
                            qw=self.qw,
                            qg=self.qg,
                            aqi=self.aqi_fc_final,
                            aqf=self.aqf_fc_final,
                            wqi=self.wqi,
                            wqf=self.wqf,
                            gqi=self.gqi,
                            gqf=self.gqf),
                )
            else:
                self.fc_final = nn.Sequential(
                    qLinear(in_features=self.hid_size,
                            out_features=self.n_classes,
                            bias=True,
                            qa=self.qa_fc_final,
                            qw=self.qw,
                            qg=self.qg,
                            aqi=self.aqi_fc_final,
                            aqf=self.aqf_fc_final,
                            wqi=self.wqi,
                            wqf=self.wqf,
                            gqi=self.gqi,
                            gqf=self.gqf),
                )

    def set_qa_fc_final(self, x):
        """
        Set quantization of activation
        :param x: If x == 1, quantize activation; else, don't quantize.
        :return:
        """
        self.qa_fc_final = x

    def forward(self, input, state: Optional[Tuple] = None, feature_lengths=None):
        # Attributes
        self.list_rnn_debug = []
        dict_debug_fc_final = {}
        if self.training:
            quantize = 0
            debug = False
        else:
            quantize = self.qa
            debug = True

        # Flatten RNN Parameters if possible
        if self.hid_type in ['GRU', 'LSTM']:
            self.cla.flatten_parameters()

        # Input Dropout
        input = self.input_dropout(input)

        # RNN Forward
        if self.hid_type != 'FC':
            if feature_lengths is not None:
                out, state, reg = self.rnn_forward(input, state, quantize, feature_lengths)
            else:
                out, state, reg = self.rnn_forward(input, state, quantize)
        else:
            out = self.fc_forward(input)
            state = None
            reg = None

        # FC Forward
        out_final, out_fc, out_fc_acc = self.final_fc_forward(out, quantize)

        outputs = (out_final, out)


        if self.debug:
            dict_debug_fc_final['fc_final_inp'] = out
            dict_debug_fc_final['fc_final_out'] = out_fc
            dict_debug_fc_final['fc_final_qout'] = out_final
            dict_debug_fc_final['fc_final_out_acc'] = out_fc_acc
            self.list_rnn_debug = self.cla.list_rnn_debug
            self.list_rnn_debug.append(dict_debug_fc_final)

        if self.hid_type == "DeltaGRU":
            self.list_debug = self.cla.list_rnn_debug

        return outputs, state, reg

    def rnn_forward(self, x, s, quantize, feature_lengths=None):
        if self.hid_type == 'DeltaGRU':
            y, s = self.cla(x)
            r = None
        elif 'Delta' in self.hid_type or 'my' in self.hid_type:
            y, s, r = self.cla(x, s, quantize, feature_lengths)
        else:
            y, s = self.cla(x, s)
            r = None
        # Transpose RNN Output to (N, T, H)
        y = y.transpose(0, 1)
        return y, s, r

    def fc_forward(self, x):
        y = self.cla(x)
        return y

    def final_fc_forward(self, x, quantize):
        if self.benchmark:
            y = x
        else:
            if self.fc_extra_size:
                out_fc = self.fc_extra(x)
                out_fc = util.quantize_tensor(out_fc, self.aqi, self.aqf, quantize)
                out_fc = self.fc_final(out_fc)
            else:
                out_fc = self.fc_final(x)
            out_fc_acc = util.quantize_tensor(out_fc, 9, 15, self.qa_fc_final)
            qout_fc = util.quantize_tensor(out_fc, self.aqi_fc_final, self.aqf_fc_final, quantize)
            # qout_fc = out_fc
        return qout_fc, out_fc, out_fc_acc

    def balance_workload(self):
        self.cla.balance_workload()
        # Swap cols of the layer following RNN layers
        input_swap_list = self.cla.hidden_swap_list[-1]
        for name, param in self.named_parameters():
            if name == 'fc_extra.0.weight':
                print("::: Swap Parameter: ", name)
                weight = param.data
                # Swap cols
                weight = weight[:, input_swap_list]
                # Update parameter
                param.data = weight


        # Adapt state sizes to hardware supported size
        # inp_size = float(all_delta_x.shape[-1])
        # hid_size = float(all_delta_h.shape[-1])
        # seq_len = all_delta_x.shape[0]
        # adapted_inp_size = int(math.ceil(inp_size / self.num_array) * self.num_array)
        # adapted_hid_size = int(math.ceil(hid_size / self.num_array) * self.num_array)
        # inp_zero_pad_size = adapted_inp_size - inp_size
        # hid_zero_pad_size = adapted_hid_size - hid_size
        # all_delta_x = np.concatenate((all_delta_x, np.zeros((seq_len, batch_size, inp_zero_pad_size))))
        # all_delta_h = np.concatenate((all_delta_h, np.zeros((seq_len, batch_size, hid_zero_pad_size))))
        # all_delta_vector = np.concatenate((all_delta_x, all_delta_h), axis=-1)