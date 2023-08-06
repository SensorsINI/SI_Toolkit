import torch

from SI_Toolkit.Functions.Pytorch.Loss import loss_msr_sequence_customizable as PT_loss
from SI_Toolkit.Functions.TF.Loss import loss_msr_sequence_customizable as TF_loss

import numpy as np
import tensorflow as tf
import torch as pt


wash_out_len = 10
post_wash_out_len = 20
discount_factor = 0.9

batch_size = 128
sequence_length = wash_out_len+post_wash_out_len
output_features = 10

y_true = np.random.randn(batch_size, sequence_length, output_features)
y_predicted = np.random.randn(batch_size, sequence_length, output_features)

pt_loss_class = PT_loss(wash_out_len, post_wash_out_len, discount_factor)
tf_loss_function = TF_loss(wash_out_len, post_wash_out_len, discount_factor)

tf_loss = tf_loss_function(tf.convert_to_tensor(y_true, dtype=tf.float32), tf.convert_to_tensor(y_predicted, dtype=tf.float32)).numpy()
pt_loss = pt_loss_class(pt.tensor(y_true, dtype=torch.float32), pt.tensor(y_predicted, dtype=torch.float32)).cpu().detach().numpy()

input_max_difference = np.max(np.abs(y_true-y_predicted))
loss_max_difference = np.max(np.abs(tf_loss-pt_loss))

print('Max elementwise difference between inputs: {}'.format(input_max_difference))
print('Max elementwise difference between losses: {}'.format(loss_max_difference))

print('Test_done')

