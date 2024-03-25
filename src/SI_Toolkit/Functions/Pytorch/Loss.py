import numpy as np
import torch
import torch.nn as nn

class loss_msr_sequence_customizable(nn.Module):
    def __init__(self, wash_out_len, post_wash_out_len, discount_factor=0.9):
        super(loss_msr_sequence_customizable, self).__init__()
        discount_vector = np.ones(shape=post_wash_out_len)
        for i in range(post_wash_out_len - 1):
            discount_vector[i + 1] = discount_vector[i] * discount_factor
        discount_vector = torch.tensor(discount_vector, dtype=torch.float32)

        standard_mse_loss = nn.MSELoss(reduction='none')

        def loss_msr_sequence(y_true, y_predicted):
            losses = standard_mse_loss(y_predicted, y_true)
            losses = torch.sum(losses, dim=-1)/y_true.size()[-1]
            # losses has shape [batch_size, time steps] -> this is the loss for every time step
            losses = losses[:, wash_out_len:]  # This discards losses for timesteps ≤ wash_out_len

            # Get discounted some of losses for a time series
            # Axis (2,1) results in the natural operation of losses * discount_vector
            # loss = keras.layers.Dot(axes=(1, 0))([losses, discount_vector])
            loss = torch.mv(losses, discount_vector)

            loss = torch.mean(loss)

            return loss

        self.loss_msr_sequence = loss_msr_sequence

    def forward(self, y_true, y_predicted):
        return self.loss_msr_sequence(y_true, y_predicted)