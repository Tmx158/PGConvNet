import torch
import torch.nn as nn
from models.Model_BRITS.rits import TemporalDecay,FeatureRegression




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.rnn_hid_size = configs.rnn_hid_size
        self.enc_in=configs.enc_in

        self.rnn_cell = nn.LSTMCell(input_size=self.enc_in * 2, hidden_size=self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size=self.enc_in, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.enc_in, output_size=self.enc_in, diag=True)


        self.hist_reg = nn.Linear(self.rnn_hid_size, self.enc_in)
        self.feat_reg = FeatureRegression(self.enc_in)
        self.weight_combine = nn.Linear(self.enc_in * 2, self.enc_in)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, N = x.shape
        h = torch.zeros((B, self.rnn_hid_size), device=x.device)
        c = torch.zeros((B, self.rnn_hid_size), device=x.device)
        imputations = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :] if mask is not None else torch.ones_like(x_t)
            d_t = torch.zeros_like(x_t)  # Assume deltas are zeros or computed elsewhere

            gamma_h = self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)
            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_c = m_t * x_t + (1 - m_t) * x_h
            z_h = self.feat_reg(x_c)

            alpha = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            c_c = m_t * x_t + (1 - m_t) * c_h

            inputs = torch.cat([c_c, m_t], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)
        return imputations
