import torch
import torch.nn as nn
import torch.nn.functional as F


from models.Model_SAITS.layers import  EncoderLayer, PositionalEncoding
"""
SAITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.n_groups = configs.saits_n_groups
        self.n_group_inner_layers = configs.saits_n_group_inner_layers
        self.d_time = configs.saits_d_time
        self.d_feature = configs.saits_d_feature
        self.d_model = configs.saits_d_model
        self.d_inner = configs.saits_d_inner
        self.n_head = configs.saits_n_head
        self.d_k = configs.saits_d_k
        self.d_v = configs.saits_d_v
        self.dropout = configs.saits_dropout

        # 从 configs 中获取布尔值和策略参数
        self.input_with_mask = configs.saits_input_with_mask
        self.actual_d_feature = self.d_feature * 2 if self.input_with_mask else self.d_feature
        self.param_sharing_strategy = configs.saits_param_sharing_strategy
        self.MIT = configs.saits_MIT
        self.device = configs.saits_device
        self.diagonal_attention_mask = configs.saits_diagonal_attention_mask

        # Encoder layers based on parameter sharing strategy
        if self.param_sharing_strategy == "between_group":
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        0,
                        diagonal_attention_mask=self.diagonal_attention_mask,
                        device=self.device
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        0,
                        diagonal_attention_mask=self.diagonal_attention_mask,
                        device=self.device
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
        else:
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        0,
                        diagonal_attention_mask=self.diagonal_attention_mask,
                        device=self.device
                    )
                    for _ in range(self.n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        0,
                        diagonal_attention_mask=self.diagonal_attention_mask,
                        device=self.device
                    )
                    for _ in range(self.n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=self.dropout)
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.d_time)

        # Layers for each block
        self.embedding_1 = nn.Linear(self.actual_d_feature, self.d_model)
        self.reduce_dim_z = nn.Linear(self.d_model, self.d_feature)
        self.embedding_2 = nn.Linear(self.actual_d_feature, self.d_model)
        self.reduce_dim_beta = nn.Linear(self.d_model, self.d_feature)
        self.reduce_dim_gamma = nn.Linear(self.d_feature, self.d_feature)
        self.weight_combine = nn.Linear(self.d_feature + self.d_time, self.d_feature)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Forward pass for the SAITS model adapted to match the benchmark interface.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, N), masked where data is missing.
            x_mark_enc, x_dec, x_mark_dec: Unused in SAITS, included for compatibility.
            mask (torch.Tensor): Mask tensor of shape (B, T, N) indicating missing data points.

        Returns:
            torch.Tensor: Imputed tensor of shape (B, T, N).
        """
        # Block 1: Prepare input and apply the first block of DMSA layers
        input_X_for_first = torch.cat([x, mask], dim=2) if self.input_with_mask else x
        enc_output = self.dropout(self.position_enc(self.embedding_1(input_X_for_first)))

        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = mask * x + (1 - mask) * X_tilde_1

        # Block 2: Second block of DMSA layers
        input_X_for_second = torch.cat([X_prime, mask], dim=2) if self.input_with_mask else X_prime
        enc_output = self.position_enc(self.embedding_2(input_X_for_second))

        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # Block 3: Attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)
        if len(attn_weights.shape) == 4:
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)


        combining_weights = torch.sigmoid(self.weight_combine(torch.cat([mask, attn_weights], dim=2)))
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1

        # Final imputed output
        X_c = mask * x + (1 - mask) * X_tilde_3
        return X_c  # Return the final imputed data directly
