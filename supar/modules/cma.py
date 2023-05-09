import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    '''
    implmentation of cma
    '''
    def __init__(self,
                d_model: int,
                num_heads: int = 6,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.n_out = d_model

        # attn
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.functional.relu

    def forward(self, t_query, s_key, s_value, s_padding_mask):
        r'''
        Args:
            t_query: text query tensor [batch_size, t_len, d_model]
            s_key: speech key tensor [batch_size, s_len, d_model]
            s_value: speech value tensor [batch_size, s_len, d_model]
            s_padding_mask: tell which position in speech is padded,
                the value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged
                [batch_size, s_len]
        return:
            [batch_size, t_len, d_model]
        '''

        t_query, s_key, s_value = t_query.transpose(0, 1), s_key.transpose(0, 1), s_value.transpose(0, 1)
        # [t_len, batch_size, d_model]
        src = self.attn(t_query, s_key, s_value, key_padding_mask=s_padding_mask)[0]
        src = t_query + self.dropout1(src)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.transpose(0, 1)