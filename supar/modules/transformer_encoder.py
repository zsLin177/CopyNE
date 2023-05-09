import torch
import torch.nn as nn
import math
from torch.nn import (TransformerEncoder, TransformerEncoderLayer)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size)).cuda()
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = pos_embedding
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding [batch_size, seq_len, dim]
        # pdb.set_trace()
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(1), :])


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout = 0.1, max_len = 1000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape: [batch_size, seq_len, d_model] 
        """
        return self.dropout(x + self.weight[:x.size(1),])


class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 position_embed = True):
        super(SelfAttentionEncoder, self).__init__()
        self.num_layers = num_encoder_layers
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.position_embed = position_embed

        if self.num_layers !=0:
            encoder_layer = TransformerEncoderLayer(
                d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers)
        if self.position_embed:
            self.positional_encoding = LearnedPositionEncoding(emb_size,
                                                        dropout=dropout)

    def forward(self, src, pad_mask):
        '''
        src: [batch_size, seq_len, dim]
        pad_mask: [batch_size, seq_len] for pad, true means need to be pad
        '''
        if(self.position_embed):
            src_emb = self.positional_encoding(src)
        else:
            src_emb = src
        if self.num_layers == 0:
            return src_emb

        memory = self.transformer_encoder(src_emb.transpose(0, 1),
                                        mask=None,
                                        src_key_padding_mask=pad_mask)
        memory = memory.transpose(0, 1)
        return memory

    def __repr__(self):
        s = f"{self.emb_size}, num_layers={self.num_layers}, num_heads={self.num_heads}, dim_ffn={self.dim_feedforward}, position_embed={self.position_embed}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"