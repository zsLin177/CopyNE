
import math
from numpy import dtype
import torch
import torch.nn as nn
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from supar.modules.lstm import VariationalLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMContextCoder(nn.Module):
    # def __init__(self,
    #             embed: nn.Embedding,
    #             eos_id,
    #             input_size=256, 
    #             hidden_size=512, 
    #             num_layers=1,
    #             dropout=0.2,
    #             pad_idx=-1,
    #             add_null_context=True,
    #             bidirectional=False):
    #     super().__init__()
    #     self.eos_id = eos_id
    #     self.pad_idx = pad_idx
    #     self.hidden_size = hidden_size
    #     self.bidirectional = bidirectional
    #     self.embed = embed
    #     self.lstm = VariationalLSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
    #     self.out_dim = hidden_size*2 if bidirectional else hidden_size
    #     if add_null_context:
    #         dim = hidden_size*2 if bidirectional else hidden_size
    #         self.null_context = nn.Parameter(torch.Tensor(1, dim))
    #         nn.init.zeros_(self.null_context)

    def __init__(self,
                vocab_size,
                eos_id,
                input_size=256, 
                hidden_size=512, 
                num_layers=1,
                dropout=0.2,
                pad_idx=-1,
                add_null_context=True,
                bidirectional=False):
        super().__init__()
        self.eos_id = eos_id
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = VariationalLSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.out_dim = hidden_size*2 if bidirectional else hidden_size
        self.add_null_context = add_null_context
        if not add_null_context:
            self.embed = nn.Embedding(vocab_size, input_size)
        else:
            self.embed = nn.Embedding(vocab_size+1, input_size)

    def forward(self, words, split=1):
        """
        words: all the words in the vocabulary. [N, maxlen]
        split: 1<=split<=N , if split > 1, split the whole vocabulary to several small vocabularies.
        """
        if self.add_null_context:
            max_len = words.shape[1]
            null_tensor = words.new_ones((1, max_len), device=words.device, dtype=torch.long) * self.pad_idx
            null_tensor[0, 0] = self.embed.weight.shape[0]-1
            words = torch.cat([words, null_tensor], 0)

        # [N, maxlen, input_size]
        # embed, _ = self.embed(words.masked_fill(words.eq(self.pad_idx), self.eos_id))
        embed = self.embed(words.masked_fill(words.eq(self.pad_idx), self.eos_id))
        x = pack_padded_sequence(embed, words.ne(self.pad_idx).sum(1).tolist(), True, False)
        x, _ = self.lstm(x)
        # bi:[N, maxlen, hidden_size*2] else:[N, maxlen, hidden_size]
        x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        if self.bidirectional:
            # [N, hidden_size]
            back = x[:, 0].split(self.hidden_size, -1)[1]
            forward = x[range(words.shape[0]), words.ne(self.pad_idx).sum(1)-1].split(self.hidden_size, -1)[0]
            # [N, hidden_size*2]
            real_context = torch.cat((forward, back), -1)
            # [N+1, hidden_size*2]
            return real_context
            # return torch.cat((real_context, self.null_context), 0)
        else:
            # [N, hidden_size]
            for_ward = x[range(words.shape[0]), words.ne(self.pad_idx).sum(1)-1]
            # [N+1, hidden_size]
            return for_ward
            # return torch.cat((for_ward, self.null_context), 0)

class TransformerContextCoder(nn.Module):
    def __init__(self,
                embed: nn.Embedding,
                eos_id,
                input_size=256,
                num_layers=2,
                dropout=0.2,
                attention_heads=4,
                pad_idx=-1,
                add_null_context=True,
                normal_before=False):
        super().__init__()
        self.eos_id = eos_id
        self.pad_idx = pad_idx
        self.embed = embed
        self.out_dim = input_size
        self.normal_before = normal_before
        # self.after_norm = torch.nn.LayerNorm(input_size, eps=1e-5)
        if add_null_context:
            self.null_context = nn.Parameter(torch.Tensor(1, input_size))
            nn.init.zeros_(self.null_context)
        
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                input_size, MultiHeadedAttention(attention_heads, input_size, dropout_rate=0),
                PositionwiseFeedForward(input_size, 2048, dropout_rate=0.1), dropout_rate=0.1, normalize_before=normal_before
            ) for _ in range(num_layers)
        ])

    def forward(self, words):
        # [N, maxlen, input_size]
        x, _ = self.embed(words.masked_fill(words.eq(self.pad_idx), self.eos_id))
        # [N, 1, maxlen]
        mask = words.ne(self.pad_idx).unsqueeze(1)
        for layer in self.encoders:
            x, mask, _, _ = layer(x, mask)
        # if self.normal_before:
        #     x = self.after_norm(x)
        # [N, max_len, dim]
        x = x.masked_fill(~mask.squeeze(1).unsqueeze(-1), 0.0)
        # [N, dim]
        x = x.sum(1) / words.ne(self.pad_idx).sum(-1).unsqueeze(-1)
        # [N+1, dim]
        x = torch.cat((x, self.null_context), dim=0)
        return x

class ContextualAttention(nn.Module):
    def __init__(self,
                att_dim=512,
                decoder_dim=256,
                cocoder_dim=256,
                conv1d_out_c=2,
                ):
        super().__init__()
        self.conv1d = nn.Conv1d(1, conv1d_out_c, 1)
        self.decoder_para = nn.Parameter(torch.Tensor(decoder_dim, att_dim))
        self.cocoder_para = nn.Parameter(torch.Tensor(cocoder_dim, att_dim))
        # self.preatt_para = nn.Parameter(torch.Tensor(conv1d_out_c, att_dim))
        self.bias = nn.Parameter(torch.Tensor(att_dim))
        self.weight = nn.Parameter(torch.Tensor(att_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.decoder_para)
        nn.init.orthogonal_(self.cocoder_para)
        # nn.init.orthogonal_(self.preatt_para)
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, d_h: torch.Tensor, c_h, need_att_mask, preatt=None):
        """
        d_h: [batch_size, max_len, d_dim]
        c_h: [N, c_dim]
        need_att_mask: [batch_size, max_len], which steps need to do contextual attention, True means need
        preatt: whether to use the attention in previous steps
        """
        batch_size, max_len = d_h.shape[0], d_h.shape[1]
        n_v, c_dim = c_h.shape[0], c_h.shape[1]
        # [batch_size, max_len, c_dim]
        context_repr = d_h.new_zeros((batch_size, max_len, c_dim), requires_grad=True, device=d_h.device)
        all_att = d_h.new_zeros((batch_size, max_len, n_v), requires_grad=True, device=d_h.device)
        if need_att_mask.sum().item() == 0:
            return context_repr, all_att
        # [k, d_dim]
        need_att_d = d_h[need_att_mask]
        # [k, N, att_dim]
        projed_d = torch.matmul(need_att_d, self.decoder_para).unsqueeze(1).expand(-1, n_v, -1)
        # [N, att_dim]
        projed_c = torch.matmul(c_h, self.cocoder_para)
        # [k, N]
        score = torch.matmul(torch.tanh(projed_d + projed_c + self.bias), self.weight)
        att = score.softmax(-1)
        all_att = all_att.masked_scatter(need_att_mask.unsqueeze(-1), att)
        # all_att[need_att_mask] = att
        # [batch_size, max_len, c_dim]
        context_repr = context_repr.masked_scatter(need_att_mask.unsqueeze(-1), torch.matmul(att, c_h))
        # context_repr[need_att_mask] = torch.matmul(att, c_h)
        return context_repr, all_att

class SimpleAttention(nn.Module):
    def __init__(self,
                att_dim=512,
                decoder_dim=256,
                cocoder_dim=256) -> None:
        super().__init__()
        self.decoder_w = nn.Linear(decoder_dim, att_dim)
        self.cocoder_w = nn.Linear(cocoder_dim, att_dim)
        self.att_dim = att_dim

    def forward(self, d_h, c_h, need_att_mask, preatt=None):
        batch_size, max_len = d_h.shape[0], d_h.shape[1]
        n_v, c_dim = c_h.shape[0], c_h.shape[1]
        all_att = d_h.new_zeros((batch_size, max_len, n_v), requires_grad=True, device=d_h.device)
        all_score = -d_h.new_ones((batch_size, max_len, n_v), requires_grad=True, device=d_h.device) * float('inf')
        # [k, d_dim]
        need_att_d = d_h[need_att_mask]
        # [k, att_dim]
        projed_d = self.decoder_w(need_att_d)
        # [N, att_dim]
        projed_c = self.cocoder_w(c_h)
        # [k, N]
        score = torch.matmul(projed_d, projed_c.transpose(0, 1)) / math.sqrt(self.att_dim)
        # [batch_size, max_len, N]
        all_score = all_score.masked_scatter(need_att_mask.unsqueeze(-1), score)
        p = torch.softmax(score, -1)
        # [batch_size, max_len, N]
        all_att = all_att.masked_scatter(need_att_mask.unsqueeze(-1), p)
        # [batch_size, max_len, c_dim]
        context_repr = torch.matmul(all_att, c_h)
        return context_repr, all_att, all_score


class DotAttention(nn.Module):
    """Used in decoding of CopyASR

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self,
                att_dim=512,
                decoder_dim=256,
                cocoder_dim=256,
                embed_dim=256,):
        super().__init__()
        self.decoder_w = nn.Linear(decoder_dim, att_dim)
        self.cocoder_w = nn.Linear(cocoder_dim, att_dim)
        self.embed_w = nn.Linear(embed_dim, att_dim)
        self.att_dim = att_dim

    def forward(self, d_h, embed, c_h, need_att_mask):
        """The output layer of CopyASR

        Args:
            d_h (torch.tensor): decoder output:
                [batch_size, max_d_len, d_dim]
            embed (torch.tensor): embed  weight of chars: 
                [v_char, e_dim]
            c_h (torch.tensor): concoder output:
                [v_context, c_dim], v_context donot cover null!!!
            need_att_mask (torch.tensor): tell which ones in max_d_len are needed:
                [batch_size, max_d_len]

        Returns:
            _type_: _description_
        """
        batch_size, max_len = d_h.shape[0], d_h.shape[1]
        v_context = c_h.shape[0]
        v_char = embed.shape[0]
        # [batch_size, max_d_len, v_context]
        all_con_score = -d_h.new_ones((batch_size, max_len, v_context), requires_grad=True, device=d_h.device) * float('inf')
        # [batch_size, max_d_len, v_char]
        all_e_score = -d_h.new_ones((batch_size, max_len, v_char), requires_grad=True, device=d_h.device) * float('inf')
        # [batch_size, max_d_len, v_char+v_context]
        # all_att = d_h.new_zeros((batch_size, max_len, v_char+v_context), requires_grad=True, device=d_h.device)
        # [k, d_dim]
        need_att_d = d_h[need_att_mask]
        # [k, att_dim]
        projed_d = self.decoder_w(need_att_d)
        # [v_context, att_dim]
        projed_c = self.cocoder_w(c_h)
        # [v_char, att_dim]
        projed_e = self.embed_w(embed)
        # [k, v_context]
        c_score = torch.matmul(projed_d, projed_c.transpose(0, 1)) / math.sqrt(self.att_dim)
        # [k, v_char]
        e_score = torch.matmul(projed_d, projed_e.transpose(0, 1)) / math.sqrt(self.att_dim)
        all_con_score = all_con_score.masked_scatter(need_att_mask.unsqueeze(-1), c_score)
        all_e_score = all_e_score.masked_scatter(need_att_mask.unsqueeze(-1), e_score)
        # # [k, v_char+v_context]
        # p = torch.cat((e_score, c_score), -1).softmax(-1)
        # # [batch_size, max_d_len, v_char+v_context]
        # all_att = all_att.masked_scatter(need_att_mask.unsqueeze(-1), p)
        # return all_att, all_e_score, all_con_score
        return all_e_score, all_con_score



class CrossAttention(nn.Module):
    def __init__(self,
                dim,
                heads,
                normal_before=False,
                dropout=0.1) -> None:
        super().__init__()
        self.normal_before = normal_before
        self.src_attn = MultiHeadedAttention(heads, dim, 0.0)
        self.feed_forward = PositionwiseFeedForward(dim, 2048, dropout)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)


    def forward(self, d_h: torch.Tensor, c_h, need_att_mask):
        """
        d_h: [batch_size, max_len, d_dim]
        c_h: [N, c_dim]
        need_att_mask: [batch_size, max_len], which steps need to do contextual attention, True means need
        """
        batch_size, max_len, d_dim = d_h.shape[0], d_h.shape[1], d_h.shape[2]
        n_v, c_dim = c_h.shape[0], c_h.shape[1]
        assert d_dim == c_dim
        memory = c_h.unsqueeze(0).expand(batch_size, -1, -1)
        memory_mask = torch.ones((batch_size, 1, n_v), device=memory.device).bool()
        residual = d_h
        if self.normal_before:
            d_h = self.norm1(d_h)
        d_h = residual + self.dropout(self.src_attn(d_h, memory, memory, memory_mask)[0])
        if not self.normal_before:
            d_h = self.norm1(d_h)
        
        residual = d_h
        if self.normal_before:
            d_h = self.norm2(d_h)
        d_h = residual + self.dropout(self.feed_forward(d_h))
        if not self.normal_before:
            d_h = self.norm2(d_h)

        # [batch_size, max_len, d_dim]
        return d_h, None








