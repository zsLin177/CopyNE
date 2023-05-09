
import torch
import torch.nn as nn
# from torch.nn import (TransformerEncoder, TransformerEncoderLayer)
from transformers.modeling_utils import prune_linear_layer
from transformers.activations import gelu, gelu_new, silu
from transformers import BertConfig
from supar.modules.transformer_encoder import SelfAttentionEncoder

import math

BertLayerNorm = torch.nn.LayerNorm
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "silu": silu, "gelu_new": gelu_new}



class CrossModalityAttention(nn.Module):
    '''
    implmentation of cma
    '''
    def __init__(self,
                d_model: int,
                num_heads: int = 8,
                dim_feedforward: int = 2048,
                dropout: float = 0.1
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

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GateAttentionFusionLayer(nn.Module):
    '''
    copied from https://github.com/DianboWork/M3T-CNERTA
    '''
    def __init__(self, d_model, audio_hidden_dim):
        super().__init__()
        self.config = BertConfig()
        self.d_model = d_model
        self.audio_hidden_dim = audio_hidden_dim
        self.config.hidden_size = d_model
        self.config.num_attention_heads = 8
        if self.audio_hidden_dim != self.config.hidden_size:
            self.dim_match = nn.Sequential(
                nn.Linear(self.audio_hidden_dim, self.config.hidden_size),
                nn.Tanh()
            )
        self.hidden_dim = self.config.hidden_size
        self.crossattention = BertAttention(self.config)
        self.intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)
        self.n_out = self.config.hidden_size * 2
        # self.text_linear = nn.Linear(self.hidden_dim, self.hidden_dim)  # Inferred from code (dim isn't written on paper)
        # self.audio_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.gate_linear = nn.Linear(self.hidden_dim * 2, 1)

    def forward(
        self,
        textual_hidden_repr,
        textual_mask,
        audio_hidden_repr,
        audio_attention_mask
    ):
        if self.audio_hidden_dim != self.config.hidden_size:
            audio_hidden_repr = self.dim_match(audio_hidden_repr)
        if audio_attention_mask.dim() == 3:
            audio_extended_attention_mask = audio_attention_mask[:, None, :, :]
        elif audio_attention_mask.dim() == 2:
            audio_extended_attention_mask = audio_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    audio_hidden_repr.shape, audio_attention_mask.shape
                )
            )
        audio_extended_attention_mask = (1.0 - audio_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=textual_hidden_repr, attention_mask=textual_mask, encoder_hidden_states=audio_hidden_repr,  encoder_attention_mask=audio_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        audio_repr = self.output(intermediate_output, attention_output)
        return torch.cat((textual_hidden_repr, audio_repr), -1)

class CTCAlignFusion(nn.Module):
    '''
    fusion based on ctc output
    currently cat txt and audio, later can use cma
    '''
    def __init__(self,
                d_model: int,
                n_layers: int = 6,
                num_heads: int = 8,
                dim_feedforward: int = 2048,
                dropout: float = 0.1
                ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.n_out = d_model * 2
        self.transformer_encoder = SelfAttentionEncoder(num_encoder_layers=n_layers,
                                                        emb_size=d_model,
                                                        num_heads=num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        position_embed=True)
    
    def forward(self, txt_repr, sp_repr, chars, probs, char_mask):
        '''
        sp_repr: [batch_size, sp_len, d]
        txt_repr: [batch_size, tx_len, d]
        chars: [batch_size, tx_len]
        probs: [batch_size, n_vocab, sp_len]
        char_mask: [batch_size, tx_len], true is to be masked
        '''
        batch_size, _ = chars.shape
        # [batch_size, tx_len, sp_len]
        gathered_probs = probs[torch.range(0, batch_size-1, device=chars.device).long().unsqueeze(-1), chars]
        # [batch_size, tx_len, d]
        tokend_sp_repr = torch.matmul(gathered_probs, sp_repr)
        tokend_sp_repr = self.transformer_encoder(tokend_sp_repr, char_mask)
        # currently cat txt and audio, later can use cma
        return torch.cat((txt_repr, tokend_sp_repr), -1)

    def get_tokenized_sp_repr(self, sp_repr, chars, probs, char_mask):
        batch_size, _ = chars.shape
        # [batch_size, tx_len, sp_len]
        gathered_probs = probs[torch.range(0, batch_size-1, device=chars.device).long().unsqueeze(-1), chars]
        # [batch_size, tx_len, d]
        tokend_sp_repr = torch.matmul(gathered_probs, sp_repr)
        tokend_sp_repr = self.transformer_encoder(tokend_sp_repr, char_mask)
        return tokend_sp_repr


class CTCAlignCMAFusion(nn.Module):
    '''
    fusion based on ctc output
    '''
    def __init__(self,
                d_model: int,
                n_layers: int = 6,
                num_heads: int = 8,
                dim_feedforward: int = 2048,
                dropout: float = 0.1
                ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.n_out = d_model
        self.transformer_encoder = SelfAttentionEncoder(num_encoder_layers=n_layers,
                                                        emb_size=d_model,
                                                        num_heads=num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        position_embed=True)
        self.cma = CrossModalityAttention(d_model)
    
    def forward(self, txt_repr, sp_repr, chars, probs, char_mask):
        '''
        sp_repr: [batch_size, sp_len, d]
        txt_repr: [batch_size, tx_len, d]
        chars: [batch_size, tx_len]
        probs: [batch_size, n_vocab, sp_len]
        char_mask: [batch_size, tx_len], true is to be masked
        '''
        batch_size, _ = chars.shape
        # [batch_size, tx_len, sp_len]
        gathered_probs = probs[torch.range(0, batch_size-1, device=chars.device).long().unsqueeze(-1), chars]
        # [batch_size, tx_len, d]
        tokend_sp_repr = torch.matmul(gathered_probs, sp_repr)
        tokend_sp_repr = self.transformer_encoder(tokend_sp_repr, char_mask)
        h = self.cma(txt_repr, tokend_sp_repr, tokend_sp_repr, char_mask)

        return h


        


