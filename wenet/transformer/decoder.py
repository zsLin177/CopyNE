# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Decoder definition."""
from typing import Dict, Tuple, List, Optional
from numpy import dtype

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import (subsequent_mask, make_pad_mask)
from model.contextual_encoder import ContextualAttention, CrossAttention, SimpleAttention, DotAttention
from wenet.transformer.lstm import LstmEncoder
from wenet.transformer.bert import BertEmbedding
import torch.nn as nn
from supar.structs.chain import LinearChainCRF


class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.attention_dim = attention_dim
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        
        if self.normalize_before:
            x = self.after_norm(x)

        # add attention module here, attention with vocabulary

        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        context: Optional[torch.Tensor] = None,
        context_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache

class ContextualTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        symbol_table: dict,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        contextual_att: bool = True,
        cocoder_out_dim: int = 512,
        conatt_type: str='contextual',
        add_copy_loss: bool = False,
        no_concat: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.attention_dim = attention_dim
        self.contextual_att = contextual_att
        self.conatt_type = conatt_type
        self.add_copy_loss = add_copy_loss
        self.concat = not no_concat

        self.symbol_table = symbol_table
        self.left_lst = [value for key, value in self.symbol_table.items() if key in ['(', '<', '[']]
        self.right_lst = [value for key, value in self.symbol_table.items() if key in [')', '>', ']']]

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        if not contextual_att:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            if conatt_type == 'contextual':
                self.conatt = ContextualAttention(decoder_dim=attention_dim, cocoder_dim=cocoder_out_dim)
                self.output_layer = torch.nn.Linear(attention_dim+cocoder_out_dim, vocab_size)
            elif conatt_type == 'crossatt':
                self.conatt = CrossAttention(attention_dim, 4)
                self.output_layer = torch.nn.Linear(attention_dim+attention_dim, vocab_size)
            elif conatt_type == 'simpleatt':
                self.conatt = SimpleAttention(decoder_dim=attention_dim, cocoder_dim=cocoder_out_dim)
                if self.concat:
                    self.output_layer = torch.nn.Linear(attention_dim+cocoder_out_dim, vocab_size)
                else:
                    self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
            else:
                raise ValueError(f'No this att type: {conatt_type}!')
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        context: torch.Tensor,
        need_att_mask: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            context: [N, c_dim]
            need_att_mask: [batch, maxlen_out]
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        
        if self.normalize_before:
            x = self.after_norm(x)

        # add attention module here, attention with vocabulary
        if self.contextual_att:
            context_repr, p, score = self.conatt(x, context, need_att_mask)
        if self.use_output_layer:
            # if self.conatt_type == 'contextual':
            #     x = self.output_layer(torch.cat((x, context_repr), -1))
            # elif self.conatt_type == 'crossatt':
                # x = self.output_layer(context_repr)
            if self.concat:
                x = self.output_layer(torch.cat((x, context_repr), -1))
            else:
                x = self.output_layer(x)
            
        olens = tgt_mask.sum(1)
        if not self.add_copy_loss:
            return x, torch.tensor(0.0), olens
        else:
            return x, torch.tensor(0.0), olens, p

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        context: Optional[torch.Tensor] = None,
        copy_forward = False,
        copy_threshold = 0.89,
        context_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
            context: (n_vocabulary, c_dim)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        # x: (batch*beam, maxlen_in, dim)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        # y: (batch*beam, dim)
        if self.contextual_att:
            # get the need_att_mask
            # need_att_mask = get_need_att_mask(self.left_lst, self.right_lst, tgt)
            need_att_mask = torch.ones_like(tgt[:, -1]).bool()
            context_repr, all_att, all_score = self.conatt(y.unsqueeze(1), context, need_att_mask.unsqueeze(-1))
            context_repr = context_repr.squeeze(1)
        if self.use_output_layer:
            if not copy_forward:
                if self.concat:
                    y = torch.log_softmax(self.output_layer(torch.cat((y, context_repr), -1)), dim=-1)
                else:
                    y = torch.log_softmax(self.output_layer(y), dim=-1)
                # else:
                #     # (batch*beam, n_char)
                #     att_score = self.output_layer(y)
                #     # (batch*beam, n_v), n_v covers null
                #     bias_score = all_score.squeeze(1)
                #     att_score = self.posterior_adaptation(att_score, bias_score, context_tensor)
                #     y = torch.log_softmax(att_score, dim=-1)
            else:
                # (batch*beam, n_v), n_v covers one null
                all_att = all_att.squeeze(1)
                n_v = all_att.shape[1]
                # do threshold!
                # (batch*beam)
                threshold_mask = all_att[:, 0:-1].max(-1)[0].le(copy_threshold)
                all_att[threshold_mask.nonzero().squeeze(-1), 0:-1] = 0.0
                all_att[threshold_mask.nonzero().squeeze(-1), -1] = 1.0
                # (batch*beam, n_char)
                if self.concat:
                    y = self.output_layer(torch.cat((y, context_repr), -1)).softmax(-1)
                else:
                    y = self.output_layer(y).softmax(-1)
                y = all_att[:, -1].unsqueeze(-1) * y
                # [batch*beam, n_char+n_v-1]
                y = torch.cat((y, all_att[:, 0:-1]), -1).log()
        return y, new_cache

    def posterior_adaptation(self, att_score, bias_score, context_tensor, prompt_v=5.0):
        """
        att_score: [batch_size, n_char]
        bias_score: [batch_size, n_v], n_v covers null
        context_tensor: [n_v-1, max_len]
        """
        null_id = bias_score.shape[1] - 1
        # [batch_size]
        max_context_id = bias_score.max(-1)[1]
        need_prompt_mask = max_context_id.ne(null_id)
        c_ids = torch.index_select(context_tensor, dim=0, index=max_context_id.masked_fill(~need_prompt_mask, 0))
        tgt_ids = (c_ids.ne(-1) & need_prompt_mask.unsqueeze(1)).nonzero()
        tgt_c_ids = c_ids[tgt_ids[:, 0], tgt_ids[:, 1]]
        att_score[tgt_ids[:, 0], tgt_c_ids] += prompt_v
        return att_score

class CopyTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        symbol_table: dict,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        contextual_att: bool = True,
        cocoder_out_dim: int = 512,
        dot_att_dim: int = 512,
        ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.attention_dim = attention_dim
        self.symbol_table = symbol_table

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")
        
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.conatt = DotAttention(att_dim=dot_att_dim, decoder_dim=attention_dim, cocoder_dim=cocoder_out_dim, embed_dim=attention_dim)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        context: torch.Tensor,
        need_att_mask: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ):
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        
        if self.normalize_before:
            x = self.after_norm(x)
        embed_score, context_score = self.conatt(x, self.embed[0].weight, context, need_att_mask)
        # [batch_size, L, n_char+n_context]
        return torch.cat((embed_score, context_score), -1), embed_score, context_score

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        context: Optional[torch.Tensor] = None,
        copy_forward = True,
        copy_threshold = 5.0,
        context_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        # x: (batch*beam, maxlen_in, dim)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        need_att_mask = torch.ones_like(tgt[:, -1]).bool()
        embed_score, context_score = self.conatt(y.unsqueeze(1), self.embed[0].weight, context, need_att_mask.unsqueeze(-1))
        # [batch*beam, n_char]
        embed_score = embed_score.squeeze(1)
        # [batch*beam, n_context], n_context donot cover null !!!
        context_score = context_score.squeeze(1)
        if copy_forward:
            # TODO: add filter here, if max(con_score) - max(embed_score) < copy_threshold then turn all con_score to -inf
            need_filt_mask = (context_score.max(-1)[0] - embed_score.max(-1)[0]) < copy_threshold
            context_score[need_filt_mask] = -float('inf')
            return torch.cat((embed_score, context_score), -1).log_softmax(-1), new_cache
        else:
            return embed_score.log_softmax(-1), new_cache

class BiTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):

        assert check_argument_types()
        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

        self.right_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)

class ParaformerDecoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        add_ne_feat: bool = False,
        ne_label_num :int = 7,
        add_bert_feat: bool = False,
        bert_path: str = 'bert-base-uncased',
        e2e_ner: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.attention_dim = attention_dim
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")
        
        if add_ne_feat:
            self.ne_lstm = LstmEncoder(ne_label_num, n_lstm_hidden=attention_dim//2, char_pad_idx=-1)

        if add_bert_feat:
            self.bert_embed = BertEmbedding(bert_path, 3, attention_dim, requires_grad=False)

        self.e2e_ner = e2e_ner
        if e2e_ner:
            self.ner_head = NERHead(attention_dim, ne_label_num)

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])
    
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
    ):
        """
        Args:
            token-level input x from predictor or sampler: (batch, T, feat)
            x_mask: (batch, T)
            memory: encoded memory, float32  (batch, S, feat)
            memory_mask: encoder memory mask, (batch, S)
        Returns:
            (tuple): tuple containing:
                y: decoded token score before softmax (batch, T,
                    vocab_size) if use_output_layer is True,
                olens: (batch, )
        """
        # form the shape of masks
        # [batch, T, T]
        x_mask = x_mask.unsqueeze(1) & x_mask.unsqueeze(2)
        # [batch, 1, S]
        memory_mask = memory_mask.unsqueeze(1)

        for layer in self.decoders:
            x, x_mask, memory, memory_mask = layer(x, x_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            # [batch, T, vocab_size]
            logits = self.output_layer(x)
        olens = x_mask.sum(1)
        
        ne_logits = None
        if self.e2e_ner:
            ne_logits = self.ner_head(x)
        return logits, olens, ne_logits
        
class NERHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_labels: int,
    ):
        super().__init__()
        self.output_layer = torch.nn.Linear(input_dim, n_labels)
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            (batch, T, n_labels)
        """
        return self.output_layer(x)
    
    def decode(self, score, mask):
        """
        Args:
            score: (batch, T, n_labels)
            mask: (batch, T)
        Returns:
            (batch, T), (batch, )
        """
        dist = LinearChainCRF(score, self.trans, lens=mask.sum(-1))
        return dist.argmax, (dist.max - dist.log_partition).exp()
    
    def loss(self, score, gold_labels, mask):
        """
        Args:
            score: (batch, T, n_labels)
            gold_labels: (batch, T)
            mask: (batch, T)
        Returns:
            (batch, )
        """
        batch_size, seq_len = mask.shape
        # loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        loss = -LinearChainCRF(score, self.trans, lens=mask.sum(-1)).log_prob(gold_labels).sum() / seq_len
        return loss
        

def get_need_att_mask(left_id_lst, right_id_lst, ids):
    """
    这里的假设是括号的生成都是合法的, 也就是说不会出现不闭合, 也不会出现括号不匹配的情况, 比如(],
    我们将“(, [, <”都看成是一种, 相同的右括号也一样.
    通过判断做括号和右括号数量是否相等, 貌似不可以通过判断前面的状态和当前的生成来完成, 无法batch化
    left_id_lst: [2, 4, 6]
    right_id_lst: [3, 5, 7]
    ids: (batch, max_len_in) long

    return: (batch, )
    """
    batch = ids.shape[0]
    left_num = ids.new_zeros((batch, ), dtype=torch.long, device=ids.device)
    for left_id in left_id_lst:
        left_num += ids.eq(left_id).sum(-1)
    right_num = ids.new_zeros((batch, ), dtype=torch.long, device=ids.device)
    for right_id in right_id_lst:
        right_num += ids.eq(right_id).sum(-1)
    return ~(left_num == right_num)
    