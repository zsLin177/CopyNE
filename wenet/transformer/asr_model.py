# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import (TransformerDecoder,
                                       BiTransformerDecoder,
                                       ContextualTransformerDecoder,
                                       CopyTransformerDecoder,
                                       ParaformerDecoder)
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.cif_predictor import CifPredictor, mae_loss
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from model.contextual_encoder import LSTMContextCoder, TransformerContextCoder
from supar.utils.fn import pad


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        blank_id=0,
        add_context_att=False,
        add_null_context=True,
        add_copy_loss=False,
        concoder_cofig: Optional[dict]=None,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.add_context_att = add_context_att
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.blank_id = blank_id
        self.add_copy_loss = add_copy_loss

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

        if self.add_context_att:
            cocoder_type = concoder_cofig.pop('type')
            if cocoder_type == 'lstm':
                # self.concoder = LSTMContextCoder(self.decoder.embed, self.eos, add_null_context=add_null_context, **concoder_cofig['lstm'])
                self.concoder = LSTMContextCoder(vocab_size, self.eos, add_null_context=add_null_context, **concoder_cofig['lstm'])
            elif cocoder_type == 'transformer':
                self.concoder = TransformerContextCoder(self.decoder.embed, self.eos, add_null_context=add_null_context, **concoder_cofig['transformer'])

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        context=None,
        need_att_mask=None,
        att_tgt=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            if not self.add_copy_loss:
                loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths, context, need_att_mask)
            else:
                # TODO
                loss_att, loss_copy, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths, context, need_att_mask, att_tgt)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        
        if not self.add_copy_loss:
            return loss, loss_att, loss_ctc
        else:
            loss = loss + 0.5 * loss_copy
            return loss, loss_att, loss_ctc, loss_copy

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        context=None,
        need_att_mask=None,
        att_tgt=None,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        if not self.add_context_att:
            decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        else:
            need_att_mask = ys_in_pad.ne(self.eos)
            need_att_mask[:, 0] = True
            if not self.add_copy_loss:
                decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     context, need_att_mask,
                                                     r_ys_in_pad, self.reverse_weight)
            else:
                decoder_out, r_decoder_out, _, att_p = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     context, need_att_mask,
                                                     r_ys_in_pad, self.reverse_weight)
                # Compute copy attention loss
                # att_p: [batch_size, max_len, n], att_tgt: [batch_size, max_len]
                loss_copy = -torch.gather(att_p[need_att_mask], -1, att_tgt[need_att_mask].unsqueeze(-1)).log().sum() / need_att_mask.shape[0]

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        if not self.add_copy_loss:
            return loss_att, acc_att
        else:
            return loss_att, loss_copy, acc_att


    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context: Optional[torch.Tensor] = None,
        context_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            context: [n_v, c_dim]

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache, context, context_tensor=context_tensor)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            n_cache = []
            for layer_cache in cache:
                n_cache.append(torch.index_select(layer_cache, dim=0, index=best_hyps_index))
            cache = n_cache
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def copy_recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context: Optional[torch.Tensor] = None,
        context_tensor: Optional[torch.Tensor] = None,
        copy_threshold: float = 0.9,
    ) -> torch.Tensor:
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # (B*N, 1)
        real_hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        real_hyps_len = torch.ones((running_size, ), dtype=torch.long,
                          device=device) # (B*N)
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            hyps = real_hyps[:, 0:i] # (B*N, i)
            # logp: (B*N, n_char+n_v-1)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache, context, copy_forward=True, copy_threshold=copy_threshold)
            # (B*N,)
            need_expand_mask = (real_hyps_len == i)
            # # (k, n_char+n_v-1)
            # logp = logp[need_expand_mask]
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            # import pdb
            # pdb.set_trace()
            if (~need_expand_mask).sum() > 0:
                top_k_logp[~need_expand_mask] = 0
                top_k_index[~need_expand_mask] = real_hyps[~need_expand_mask][:, i].unsqueeze(1).expand(-1, beam_size)

            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            # k_scores = scores[need_expand_mask] + top_k_logp # (k, N)
            # scores[need_expand_mask] = k_scores # (B*N, N)
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size) # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)
            
            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            end_flag = best_k_pred.view(-1, 1).eq(self.eos)

            n_cache = []
            for layer_cache in cache:
                n_cache.append(torch.index_select(layer_cache, dim=0, index=best_hyps_index))
            cache = n_cache

            real_hyps, real_hyps_len = self.batch_build_hyps(best_k_pred, best_hyps_index, real_hyps, real_hyps_len, i, context_tensor, self.vocab_size)
        
        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        best_hyps.masked_fill_(best_hyps.eq(-1), self.eos)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def batch_build_hyps(self, best_k_pred, best_hyps_index, real_hyps, real_len, step, context_tensor, n_c):
        """
        input: 
            best_k_pred: (B*N)
            best_hyps_index: (B*N)
            real_hyps: (B*N, max_len)
            real_len: (B*N)
            step: int
            context_tensor: (n_v-1, max_v_len)
            n_c: int
        """            
        # (n_v-1)
        context_len = context_tensor.ne(-1).sum(-1)
        # (B*N, max_len)
        last_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        # (B*N)
        last_hyps_real_len = torch.index_select(real_len, dim=0, index=best_hyps_index)
        # (B*N)
        need_change_mask = last_hyps_real_len == step
        if need_change_mask.sum() == 0:
            return last_hyps, last_hyps_real_len
        # (k, )
        pred_idx = best_k_pred[need_change_mask]
        pred_con_mask = pred_idx >= n_c
        # (m, )
        pred_con_idx = pred_idx[pred_con_mask] - n_c
        max_v_len = context_tensor.shape[1]
        # (k, max_v_len)
        pred_matrix = -torch.ones((pred_idx.shape[0], max_v_len), dtype=torch.long, device=context_len.device)
        pred_matrix[pred_con_mask] = torch.index_select(context_tensor, dim=0, index=pred_con_idx)
        pred_matrix[(~pred_con_mask).nonzero().squeeze(-1), 0] = pred_idx[~pred_con_mask]
        pred_matrix_len = pred_matrix.ne(-1).sum(-1).max(-1)[0].item()
        pred_matrix = pred_matrix[:, 0:pred_matrix_len]

        if (~need_change_mask).sum() == 0:
            n_real_hyps = -torch.ones((last_hyps.shape[0], step+pred_matrix.shape[1]), dtype=torch.long, device=last_hyps.device)
            n_real_hyps[:, 0:step] = last_hyps[:, 0:step]
            n_real_hyps[need_change_mask.nonzero().squeeze(-1), step:] = pred_matrix[0:need_change_mask.sum().item(), :]
        else:
            # (B*N-k, len)
            not_change_suffix = last_hyps[~need_change_mask][:, step:]
            max_notc_len = not_change_suffix.ne(-1).sum(-1).max(-1)[0].item()
            not_change_suffix = not_change_suffix[:, 0:max_notc_len]
            # (2, max(k, B*N-k), max(max_notc_len, pred_matrix_len))
            paded_matrix = pad((not_change_suffix, pred_matrix), padding_value=-1)
            # (B*N, step+max(max_notc_len, pred_matrix_len))
            n_real_hyps = -torch.ones((last_hyps.shape[0], step+paded_matrix.shape[2]), dtype=torch.long, device=last_hyps.device)
            n_real_hyps[:, 0:step] = last_hyps[:, 0:step]
            n_real_hyps[need_change_mask.nonzero().squeeze(-1), step:] = paded_matrix[1][0:need_change_mask.sum().item(), :]
            n_real_hyps[(~need_change_mask).nonzero().squeeze(-1), step:] = paded_matrix[0][0:(~need_change_mask).sum().item(), :]

        return n_real_hyps, n_real_hyps.ne(-1).sum(-1)
            

    def build_hyps(self, best_k_pred, best_hyps_index, real_hyps, real_len, step, context_tensor, n_c):
        """
        input: 
            best_k_pred: (B*N)
            best_hyps_index: (B*N)
            real_hyps: (B*N, max_len)
            real_len: (B*N)
            step: int
            context_tensor: (n_v-1, max_v_len)
            n_c: int
        """            
        # (n_v-1)
        context_len = context_tensor.ne(-1).sum(-1)
        # (B*N, max_len)
        last_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        # (B*N)
        last_hyps_real_len = torch.index_select(real_len, dim=0, index=best_hyps_index)
        # (B*N)
        need_change_mask = last_hyps_real_len == step
        n_real_hyps = []
        for i in range(last_hyps.shape[0]):
            if not need_change_mask[i]:
                n_real_hyps.append(last_hyps[i])
            else:
                pred_idx = best_k_pred[i].item()
                if pred_idx < n_c:
                    n_real_hyps.append(torch.cat((last_hyps[i][0: step], torch.tensor([pred_idx], dtype=torch.long, device=last_hyps.device)), -1))
                else:
                    n_real_hyps.append(torch.cat((last_hyps[i][0: step], context_tensor[pred_idx-n_c][0: context_len[pred_idx-n_c]]), -1))
        n_real_hyps = pad(n_real_hyps, padding_value=-1)
        n_real_len = n_real_hyps.ne(-1).sum(-1)
        return n_real_hyps, n_real_len

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, blank_id=self.blank_id) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        return hyps[0]

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out

class CopyASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model with copy mechanism"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        blank_id=0,
        concoder_cofig: Optional[dict]=None,
        ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        super().__init__()
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.blank_id = blank_id

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        cocoder_type = concoder_cofig.pop('type')
        if cocoder_type == 'lstm':
            self.concoder = LSTMContextCoder(vocab_size, self.eos, add_null_context=False, **concoder_cofig['lstm'])
        elif cocoder_type == 'transformer':
            self.concoder = TransformerContextCoder(self.decoder.embed, self.eos, add_null_context=False, **concoder_cofig['transformer'])
        else:
            raise NotImplementedError
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.InfoNCE = nn.CrossEntropyLoss()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        context: torch.Tensor,
        info_tgt: torch.Tensor,
        info_mask: torch.Tensor,
    ):
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, loss_infonce = self._calc_att_loss(encoder_out,  encoder_mask,
                                                    text, text_lengths, context, info_tgt, info_mask)
        else:
            loss_att = None
        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None
        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        
        loss = loss + loss_infonce
        return loss, loss_att, loss_ctc, loss_infonce

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        context: torch.Tensor,
        info_tgt: torch.Tensor,
        info_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        
        need_att_mask = ys_in_pad.ne(self.eos)
        need_att_mask[:, 0] = True
        char_con_score, char_score, context_score = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_in_lens, context, need_att_mask)
        # Compute InfoNCE loss
        # char_con_score: [batch_size, max_len, n_char+n_con], info_tgt: [batch_size, max_len], info_mask: [batch_size, max_len]
        loss_infonce = self.InfoNCE(char_con_score[info_mask], info_tgt[info_mask])

        # 2. Compute attention loss
        loss_att = self.criterion_att(char_score, ys_out_pad)
        return loss_att, loss_infonce

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context: Optional[torch.Tensor] = None,
        context_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
        beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            context: [n_v, c_dim]

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)
        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache, context, copy_forward=False)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            n_cache = []
            for layer_cache in cache:
                n_cache.append(torch.index_select(layer_cache, dim=0, index=best_hyps_index))
            cache = n_cache
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def copy_recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context: Optional[torch.Tensor] = None,
        context_tensor: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)
        
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # (B*N, 1)
        real_hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        real_hyps_len = torch.ones((running_size, ), dtype=torch.long,
                          device=device) # (B*N)
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            hyps = real_hyps[:, 0:i] # (B*N, i)
            # logp: (B*N, n_char+n_v) n_v donot cover null
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache, context, copy_forward=True)
            # (B*N,)
            need_expand_mask = (real_hyps_len == i)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            if (~need_expand_mask).sum() > 0:
                top_k_logp[~need_expand_mask] = 0
                top_k_index[~need_expand_mask] = real_hyps[~need_expand_mask][:, i].unsqueeze(1).expand(-1, beam_size)
            
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            # k_scores = scores[need_expand_mask] + top_k_logp # (k, N)
            # scores[need_expand_mask] = k_scores # (B*N, N)
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size) # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)
            
            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            end_flag = best_k_pred.view(-1, 1).eq(self.eos)

            n_cache = []
            for layer_cache in cache:
                n_cache.append(torch.index_select(layer_cache, dim=0, index=best_hyps_index))
            cache = n_cache

            real_hyps, real_hyps_len = self.batch_build_hyps(best_k_pred, best_hyps_index, real_hyps, real_hyps_len, i, context_tensor, self.vocab_size)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        best_hyps.masked_fill_(best_hyps.eq(-1), self.eos)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores
    
    def batch_build_hyps(self, best_k_pred, best_hyps_index, real_hyps, real_len, step, context_tensor, n_c):
        """
        input: 
            best_k_pred: (B*N)
            best_hyps_index: (B*N)
            real_hyps: (B*N, max_len)
            real_len: (B*N)
            step: int
            context_tensor: (n_v-1, max_v_len)
            n_c: int
        """            
        # (n_v-1)
        # (n_v-1)
        context_len = context_tensor.ne(-1).sum(-1)
        # (B*N, max_len)
        last_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        # (B*N)
        last_hyps_real_len = torch.index_select(real_len, dim=0, index=best_hyps_index)
        # (B*N)
        need_change_mask = last_hyps_real_len == step
        if need_change_mask.sum() == 0:
            return last_hyps, last_hyps_real_len
        # (k, )
        pred_idx = best_k_pred[need_change_mask]
        pred_con_mask = pred_idx >= n_c
        # (m, )
        pred_con_idx = pred_idx[pred_con_mask] - n_c
        max_v_len = context_tensor.shape[1]
        # (k, max_v_len)
        pred_matrix = -torch.ones((pred_idx.shape[0], max_v_len), dtype=torch.long, device=context_len.device)
        pred_matrix[pred_con_mask] = torch.index_select(context_tensor, dim=0, index=pred_con_idx)
        pred_matrix[(~pred_con_mask).nonzero().squeeze(-1), 0] = pred_idx[~pred_con_mask]
        pred_matrix_len = pred_matrix.ne(-1).sum(-1).max(-1)[0].item()
        pred_matrix = pred_matrix[:, 0:pred_matrix_len]

        if (~need_change_mask).sum() == 0:
            n_real_hyps = -torch.ones((last_hyps.shape[0], step+pred_matrix.shape[1]), dtype=torch.long, device=last_hyps.device)
            n_real_hyps[:, 0:step] = last_hyps[:, 0:step]
            n_real_hyps[need_change_mask.nonzero().squeeze(-1), step:] = pred_matrix[0:need_change_mask.sum().item(), :]
        else:
            # (B*N-k, len)
            not_change_suffix = last_hyps[~need_change_mask][:, step:]
            max_notc_len = not_change_suffix.ne(-1).sum(-1).max(-1)[0].item()
            not_change_suffix = not_change_suffix[:, 0:max_notc_len]
            # (2, max(k, B*N-k), max(max_notc_len, pred_matrix_len))
            paded_matrix = pad((not_change_suffix, pred_matrix), padding_value=-1)
            # (B*N, step+max(max_notc_len, pred_matrix_len))
            n_real_hyps = -torch.ones((last_hyps.shape[0], step+paded_matrix.shape[2]), dtype=torch.long, device=last_hyps.device)
            n_real_hyps[:, 0:step] = last_hyps[:, 0:step]
            n_real_hyps[need_change_mask.nonzero().squeeze(-1), step:] = paded_matrix[1][0:need_change_mask.sum().item(), :]
            n_real_hyps[(~need_change_mask).nonzero().squeeze(-1), step:] = paded_matrix[0][0:(~need_change_mask).sum().item(), :]

        return n_real_hyps, n_real_hyps.ne(-1).sum(-1)

    def build_hyps(self, best_k_pred, best_hyps_index, real_hyps, real_len, step, context_tensor, n_c):
        """
        input: 
            best_k_pred: (B*N)
            best_hyps_index: (B*N)
            real_hyps: (B*N, max_len)
            real_len: (B*N)
            step: int
            context_tensor: (n_v-1, max_v_len)
            n_c: int
        """
        # (n_v-1)
        context_len = context_tensor.ne(-1).sum(-1)
        # (B*N, max_len)
        last_hyps = torch.index_select(real_hyps, dim=0, index=best_hyps_index)
        # (B*N)
        last_hyps_real_len = torch.index_select(real_len, dim=0, index=best_hyps_index)
        # (B*N)
        need_change_mask = last_hyps_real_len == step
        n_real_hyps = []
        for i in range(last_hyps.shape[0]):
            if not need_change_mask[i]:
                n_real_hyps.append(last_hyps[i])
            else:
                pred_idx = best_k_pred[i].item()
                if pred_idx < n_c:
                    n_real_hyps.append(torch.cat((last_hyps[i][0: step], torch.tensor([pred_idx], dtype=torch.long, device=last_hyps.device)), -1))
                else:
                    n_real_hyps.append(torch.cat((last_hyps[i][0: step], context_tensor[pred_idx-n_c][0: context_len[pred_idx-n_c]]), -1))
        n_real_hyps = pad(n_real_hyps, padding_value=-1)
        n_real_len = n_real_hyps.ne(-1).sum(-1)
        return n_real_hyps, n_real_len

class ParaformerASRModel(torch.nn.Module):
    """
    Implementation of ParaformerASRModel
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, 
                 vocab_size: int,
                 encoder: TransformerEncoder,
                 decoder: ParaformerDecoder,
                 predictor: CifPredictor,
                 ctc: CTC,
                 ctc_weight: float = 0.3,
                 blank_id=0,
                 predictor_weight: float = 1.0,
                 predictor_bias: int = 0,
                 sampling_ratio: float = 0.4,
                 ignore_id: int = IGNORE_ID,
                 lsm_weight: float = 0.0,
                 length_normalized_loss: bool = False,
                 use_1st_decoder_loss: bool = False,
                ):
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.ctc = ctc
        self.predictor_weight = predictor_weight
        self.ctc_weight = ctc_weight
        self.blank_id = blank_id
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.length_normalized_loss = length_normalized_loss
        self.use_1st_decoder_loss = use_1st_decoder_loss

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        # encoder_out: (B, T, encoder_dim), encoder_mask: (B, 1, T)
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        # encoder_out_lens: (B, )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2.a Decoder
        loss_att, loss_pre, pre_loss_att = self._calc_att_loss(encoder_out, encoder_mask, text, text_lengths)

        # 2.b CTC
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc = None

        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

        if self.use_1st_decoder_loss and pre_loss_att is not None:
            loss = loss + (1 - self.ctc_weight) * pre_loss_att
        return loss, loss_att, loss_pre, loss_ctc, pre_loss_att
    
    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        """
        Args:
            encoder_out: (B, T, encoder_dim)
            encoder_mask: (B, 1, T)
            ys_pad: (B, L)
            ys_pad_lens: (B, )
        """
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        
        # pred: (B, L, dim)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, ys_pad, encoder_mask, ignore_id=self.ignore_id)

        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:
            if not self.use_1st_decoder_loss:
                sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_mask, ys_pad, ys_pad_lens, pre_acoustic_embeds)
            else:
                sematic_embeds, decoder_out_1st, pre_loss_att = self.sampler_with_grad(encoder_out, encoder_mask, ys_pad, ys_pad_lens, pre_acoustic_embeds)
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        tgt_mask = (~make_pad_mask(ys_pad_lens, max_len=ys_pad_lens.max())).to(ys_pad.device)
        decoder_outs = self.decoder(sematic_embeds, tgt_mask, encoder_out, encoder_mask.squeeze(1))
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        return loss_att, loss_pre, pre_loss_att
    
    def sampler_with_grad(self, encoder_out, encoder_mask, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        tgt_mask = (~make_pad_mask(ys_pad_lens, max_len=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]

        ys_pad_embed, _ = self.decoder.embed(ys_pad_masked)
        decoder_outs = self.decoder(
                pre_acoustic_embeds, tgt_mask.squeeze(-1), encoder_out, encoder_mask.squeeze(1))
        pre_loss_att = self.criterion_att(decoder_outs[0], ys_pad)

        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        pred_tokens = decoder_out.argmax(-1)
        nonpad_positions = ys_pad.ne(self.ignore_id)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
        input_mask = torch.ones_like(nonpad_positions)
        bsz, seq_len = ys_pad.size()
        for li in range(bsz):
            target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
            if target_num > 0:
                input_mask[li].scatter_(dim=0,
                                        index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
                                        value=0)
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(~nonpad_positions, False)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask, pre_loss_att


    def sampler(self, encoder_out, encoder_mask, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        tgt_mask = (~make_pad_mask(ys_pad_lens, max_len=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]

        ys_pad_embed, _ = self.decoder.embed(ys_pad_masked)

        with torch.no_grad():
            decoder_outs = self.decoder(
                pre_acoustic_embeds, tgt_mask.squeeze(-1), encoder_out, encoder_mask.squeeze(1))
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0,
                                            index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
                                            value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask
    
    def calc_predictor(self, encoder_out, encoder_out_lens):
        
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, max_len=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index
    
    def cal_decoder_with_predictor(self, encoder_out, encoder_mask, sematic_embeds, semantic_mask):
        
        decoder_outs = self.decoder(
            sematic_embeds, semantic_mask, encoder_out, encoder_mask
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, semantic_mask.sum(-1)

    def recognize(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Currently, only support greedy decoding
        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        device = speech.device
        batch_size = speech.shape[0]
        
        # encoder
        # encoder_out: (B, T, encoder_dim), encoder_mask: (B, 1, T)
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)

        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()

        if torch.max(pre_token_length) < 1:
            return torch.ones((batch_size, 1), device=device, dtype=torch.long) * self.eos
        
        # (B, L)
        tgt_mask = (~make_pad_mask(pre_token_length, max_len=pre_token_length.max())).to(device)
        decoder_outs = self.cal_decoder_with_predictor(encoder_out, encoder_mask.squeeze(1), pre_acoustic_embeds, tgt_mask)
        # (B, L, V)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        scores, hypos = decoder_out.max(-1)
        hypos = hypos.masked_fill(~tgt_mask, self.eos)
        return hypos, scores

def init_asr_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        # global_cmvn = GlobalCMVN(
        #     torch.from_numpy(mean).float(),
        #     torch.from_numpy(istd).float())
        global_cmvn = GlobalCMVN(
            torch.tensor(mean).float(),
            torch.tensor(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    
    add_copy_loss = configs.get('add_copy_loss', False)
    if 'add_copy_loss' in configs:
        configs.pop('add_copy_loss')

    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'conattransformer':
        symbol_table = configs.pop('symbol_table')
        att_type = configs.pop('att_type')
        decoder = ContextualTransformerDecoder(vocab_size, encoder.output_size(), symbol_table, conatt_type=att_type, add_copy_loss=add_copy_loss, no_concat=configs['no_concat'], **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())
    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        add_copy_loss=add_copy_loss,
        **configs['model_conf'],
    )
    return model

def init_copyasr_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.tensor(mean).float(),
            torch.tensor(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    dot_att_dim = configs['dot_dim']

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    
    if decoder_type == 'copytransformer':
        symbol_table = configs.pop('symbol_table')
        decoder = CopyTransformerDecoder(vocab_size, encoder.output_size(), symbol_table, dot_att_dim=dot_att_dim, **configs['decoder_conf'])
    else:
        raise NotImplementedError
    ctc = CTC(vocab_size, encoder.output_size())
    model = CopyASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )
    return model

def init_paraformer_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.tensor(mean).float(),
            torch.tensor(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
        
    decoder = ParaformerDecoder(vocab_size, encoder.output_size(), **configs['decoder_conf'])

    predictor = CifPredictor(configs["predictor_conf"]["idim"], configs["predictor_conf"]["l_order"], configs["predictor_conf"]["r_order"], threshold=configs["predictor_conf"]["threshold"], tail_threshold=configs["predictor_conf"]["tail_threshold"])

    ctc = CTC(vocab_size, encoder.output_size(), blank_id=configs['model_conf']['blank_id'])

    model = ParaformerASRModel(
        vocab_size,
        encoder,
        decoder,
        predictor,
        ctc,
        **configs['model_conf'],
    )
    return model

def init_encoder(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.tensor(mean).float(),
            torch.tensor(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    encoder_type = configs.get('encoder', 'conformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    return encoder

class CTCModel(torch.nn.Module):
    '''
    Only use ctc
    '''
    def __init__(
        self,
        vocab_size: int,
        proj_size: int,
        encoder: TransformerEncoder,
        ctc: CTC,
        sos=None,
        eos=None,
        blank_id=0,
    ):
        super().__init__()
        if sos is None and eos is None:
            self.sos = vocab_size - 1
            self.eos = vocab_size - 1
        else:
            self.sos = sos
            self.eos = eos
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.encoder = encoder

        self.proj_size = proj_size
        if proj_size != encoder.output_size():
            # self.proj_mlp = MLP(n_in=encoder.output_size(),
            #                     n_out=proj_size,
            #                     activation=False)
            self.proj_mlp = nn.Linear(encoder.output_size(), proj_size, bias=False)
        else:
            self.proj_mlp = nn.Identity()

        self.ctc = ctc

    def forward(self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.proj_mlp(encoder_out)

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        loss = loss_ctc
        return loss, None, loss_ctc
    
    def loss(self, encoder_out, encoder_out_lens, text, text_lengths):
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        return loss_ctc

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        encoder_out = self.proj_mlp(encoder_out)
        return encoder_out, encoder_mask

    def n_ctc_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ):
        assert decoding_chunk_size != 0
        batch_size = encoder_out.shape[0]
        # Let's assume B = batch_size
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.ctc.ctc_lo.out_features-1)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, blank_id=self.blank_id) for hyp in hyps]
        return hyps, scores

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, blank_id=self.blank_id) for hyp in hyps]
        return hyps, scores

    def batchify(self, batch, configs):
        guids = [ele.guid for ele in batch]

        ## Audio Feature
        if batch[0].audio_feature is not None:
            audio_features = [ele.audio_feature for ele in batch]
            audio_feature_lengths = torch.tensor([ele.audio_feature_length for ele in batch], dtype=torch.int)
            audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)
            max_len = audio_feature_lengths.max().item()
            audio_feature_masks = ~(make_pad_mask(audio_feature_lengths, max_len))
            if configs['use_gpu']:
                audio_features = audio_features.cuda()
                audio_feature_masks = audio_feature_masks.cuda()
                audio_feature_lengths = audio_feature_lengths.cuda()

            # padded_audio_features = []
            # padded_audio_feature_masks = []
            # audio_feature_lengths = [ele.audio_feature_length for ele in batch]
            # max_feature_length = self.configs['max_audio_length']

            # for ele in batch:
            #     padding_feature_len = max_feature_length - ele.audio_feature_length
            #     padded_audio_features.append(
            #         F.pad(ele.audio_feature, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
            #     padded_audio_feature_masks.append([1] * ele.audio_feature_length + [0] * padding_feature_len)
            # audio_features = torch.cat(padded_audio_features, dim=0)
            # audio_feature_masks = torch.IntTensor(padded_audio_feature_masks) > 0
            # audio_feature_lengths = torch.IntTensor(audio_feature_lengths)
            # if self.configs['use_gpu']:
            #     audio_features = audio_features.cuda()
            #     audio_feature_masks = audio_feature_masks.cuda()
            #     audio_feature_lengths = audio_feature_lengths.cuda()
        else:
            audio_features, audio_feature_masks, audio_feature_lengths = None, None, None
        
        char_emb_ids = torch.LongTensor([ele.char_emb_ids for ele in batch])
        char_lens = torch.LongTensor([ele.char_len for ele in batch])
        char_emb_mask = torch.FloatTensor([ele.char_emb_mask for ele in batch])

        if configs['use_gpu']:
            char_emb_ids = char_emb_ids.cuda()
            char_lens = char_lens.cuda()
            char_emb_mask = char_emb_mask.cuda()

        return {"guids": guids, "audio_features": audio_features,
                        "audio_feature_masks": audio_feature_masks, "audio_feature_lengths": audio_feature_lengths,
                        "char_emb_ids": char_emb_ids, "char_lens": char_lens, "char_emb_mask": char_emb_mask}

def init_ctc_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.tensor(mean).float(),
            torch.tensor(istd).float())
    else:
        global_cmvn = None
    
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')

    proj_size = configs['encoder_conf'].get('proj_size', configs['encoder_conf']['output_size'])
    if 'proj_size' in configs['encoder_conf']:
        configs['encoder_conf'].pop('proj_size')
    
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    ctc = CTC(vocab_size, proj_size, blank_id=configs['blank_id'])

    model = CTCModel(vocab_size,
                    proj_size,
                    encoder,
                    ctc,
                    sos=configs['sos'],
                    eos=configs['eos'],
                    blank_id=configs['blank_id'])
    return model
