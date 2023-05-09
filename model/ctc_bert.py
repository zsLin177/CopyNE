import imp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from supar.modules.transformer import TransformerEmbedding
from supar.modules.mlp import MLP
from supar.modules.affine import Biaffine
from supar.modules.cma import CrossAttention
from supar.modules.lstm import VariationalLSTM
from supar.utils.fn import stripe
from wenet.transformer.asr_model import init_ctc_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MIN = -1e16

class BiLSTMPredictor(nn.Module):
    def __init__(self,
                n_words,
                n_embed=100,
                n_out=768,
                pad_index=0,
                blank_idx=1):
        super().__init__()
        self.pad_idx = pad_index
        self.blank_idx = blank_idx
        self.n_out = n_out
        self.word_embed = nn.Embedding(num_embeddings=n_words, embedding_dim=n_embed)
        self.encoder = VariationalLSTM(n_embed, n_out//2, num_layers=3, bidirectional=True, dropout=0.2)

    def forward(self, words):
        """
        words: [B, Nmax]
        """
        word_embed = self.word_embed(words)
        x = pack_padded_sequence(word_embed, words.ne(self.pad_idx).sum(1).tolist(), True, False)
        x, _ = self.encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        return x


class CTCBertModel(nn.Module):
    def __init__(self,
                configs,
                ctc_path,
                bert,
                if_fix_bert=True,
                with_ctc_loss=True,
                with_align_loss=False,
                use_lstm_predictor=False,
                n_words=21128,
                bert_pad_idx=0,
                bert_blank_idx=1,
                bert_insert_blank=False
                ):
        """
        ctc_path: path to the pre-trained ctc model. Or the None, it means that train with ctc loss and align loss at the same time
        bert_path: path to the bert model
        if_fix_bert: whether to fine-tune bert
        with_ctc_loss: whether there is ctc loss
        """
        super().__init__()
        self.with_ctc_loss = with_ctc_loss
        self.with_align_loss = with_align_loss
        self.use_lstm_predictor = use_lstm_predictor
        self.bert_blank_idx = bert_blank_idx
        self.bert_insert_blank = bert_insert_blank
        # default 768, no dropout, that directly use the output of last layer
        self.bert_embed = TransformerEmbedding(model=bert,
                                               n_layers=1,
                                               requires_grad=not if_fix_bert)
        
        if ctc_path != 'None':
            self.ctc_model = init_ctc_model(configs)
            self.ctc_model.load_state_dict(torch.load(ctc_path))
        else:
            self.ctc_model = init_ctc_model(configs)

        if self.use_lstm_predictor:
            self.predictor = BiLSTMPredictor(n_words, pad_index=0, blank_idx=1)
            assert self.predictor.n_out == self.bert_embed.n_out

        if self.with_align_loss:
            # 0: shift; 1:reduce
            self.joiner = nn.Sequential(
                nn.Linear(self.bert_embed.n_out, self.bert_embed.n_out//2),
                nn.ReLU(),
                nn.Linear(self.bert_embed.n_out//2, 2)
            )
            # self.joiner = nn.Linear(self.bert_embed.n_out, 2)

        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, 
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                bert_input: torch.Tensor,
                bert_lens=None):
        
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Speech encoder
        encoder_out, encoder_mask = self.ctc_model.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.ctc_model.proj_mlp(encoder_out)

        # 2. Text encoder
        if not self.use_lstm_predictor:
            text_coder_out = self.bert_embed(bert_input)
            txt_out = text_coder_out
            if self.bert_insert_blank:
                # [768,]
                blank_embed = self.bert_embed.get_embed(self.bert_blank_idx)
                n_txtcoder_out = text_coder_out.new_zeros(text_coder_out.shape[0], 2*text_coder_out.shape[1]-1, text_coder_out.shape[2])
                n_txtcoder_out[:, range(0, n_txtcoder_out.shape[1], 2)] = text_coder_out
                n_txtcoder_out[:, range(1, n_txtcoder_out.shape[1], 2)] = blank_embed
                txt_out = n_txtcoder_out[:, 1:]
        else:
            bert_input = bert_input.squeeze(-1)
            bert_input[:, 0] = self.predictor.blank_idx
            bert_input[range(bert_input.shape[0]), bert_lens-1] = self.predictor.blank_idx
            txt_out = self.predictor(bert_input)

        return encoder_out, encoder_out_lens, txt_out

    def loss(self, encoder_out, encoder_out_lens, bert_out, text, text_lengths, bert_out_lens, epoch=30, theta=30):
        """
        compute the pre-trained loss, ctc loss and align loss
        """
        if self.with_ctc_loss:
            loss_ctc = self.ctc_model.loss(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc = 0
        if self.with_align_loss:
            # batch_size = encoder_out.shape[0]
            # loss_align = self.criterion(encoder_out[:, 0], bert_out[:, 0]) / batch_size
            if self.bert_insert_blank:
                n_bert_out_lens = (bert_out_lens-2)*2+1
            loss_align = self.align_loss(encoder_out, encoder_out_lens, bert_out, n_bert_out_lens)
        else:
            loss_align = 0
        
        return loss_ctc + loss_align, loss_ctc, loss_align

        # interp = min(0.5, 0.1+(0.4/theta)*epoch)
        # return interp * loss_align + (1-interp) * loss_ctc

    def joiner_forward(self, speech_out, bert_out):
        """
        speech_out: [B, Tmax, D]
        bert_out: [B, Nmax, D]
        """
        # [B, Nmax, Tmax, D]
        out = speech_out.unsqueeze(1) + bert_out.unsqueeze(2)
        out = F.relu(out)
        # [B, Nmax, Tmax, 2]
        out = self.joiner(out)
        return out

    @torch.enable_grad()
    def align(self, speech_out, speech_out_lens, bert_out, bert_out_lens):
        """
        speech_out: [B, Tmax, D]
        speech_out_lens: [B]
        bert_out: [B, Nmax, D]
        bert_out_lens: [B] (include [CLS] and [SEP])
        """
        if self.bert_insert_blank:
            n_bert_out_lens = (bert_out_lens-2)*2+1

        batch_size, Tmax, _ = speech_out.shape
        Nmax = bert_out.shape[1]
        # [batch_size, Nmax, Tmax, 2]
        joiner_out = self.joiner_forward(speech_out, bert_out).log_softmax(-1)
        # [batch_size, Nmax, Tmax]
        # log_alpha = torch.zeros(batch_size, Nmax, Tmax, requires_grad=True).to(speech_out.device)
        # log_alpha[:, 0, 0] = 1
        log_alpha = torch.ones(batch_size, Nmax, Tmax, requires_grad=True).to(speech_out.device)
        log_alpha = log_alpha * MIN
        log_alpha[:, 0, 0] = 0
        for i in range(1, Tmax):
            log_alpha[:, 0, i] = log_alpha[:, 0, i-1] + joiner_out[:, 0, i-1, 0]

        Realmax = torch.min(n_bert_out_lens, speech_out_lens).max().item()
        for i in range(1, Realmax):
            log_alpha[:, i, i] = log_alpha[:, i-1, i-1] + joiner_out[:, i-1, i-1, 1]

        for i in range(1, Realmax):
            for j in range(i+1, Tmax):
                log_alpha[:, i, j], _ = torch.stack([
                    log_alpha[:, i, j-1] + joiner_out[:, i, j-1, 0],
                    log_alpha[:, i-1, j-1] + joiner_out[:, i-1, j-1, 1]
                ]).max(0)

        log_probs = log_alpha[range(batch_size), torch.min(n_bert_out_lens-1, speech_out_lens-1), speech_out_lens-1].sum()
        grd, = autograd.grad(log_probs, joiner_out)
        choose_mask = grd.long().bool()
        return choose_mask

    def align_loss(self, speech_out, speech_out_lens, bert_out, bert_out_lens):
        """
        speech_out: [B, Tmax, D]
        speech_out_lens: [B]
        bert_out: [B, Nmax, D]
        bert_out_lens: [B] (include [CLS] and [SEP])
        """
        batch_size, Tmax, _ = speech_out.shape
        Nmax = bert_out.shape[1]
        # [batch_size, Nmax, Tmax, 2]
        joiner_out = self.joiner_forward(speech_out, bert_out).log_softmax(-1)
        # [batch_size, Nmax, Tmax]
        # log_alpha = torch.zeros(batch_size, Nmax, Tmax, requires_grad=True).to(speech_out.device)
        # log_alpha[:, 0, 0] = 1
        log_alpha = torch.ones(batch_size, Nmax, Tmax, requires_grad=True).to(speech_out.device)
        log_alpha = log_alpha * MIN
        log_alpha[:, 0, 0] = 0
        for i in range(1, Tmax):
            log_alpha[:, 0, i] = log_alpha[:, 0, i-1] + joiner_out[:, 0, i-1, 0]

        Realmax = torch.min(bert_out_lens, speech_out_lens).max().item()
        for i in range(1, Realmax):
            log_alpha[:, i, i] = log_alpha[:, i-1, i-1] + joiner_out[:, i-1, i-1, 1]

        for i in range(1, Realmax):
            for j in range(i+1, Tmax):
                log_alpha[:, i, j] = torch.logsumexp(torch.stack([
                    log_alpha[:, i, j-1] + joiner_out[:, i, j-1, 0],
                    log_alpha[:, i-1, j-1] + joiner_out[:, i-1, j-1, 1]
                ]), dim=0)
        
        # [batch_size]
        log_probs = log_alpha[range(batch_size), torch.min(bert_out_lens-1, speech_out_lens-1), speech_out_lens-1]
        return -log_probs.mean()

    def ctc_loss(self, encoder_out, encoder_out_lens, text, text_lengths):
        loss_ctc = self.ctc_model.loss(encoder_out, encoder_out_lens, text, text_lengths)
        return loss_ctc

    def token_level_speech_repr(self, encoder_out, bert_input, T=2.0):
        """
        encoder_out: [B, Tmax, d]
        bert_input: [B, Nmax]
        """
        # [B, vocab, Tmax]
        logits = self.ctc_model.ctc.logits(encoder_out, T)
        # [B, Nmax, Tmax]
        selected_logits = logits[torch.arange(0, logits.shape[0], device=logits.device, dtype=torch.long).unsqueeze(-1), bert_input]
        # [B, Nmax, d]
        token_level_sp_repr = torch.matmul(selected_logits.softmax(-1), encoder_out)
        return token_level_sp_repr


def load_ctcbertmodel(configs,
                     ctc_path,
                     bert,
                     ctc_bert_path,
                     if_fix_bert=True,
                     with_ctc_loss=True,
                     with_align_loss=False):
    model = CTCBertModel(configs, ctc_path, bert, if_fix_bert, with_ctc_loss, with_align_loss)
    if ctc_bert_path != 'None':
        model.load_state_dict(torch.load(ctc_bert_path))
    return model

class CTCBertNERModel(nn.Module):
    def __init__(self,
                n_labels,
                configs,
                ctc_path,
                bert,
                ctc_bert_path,
                if_fix_bert=False,
                with_ctc_loss=True,
                with_align_loss=False,
                encoder_out=768,
                n_mlp=150,
                mlp_dropout=0.2,
                use_speech=True,
                use_tokenized=False,
                bert_pad_idx=0):
        super().__init__()
        """
        use the above CTCBertModel as the encoder
        """
        self.use_speech = use_speech
        self.n_labels = n_labels
        self.with_ctc_loss = with_ctc_loss
        self.use_tokenized = use_tokenized
        self.bert_pad_idx = bert_pad_idx
        self.encoder = load_ctcbertmodel(configs,
                                        ctc_path,
                                        bert,
                                        ctc_bert_path,
                                        if_fix_bert,
                                        with_ctc_loss,
                                        with_align_loss)
        self.cma = CrossAttention(encoder_out)
        self.mlp_e = MLP(n_in=encoder_out, n_out=n_mlp, dropout=mlp_dropout, activation=False)
        self.mlp_s = MLP(n_in=encoder_out, n_out=n_mlp, dropout=mlp_dropout, activation=False)

        # n_labels do not contain [NULL]
        self.biaffine_attn = Biaffine(n_in=n_mlp, n_out=n_labels+1, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, 
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                bert_input: torch.Tensor):

        # 2. Text encoder
        # [batch_size, t_len, d]
        bert_out = self.encoder.bert_embed(bert_input)
        
        if self.use_speech:
            assert text_lengths.dim() == 1, text_lengths.shape
            # Check that batch_size is unified
            assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                    text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                            text.shape, text_lengths.shape)
            # 1. Speech encoder
            encoder_out, encoder_mask = self.encoder.ctc_model.encoder(speech, speech_lengths)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)
            # [batch_size, s_len, d]
            encoder_out = self.encoder.ctc_model.proj_mlp(encoder_out)

            # 3. use cma to fuse
            if not self.use_tokenized:
                # [batch_size, t_len, d]
                out = self.cma(bert_out, encoder_out, encoder_out, ~(encoder_mask.squeeze(1).bool()))
            else:
                # [batch_size, t_len, d]
                token_level_sp_repr = self.encoder.token_level_speech_repr(encoder_out, bert_input.squeeze(-1))
                out = self.cma(bert_out, token_level_sp_repr, token_level_sp_repr, bert_input.squeeze(-1).eq(self.bert_pad_idx))
        else:
            out = bert_out
        
        if self.with_ctc_loss:
            ctc_loss = self.encoder.ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            ctc_loss = 0

        # 4. compute biaffine scores
        e_repr = self.mlp_e(out)
        s_repr = self.mlp_s(out)
        # [batch_size, t_len, t_len, n_labels+1]
        # row: end; column: start
        score = self.biaffine_attn(e_repr, s_repr).permute(0, 2, 3, 1)
        return score, ctc_loss

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, t_len, t_len, n_labels+1]
        gold_labels: [batch_size, t_len, t_len]
        mask: [batch_size, t_len, t_len]: lower triangular & pad
        """
        loss = self.criterion(score[mask], gold_labels[mask])
        return loss

    @torch.enable_grad()
    def decode(self, score, mask, pad_mask):
        """
        this is for flat ner
        score: [batch_size, t_len, t_len, n_labels+1]
        mask: [batch_size, t_len, t_len]: lower triangular & pad
        pad_mask: [batch_size, t_len] [CLS] and [SEP] is True
        """
        batch_size, t_len = score.shape[0], score.shape[1]
        values, indices = score.softmax(-1).max(-1)
        null_mask = indices.eq(self.n_labels)
        # set the null label spans and illegal spans to 0, others all larger than 0
        # [batch_size, t_len, t_len] row:end column:start
        # here we set the score of illegal spans to 0, if set to -1, it seems not true
        values = values.masked_fill(null_mask+(~mask), -1)

        # dp = torch.zeros((batch_size, t_len), dtype=torch.float, device=score.device, requires_grad=True)
        dp = score.new_zeros((batch_size, t_len))
        # [t_len, batch_size]
        dp = dp.transpose(0, 1)
        # [start, end, batch_size]
        values = values.permute(2, 1, 0)
        for i in range(1, t_len):
            tmp_tensor = score.new_zeros(i, batch_size)
            # tmp_tensor = torch.zeros((i, batch_size), dtype=torch.float, device=score.device, requires_grad=True)
            # tmp_tensor = dp[:i] + values[1:i+1, i]
            tmp_tensor = torch.cat([dp[:i]+values[1:i+1, i], dp[:i]], dim=0)
            dp[i], _ = tmp_tensor.max(0)
        # [batch_size, t_len]
        dp = dp.transpose(0, 1)
        need_len = pad_mask.sum(-1) - 1
        z = dp[range(batch_size), need_len].sum()
        grd, = autograd.grad(z, values)
        # [batch_size, end, start]
        choose_mask = grd.permute(2, 1, 0).long().bool()
        indices = indices.masked_fill(~choose_mask, -1)
        return indices

    @torch.enable_grad()
    def nested_decode(self, score, mask, pad_mask):
        """
        this is for nested ner
        score: [batch_size, t_len, t_len, n_labels+1]
        mask: [batch_size, t_len, t_len]: lower triangular & pad
        pad_mask: [batch_size, t_len] [CLS] and [SEP] is True
        """
        batch_size, t_len = score.shape[0], score.shape[1]
        values, indices = score.softmax(-1).max(-1)
        null_mask = indices.eq(self.n_labels)
        # set the null label spans and illegal spans to 0, or -1, others all larger than 0
        # [batch_size, t_len, t_len] row:end column:start
        values = values.masked_fill(null_mask+(~mask), -1)
        dp = score.new_zeros((batch_size, t_len, t_len))
        # [start, end, batch_size]
        dp = dp.permute(1, 2, 0)
        # [start, end, batch_size]
        values = values.permute(2, 1, 0)

        # C.K.Y
        for lens in range(2, t_len+1):
            for st in range(0, t_len-lens+1):
                # dp[st, st+lens-1]
                # st<=k<=st+lens-2
                tmp_tensor = torch.cat([
                dp[st, st:st+lens-1] + values[st, st:st+lens-1] + dp[st+1:st+lens, st+lens-1] + values[st+1:st+lens, st+lens-1], 
                dp[st, st:st+lens-1] + dp[st+1:st+lens, st+lens-1] + values[st+1:st+lens, st+lens-1],
                dp[st, st:st+lens-1] + values[st, st:st+lens-1] + dp[st+1:st+lens, st+lens-1],
                dp[st, st:st+lens-1] + dp[st+1:st+lens, st+lens-1]
                ], dim=0)
                dp[st, st+lens-1], _ = tmp_tensor.max(0)
        # [batch_size, start, end]
        dp = dp.permute(2, 0, 1)
        need_len = pad_mask.sum(-1) - 1
        z = dp[range(batch_size), 0, need_len].sum()
        grd, = autograd.grad(z, values)
        # [batch_size, end, start]
        choose_mask = grd.permute(2, 1, 0).long().bool()
        indices = indices.masked_fill(~choose_mask, -1)
        return indices

    @torch.enable_grad()
    def fast_nested_decode(self, score, mask, pad_mask):
        """
        this is for nested ner, use stripe to accelerate to compution of CKY.
        score: [batch_size, t_len, t_len, n_labels+1]
        mask: [batch_size, t_len, t_len]: lower triangular & pad
        pad_mask: [batch_size, t_len] [CLS] and [SEP] is True
        """
        batch_size, t_len = score.shape[0], score.shape[1]
        values, indices = score.softmax(-1).max(-1)
        null_mask = indices.eq(self.n_labels)
        # set the null label spans and illegal spans to 0, or -1, others all larger than 0
        # [batch_size, t_len, t_len] row:end column:start
        values = values.masked_fill(null_mask+(~mask), -1)
        dp = score.new_zeros((batch_size, t_len, t_len))
        # [start, end, batch_size]
        dp = dp.permute(1, 2, 0)
        # [start, end, batch_size]
        values = values.permute(2, 1, 0)

        # CKY
        for lens in range(2, t_len+1):
            n = t_len - lens + 1
            # [n, lens-1, batch_size]
            pre_dps = stripe(dp, n, lens-1) + stripe(dp, n, lens-1, (1, lens-1), 0)
            # [n, (lens-1)*4, batch_size]
            tmp_tensor = torch.cat([pre_dps, pre_dps+stripe(values, n, lens-1), pre_dps+stripe(values, n, lens-1, (1, lens-1), 0), pre_dps+stripe(values, n, lens-1)+stripe(values, n, lens-1, (1, lens-1), 0)], dim=1)
            # [batch_size, n, (lens-1)*4]
            tmp_tensor = tmp_tensor.permute(2, 0, 1)
            max_score, _ = tmp_tensor.max(-1)
            dp.diagonal(lens-1).copy_(max_score)
        # [batch_size, start, end]
        dp = dp.permute(2, 0, 1)
        need_len = pad_mask.sum(-1) - 1
        z = dp[range(batch_size), 0, need_len].sum()
        grd, = autograd.grad(z, values)
        # [batch_size, end, start]
        choose_mask = grd.permute(2, 1, 0).long().bool()
        indices = indices.masked_fill(~choose_mask, -1)
        return indices
