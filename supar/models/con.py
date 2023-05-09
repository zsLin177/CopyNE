# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine
from supar.structs import CRFConstituency, MFVIConstituency
from supar.utils import Config
from supar.utils.alg import cky


class CRFConstituencyModel(Model):
    r"""
    The implementation of CRF Constituency Parser (:cite:`zhang-etal-2020-fast`),
    also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if ``feat='char'``. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            Either take the first subtoken ('first'), the last subtoken ('last'), or a mean over all ('mean').
            Default: 'mean'.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        freeze (bool):
            If ``True``, freezes bert layers. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 feat=['char'],
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_span_l = MLP(n_in=self.args.n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_span_r = MLP(n_in=self.args.n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=self.args.n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=self.args.n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)

        self.span_attn = Biaffine(n_in=n_mlp_span, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """

        x = self.encode(words, feats)

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels, in which positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and original constituent scores
                of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.crf(s_span, mask, span_mask, mbr)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """

        span_preds = cky(s_span.unsqueeze(-1), mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j, _ in cons] for cons, labels in zip(span_preds, label_preds)]


class VIConstituencyModel(CRFConstituencyModel):
    r"""
    The implementation of Constituency Parser using variational inference.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if ``feat='char'``. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            Either take the first subtoken ('first'), the last subtoken ('last'), or a mean over all ('mean').
            Default: 'mean'.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        freeze (bool):
            If ``True``, freezes bert layers. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_pair (int):
            Binary factor MLP size. Default: 100.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        max_iter (int):
            Max iteration times for Variational Inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 feat=['char'],
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_pair=100,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_span_l = MLP(n_in=self.args.n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_span_r = MLP(n_in=self.args.n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_pair_l = MLP(n_in=self.args.n_hidden, n_out=n_mlp_pair, dropout=mlp_dropout)
        self.mlp_pair_r = MLP(n_in=self.args.n_hidden, n_out=n_mlp_pair, dropout=mlp_dropout)
        self.mlp_pair_b = MLP(n_in=self.args.n_hidden, n_out=n_mlp_pair, dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=self.args.n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=self.args.n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)

        self.span_attn = Biaffine(n_in=n_mlp_span, bias_x=True, bias_y=False)
        self.pair_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.inference = MFVIConstituency(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """

        x = self.encode(words, feats)

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        pair_l = self.mlp_pair_l(x)
        pair_r = self.mlp_pair_r(x)
        pair_b = self.mlp_pair_b(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        s_pair = self.pair_attn(pair_l, pair_r, pair_b).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_pair, s_label

    def loss(self, s_span, s_pair, s_label, charts, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels, in which positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.inference((s_span, s_pair), mask, span_mask)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = self.args.interpolation * label_loss + (1 - self.args.interpolation) * span_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """

        span_preds = cky(s_span.unsqueeze(-1), mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j, _ in cons] for cons, labels in zip(span_preds, label_preds)]
