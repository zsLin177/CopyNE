# -*- coding: utf-8 -*-

import torch.nn as nn
from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine
from supar.structs import LBPSemanticDependency, MFVISemanticDependency
from supar.utils import Config


class BiaffineSemanticDependencyModel(Model):
    r"""
    The implementation of Biaffine Semantic Dependency Parser (:cite:`dozat-etal-2018-simpler`).

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word representations. Default: 125.
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
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Edge MLP size. Default: 600.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
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
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_edge_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_edge_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_label_d = MLP(n_in=self.args.n_hidden,
                               n_out=n_mlp_label,
                               dropout=label_mlp_dropout,
                               activation=False)
        self.mlp_label_h = MLP(n_in=self.args.n_hidden,
                               n_out=n_mlp_label,
                               dropout=label_mlp_dropout,
                               activation=False)

        self.edge_attn = Biaffine(n_in=n_mlp_edge,
                                  n_out=2,
                                  bias_x=True,
                                  bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
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
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """

        x = self.encode(words, feats)

        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)
        label_d = self.mlp_label_d(x)
        label_h = self.mlp_label_h(x)

        # [batch_size, seq_len, seq_len, 2]
        s_egde = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_egde, s_label

    def loss(self, s_egde, s_label, labels, mask):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_egde[mask], edge_mask[mask].long())
        if (edge_mask.any()):
            label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            return self.args.interpolation * label_loss + (
                1 - self.args.interpolation) * edge_loss
        else:
            return edge_loss

    def decode(self, s_egde, s_label):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.BoolTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_egde.argmax(-1).lt(1), -1)


class VISemanticDependencyModel(BiaffineSemanticDependencyModel):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference (:cite:`wang-etal-2019-second`).

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
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
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Unary factor MLP size. Default: 600.
        n_mlp_pair (int):
            Binary factor MLP size. Default: 150.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .25.
        pair_mlp_dropout (float):
            The dropout ratio of binary factor MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: 'mfvi'.
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
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_pair=150,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 pair_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 inference='mfvi',
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_edge_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_edge_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_pair_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        self.mlp_pair_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        self.mlp_pair_g = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        self.mlp_label_d = MLP(n_in=self.args.n_hidden,
                               n_out=n_mlp_label,
                               dropout=label_mlp_dropout,
                               activation=False)
        self.mlp_label_h = MLP(n_in=self.args.n_hidden,
                               n_out=n_mlp_label,
                               dropout=label_mlp_dropout,
                               activation=False)

        self.edge_attn = Biaffine(n_in=n_mlp_edge, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.inference = (MFVISemanticDependency if inference == 'mfvi' else
                          LBPSemanticDependency)(max_iter)
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
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible edges of shape ``[batch_size, seq_len, seq_len]``,
                dependent-head-sibling triples of shape ``[batch_size, seq_len, seq_len, seq_len]`` and
                all possible labels on each edge of shape ``[batch_size, seq_len, seq_len, n_labels]``.
        """

        x = self.encode(words, feats)

        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)
        pair_d = self.mlp_pair_d(x)
        pair_h = self.mlp_pair_h(x)
        pair_g = self.mlp_pair_g(x)
        label_h = self.mlp_label_h(x)
        label_d = self.mlp_label_d(x)
        label_h = self.mlp_label_h(x)

        # [batch_size, seq_len, seq_len]
        s_egde = self.edge_attn(edge_d, edge_h)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->s)
        s_sib = self.sib_attn(pair_d, pair_d, pair_h)
        s_sib = (s_sib.triu() + s_sib.triu(1).transpose(-1, -2)).permute(
            0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->c)
        s_cop = self.cop_attn(pair_h, pair_d, pair_h).permute(0, 3, 1, 2)
        s_cop = s_cop.triu() + s_cop.triu(1).transpose(-1, -2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->g)
        s_grd = self.grd_attn(pair_g, pair_d, pair_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_egde, s_sib, s_cop, s_grd, s_label

    def loss(self, s_egde, s_sib, s_cop, s_grd, s_label, labels, mask):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_cop (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-coparent triples.
            s_grd (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-grandparent triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = labels.ge(0) & mask
        edge_loss, marginals = self.inference((s_egde, s_sib, s_cop, s_grd),
                                              mask, edge_mask.long())
        if(edge_mask.any()):
            label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            loss = self.args.interpolation * label_loss + (
                1 - self.args.interpolation) * edge_loss
            return loss, marginals
        else:
            return edge_loss, marginals

    def decode(self, s_egde, s_label):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_egde.lt(0.5), -1)
