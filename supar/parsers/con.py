# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import CRFConstituencyModel, VIConstituencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, eos, pad, unk
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import SpanMetric
from supar.utils.transform import Tree

logger = get_logger(__name__)


class CRFConstituencyParser(Parser):
    r"""
    The implementation of CRF Constituency Parser (:cite:`zhang-etal-2020-fast`).
    """

    NAME = 'crf-constituency'
    MODEL = CRFConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              mbr=True,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, mbr=True,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang='en', buckets=8, batch_size=5000, prob=False, mbr=True, verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, (words, *feats, trees, charts) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, _ = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SpanMetric()

        for words, *feats, trees, charts in loader:
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, s_span = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
            total_loss += loss.item()
            metric([Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                   [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'trees': [], 'probs': [] if self.args.prob else None}
        for words, *feats, trees in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            lens = mask[:, 0].sum(-1)
            s_span, s_label = self.model(words, feats)
            if self.args.mbr:
                s_span = self.model.crf(s_span, mask, mbr=True)
            chart_preds = self.model.decode(s_span, s_label, mask)
            preds['trees'].extend([Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                                   for tree, chart in zip(trees, chart_preds)])
            if self.args.prob:
                preds['probs'].extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)])

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
        TAG, CHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField('words',
                                pad=t.pad_token,
                                unk=t.unk_token,
                                bos=t.cls_token or t.cls_token,
                                eos=t.sep_token or t.sep_token,
                                fix_len=args.fix_len,
                                tokenize=t.tokenize,
                                fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos, eos=eos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField('bert',
                                    pad=t.pad_token,
                                    unk=t.unk_token,
                                    bos=t.cls_token or t.cls_token,
                                    eos=t.sep_token or t.sep_token,
                                    fix_len=args.fix_len,
                                    tokenize=t.tokenize,
                                    fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
                BERT.vocab = t.get_vocab()
        TREE = RawField('trees')
        CHART = ChartField('charts')
        transform = Tree(WORD=(WORD, CHAR, BERT), POS=TAG, TREE=TREE, CHART=CHART)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        CHART.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_labels': len(CHART.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class VIConstituencyParser(CRFConstituencyParser):
    r"""
    The implementation of Constituency Parser using variational inference.
    """

    NAME = 'vi-constituency'
    MODEL = VIConstituencyModel

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang='en', buckets=8, batch_size=5000, prob=False,  verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, (words, *feats, trees, charts) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_span, s_pair, s_label = self.model(words, feats)
            loss, _ = self.model.loss(s_span, s_pair, s_label, charts, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SpanMetric()

        for words, *feats, trees, charts in loader:
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_span, s_pair, s_label = self.model(words, feats)
            loss, s_span = self.model.loss(s_span, s_pair, s_label, charts, mask)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
            total_loss += loss.item()
            metric([Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                   [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'trees': [], 'probs': [] if self.args.prob else None}
        for words, *feats, trees in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            lens = mask[:, 0].sum(-1)
            s_span, s_pair, s_label = self.model(words, feats)
            s_span = self.model.inference((s_span, s_pair), mask)
            chart_preds = self.model.decode(s_span, s_label, mask)
            preds['trees'].extend([Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                                   for tree, chart in zip(trees, chart_preds)])
            if self.args.prob:
                preds['probs'].extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)])

        return preds
