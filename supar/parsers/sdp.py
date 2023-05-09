# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import (BiaffineSemanticDependencyModel,
                          VISemanticDependencyModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class BiaffineSemanticDependencyParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser (:cite:`dozat-etal-2018-simpler`).
    """

    NAME = 'biaffine-semantic-dependency'
    MODEL = BiaffineSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
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
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
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
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
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

        bar, metric = progress_bar(loader), ChartMetric()

        for i, (words, *feats, labels) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        for words, *feats, labels in loader:
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            # loss = self.model.loss(s_edge, s_label, labels, mask)
            # total_loss += loss.item()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        for words, *feats in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
            label_preds = self.model.decode(s_edge,
                                            s_label).masked_fill(~mask, -1)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            if self.args.prob:
                preds['probs'].extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        return preds

    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
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
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                fix_len=args.fix_len,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            WORD.build(
                train, args.min_freq,
                (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        LABEL.build(train)
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'unk_index':
            WORD.unk_index,
            'bos_index':
            WORD.bos_index,
            'lr':
            5e-5,
            'epochs': 10, 
            'warmup':
            0.1,
            'interpolation': args.itp
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class VISemanticDependencyParser(BiaffineSemanticDependencyParser):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference (:cite:`wang-etal-2019-second`).
    """

    NAME = 'vi-semantic-dependency'
    MODEL = VISemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
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
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
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
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
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

        bar, metric = progress_bar(loader), ChartMetric()

        for i, (words, *feats, labels) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           s_label, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        for words, *feats, labels in loader:
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           s_label, labels, mask)
            total_loss += loss.item()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
        total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        for words, *feats in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
            label_preds = self.model.decode(s_edge,
                                            s_label).masked_fill(~mask, -1)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            if self.args.prob:
                preds['probs'].extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        return preds

    
    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
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
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                fix_len=args.fix_len,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            WORD.build(
                train, args.min_freq,
                (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        LABEL.build(train)
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'unk_index':
            WORD.unk_index,
            'bos_index':
            WORD.bos_index,
            'lr':
            5e-5,
            'epochs': 10, 
            'warmup':
            0.1
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)
