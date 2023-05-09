# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
import subprocess
import dill
# import supar
import torch
import torch.distributed as dist
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.fn import download
from supar.utils.logging import init_logger, logger
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import random

def change_BES(source_file, tgt_file, task):
    '''
    for BES
    '''
    sent_idx = 0
    sum_conf1_count = 0
    sum_conf2_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sent_idx += 1
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2 = produce_column_BES(this_prd_arc, this_prd_idx)
            sum_conf1_count += con1
            sum_conf2_count += con2
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_conf1_count))
    print('conflict label:'+str(sum_conf2_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')

def produce_column_BES(relas, prd_idx):
    # used for BES
    flag = 0
    count = 0
    count2 = 0
    column = ['*'] * len(relas)
    column[prd_idx-1] = '(V*)'
    args = []
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            # column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            # column.append('*')
            i += 1
        elif (len(rel) == 0):
            # column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if(position_tag == 'E'):
                column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
            elif position_tag == 'S':
                args.append([i, i, label])
                i += 1
            else:
                span_start = i
                i += 1
                if i>=len(relas):
                    # column.append('(' + label + '*' + ')')
                    i += 1
                elif len(relas[i]) == 0:
                    while i < len(relas) and len(relas[i]) == 0:
                        i += 1
                    if i < len(relas):
                        if relas[i][0].startswith('E-'):
                            new_label = relas[i][0][2:]
                            if label != new_label:
                                count2 += 1
                            else:
                                args.append([span_start, i, label])
                        else:
                            count += 1
                        i += 1
                elif relas[i][0].startswith('B-'):
                    count += 1
                    continue
                elif relas[i][0].startswith('E-'):
                    new_label = relas[i][0][2:]
                    args.append([span_start, i, label])
                    if label != new_label:
                        count2 += 1
                    i += 1
                else:
                    # relas[i][0].startswith('S-')
                    new_label = relas[i][0][2:]
                    args.append([i, i, new_label])
                    i += 1
                    count += 1

    for st, ed, role in args:
        length = ed-st+1
        if length == 1:
            column[st] = '(' + role + '*' + ')'
        else:
            column[st] = '(' + role + '*'
            column[ed] = '*' + ')'

    return column, count, count2

def change_BE(source_file, tgt_file, task):
    sum_false_count = 0
    sum_mismatch = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    # change simple crosstag conllu to target type
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # 先找出所有的谓词
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, count, mis_match = produce_column_BE(this_prd_arc, this_prd_idx)
            sum_false_count += count
            sum_mismatch += mis_match
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_false_count))
    print('conflict mismatch:'+str(sum_mismatch))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')


def produce_column_BE(relas, prd_idx):
    # used for simple crosstag
    # 暂时是直接按照预测的B、I进行划分
    count = 0
    mismatch = 0
    column = []
    # span_start = -1
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        # print(i)
        # print(relas)
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            column.append('*')
            i += 1
        elif(rel == ['Other']):
            column.append('*')
            i += 1
        elif (len(rel) == 0):
            column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if(position_tag == 'I'):
                column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
            else:
            # if(position_tag in ('B', 'I')):
                span_start = i
                span_end = -1
                i += 1
                # labels = {}
                # labels[label] = 1
                while (i < len(relas)):
                    if (len(relas[i]) == 0 or relas[i] == ['Other']):
                        i += 1
                        continue
                    else:
                        # relas[i][0][0] == 'B' or 'I'
                        if (relas[i][0][0] == 'B'):
                            break
                        else:
                            span_end = i
                            label2 = relas[i][0][2:]  # 以后面那个作为label
                            i += 1
                            break
                if (span_end != -1):
                    if (label == label2):
                        length = span_end - span_start + 1
                        column.append('(' + label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
                    else:
                        length = span_end - span_start + 1
                        # del
                        column += ['*'] * length

                        # contain label1
                        # column.append('(' + label + '*')
                        # column += ['*'] * (length - 2)
                        # column.append('*' + ')')

                        # contain label2
                        # column.append('(' + label2 + '*')
                        # column += ['*'] * (length - 2)
                        # column.append('*' + ')')

                        mismatch += 1
                else:
                    column.append('(' + label + '*' + ')')
                    column += ['*'] * (i - 1 - span_start)
    return column, count, mismatch



def get_results(gold_path, pred_path, file_seed, task, schema):
    _SRL_CONLL_EVAL_SCRIPT = 'conll05-original-style/eval.sh'
    tgt_temp_file = 'tgt_temp_file' + file_seed
    if schema == 'BE':
        change_BE(pred_path, tgt_temp_file, task)
    elif schema == "BES":
        change_BES(pred_path, tgt_temp_file, task)
    else:
        raise NotImplementedError
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]

    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, tgt_temp_file, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    os.remove(tgt_temp_file)
    conll_recall = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_precision = float(str(eval_info2).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[5])

    return conll_recall, conll_precision, conll_f1, lisa_f1

class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              clip=5.0, patience=100, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size//args.update_steps, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        else:
            from transformers import AdamW, get_linear_schedule_with_warmup
            steps = len(train.loader) * args.epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': c, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, c in self.model.named_parameters()],
                args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            dev_metric = self._evaluate(dev.loader)
            logger.info(f"- {dev_metric}")
            test_metric = self._evaluate(test.loader)
            logger.info(f"- {test_metric}")

            t = datetime.now() - start
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        metric = self.load(**args)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':5} {best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        metric = self._evaluate(dataset.loader, args.given_prd)
        elapsed = datetime.now() - start
        logger.info(f"- {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return metric

    def predict(self, data, pred=None, lang='en', buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        # self.transform.eval()
        self.transform.train()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        if(args.task in ('05', '12')):
            rand_file_seed1 = random.randint(1,100)
            rand_file_seed2 = random.randint(1,100)
            test_conll_f1, test_lisa_f1 = 0, 0
            conll_recall, conll_precision, test_conll_f1, test_lisa_f1 = get_results(args.gold, pred, str(rand_file_seed1)+'-'+str(rand_file_seed2), args.task, args.schema)
            logger.info(f"-P:{conll_precision:6.4} R:{conll_recall:6.4} F1:{test_conll_f1:6.4}")

        return dataset

    def api(self, input, task='09', prd_idxs=None, lang='en', buckets=8, batch_size=5000, prob=False, **kwargs):
        """
        input: a string like "中国 建筑业 对 外 开放 始于 八十年代 。"
        return: spans
        """
        args = self.args.update(locals())
        self.transform.train()
        dataset = Dataset(self.transform, input, lang=lang)
        dataset.build(args.batch_size, args.buckets)
        preds = self._api(dataset.loader, prd_idxs, task)
        print(preds)
        return preds


    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError
    
    @torch.no_grad()
    def _api(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, reload=False, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained parser defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'crf-dep-en'``.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations and initiate the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(supar.MODEL.get(path, path), reload=reload))
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)
