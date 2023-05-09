# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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

from __future__ import print_function

import argparse
from cgi import test
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset, BaseNerDataset
from wenet.transformer.asr_model import init_asr_model
from wenet.transformer.sn_model import init_lstmbase_model, init_lstmbasesimple_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config
from wenet.utils.metric import SeqTagMetric
from wenet.utils.compute_evaluation_indicators import call_as_metric

import pdb

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--ner_dict', help='ner dict file', default=None)
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--connect_symbol',
                        default='',
                        type=str,
                        help='used to connect the output characters')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    ner_table = read_symbol_table(args.ner_dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf.update(copy.deepcopy(configs['bert_conf']))
    test_conf.update(copy.deepcopy(configs['ner_conf']))

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = BaseNerDataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False,
                           ner_table=ner_table)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    configs['num_ner_labels'] = len(ner_table)
    # Init asr model from configs
    model = init_lstmbase_model(configs)
    # model = init_lstmbasesimple_model(configs)


    # Load dict
    # char_dict = {v: k for k, v in symbol_table.items()}
    # eos = len(char_dict) - 1

    ner_dict = {v: k for k, v in ner_table.items()}


    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    metric = SeqTagMetric(ner_dict)
    model.eval()

    bert_pad_idx = configs['bert_conf']['pad_idx']
    bert_bos_idx = configs['bert_conf']['bos_idx']

    char_pad_idx = configs['lstm_conf']['char_pad_idx']
    char_bos_idx = configs['lstm_conf']['char_bos_idx']
    sum_num_sents = 0
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, bert_tokenids, ner_seq, char_tokenids = batch
            # [batch_size, seq_len] plus cls
            # bert_tokenids = bert_tokenids.to(device)
            ner_seq = ner_seq.to(device)
            char_tokenids = char_tokenids.to(device)
            score = model(char_tokenids)
            mask = char_tokenids.ne(char_pad_idx) & char_tokenids.ne(char_bos_idx)
            loss = model.loss(score, ner_seq, mask)
            preds = model.decode(score, mask)
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), ner_seq.masked_fill(~mask, -1))
            lengths = mask.sum(-1).tolist()
            preds = preds.tolist()
            sum_num_sents += ner_seq.size(0)
            
            for i, key in enumerate(keys):
                content = []
                for w in preds[i][0:lengths[i]]:
                    content.append(ner_dict[w])
                # logging.info('{} [{}]'.format(key, ','.join(content)))
                fout.write('{} [{}]\n'.format(key, ','.join(content)))
        logging.info('- {}\n'.format(metric))
    
    pred_file = args.result_file
    gold_file = args.test_data
    nested_ner_F, indicators = call_as_metric(pred_file, gold_file)
    logging.debug(f'test sum sents:{sum_num_sents}')
    logging.debug(f'P: {indicators[0]:6.2%} R: {indicators[1]:6.2%} F: {indicators[2]:6.2%}')


if __name__ == '__main__':
    main()
