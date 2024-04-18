import argparse

from asr import CTCAttentionASRParser, CLASCTCAttentionASRParser, CopyNEASRParser, nParaformerASRParser
from supar.utils.logging import init_logger, logger
import os
import torch
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def parse(parser):
    ddp_setup()
    parser.add_argument('--path', help='path to model file')
    parser.add_argument('--pre_model', type=str, default="None")
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_workers',
                        default=6,
                        type=int)
    parser.add_argument('--char_dict', 
                        default='data/sp_ner/chinese_char.txt', 
                        help='path to the char dict file')
    parser.add_argument('--cmvn', 
                        default='data_to_upload/aishell1_global_cmvn_mel80', 
                        help='global cmvn file')
    parser.add_argument('--config', 
                        default='conf/ctc_mel80.yaml', 
                        help='config file')
    parser.add_argument('--frame_length',
                        default=25,
                        type=int)
    parser.add_argument('--frame_shift',
                        default=10,
                        type=int)
    parser.add_argument('--max_frame_num',
                        default=10000,
                        type=int)
    parser.add_argument('--mysampler',
                        default=False,
                        action='store_true')
    parser.add_argument('--add_ne_feat',
                        default=False,
                        action='store_true')
    parser.add_argument('--ne_dict_file', default='data_to_upload/aishell_vocab/nelabel.vocab', help='path to the ne label dict file')
    parser.add_argument('--add_bert_feat',
                        default=False,
                        action='store_true')
    parser.add_argument('--bert_path', default='bert-base-chinese', help='path to the bert model')
    parser.add_argument('--e2e_ner',
                        default=False,
                        action='store_true')

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)

    torch.manual_seed(args.seed)
    if int((torch.__version__)[0]) > 1:
        torch.set_float32_matmul_precision('high') # it should be set to high for torch2.0
    init_logger(logger, os.path.join(args.path, f"{args.mode}.log"))
    logger.info('\n' + str(args))

    try:
        if args.mode == 'train':
            parser = nParaformerASRParser(args)
            logger.info(f'{parser.model}\n')
            parser.train()
        elif args.mode == 'evaluate':
            parser = nParaformerASRParser(args)
            logger.info(f'{parser.model}\n')
            parser.eval()
    except Exception as e:
        logger.error(e)
    finally:
        destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--train', default='data/sp_ner/new_train.json', help='path to train file')
    subparser.add_argument('--dev', default='data/end2end/dev_single_bracket.json', help='path to dev file')
    subparser.add_argument('--test', default='data/sp_ner/new_test.json', help='path to test file')

    subparser = subparsers.add_parser('evaluate', help='Evaluation.')
    subparser.add_argument('--input', default='data/aishell1_asr/test.json', help='path to input file')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')
    subparser.add_argument('--use_avg', default=False, action='store_true', help='use average model')

    parse(parser)
    
