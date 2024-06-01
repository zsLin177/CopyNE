
import os
import yaml
import copy
import torch
import torch.nn as nn
from model.bartsp import BartASRCorrection, BartSpeechNER
from utils.data import Dataset, collate_fn, NERDataset, ner_collate_fn, BartSeq2SeqDataset, bartseq2seq_collate_fn, BartTxtSeq2SeqDataset, barttxtseq2seq_collate_fn, CLASDataset, clas_collate_fn, copyne_collate_fn, CopyASRDataset, copyasr_collate_fn, process_audio
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler, DataLoader
from supar.utils.logging import init_logger, logger, progress_bar, get_logger
from wenet.transformer.asr_model import init_ctc_model, init_asr_model, init_copyasr_model, init_paraformer_model
from model.ctc_bert import CTCBertModel, CTCBertNERModel
from utils.process import read_symbol_table, read_context_table, build_context_tensor
from wenet.utils.scheduler import WarmupLR
import torch.optim as optim
from datetime import datetime, timedelta
from transformers import AutoTokenizer, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from model.bart_speech_ner import BartForEnd2EndSpeechNER
from supar.utils.metric import ChartMetric, Metric, AishellNerMetric
from utils.newoptim import PolynomialLR
from collections import Counter

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torchaudio as ta
from torchaudio.transforms import Resample

class Parser(object):
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
        
        vocab_size = len(read_symbol_table(args.char_dict))

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        self.configs = configs

        self.model = init_ctc_model(configs)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        
        if self.args.add_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        else:
            self.bert_tokenizer = None

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = Dataset(self.args.train, self.args.char_dict, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer)
        train_loader = DataLoader(train_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        
        dev_set = Dataset(self.args.dev, self.args.char_dict, bert_tokenizer=self.bert_tokenizer)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=4
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")
        
        num_epochs = self.configs.get('max_epoch', 100)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                
                loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()
        
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
            loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
            sum_dev_loss += loss.item()
        
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}\n')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        dataset = Dataset(self.args.input, self.args.char_dict, bert_tokenizer=self.bert_tokenizer)
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")
        
        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1

        bar = progress_bar(dataloader)
        with open(self.args.res, 'w') as fout:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                
                hyps, _ = self.model.ctc_greedy_search(
                    audio_feat,
                    feats_lengths,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    simulate_streaming=False)
                
                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    logger.info('{} {}'.format(key, ''.join(content)))
                    fout.write('{} {}\n'.format(key, ''.join(content)))

class CTCBertParser(object):
    def __init__(self, args) -> None:
        self.args = args
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
        
        if self.args.use_same_tokenizer:
            vocab_size = len(self.bert_tokenizer.get_vocab())
            configs['blank_id'] = 1
            configs['sos'] = 101
            configs['eos'] = 102
        else:
            vocab_size = len(read_symbol_table(args.char_dict))
            configs['blank_id'] = 0
            configs['sos'] = None
            configs['eos'] = None

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        self.configs = configs

        # if not hasattr(args, 'fix_bert'):
        #     fix_bert = True
        # else:
        #     fix_bert = args.fix_bert
        
        # if not hasattr(args, 'with_ctc_loss'):
        #     with_ctc_loss = False
        # else:
        #     with_ctc_loss = args.with_ctc_loss

        self.model = CTCBertModel(configs, args.ctc_path, args.bert, args.fix_bert, with_align_loss=args.with_align_loss, with_ctc_loss=args.with_ctc_loss, use_lstm_predictor=args.use_lstm_predictor, n_words=vocab_size, bert_insert_blank=args.bert_insert_blank)

        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = Dataset(self.args.train, self.args.char_dict, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer, frame_shift=self.args.frame_shift)
        train_loader = DataLoader(train_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        dev_set = Dataset(self.args.dev, self.args.char_dict, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer, frame_shift=self.args.frame_shift)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=4
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")
        
        num_epochs = self.configs.get('max_epoch', 50)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 3.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        
        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            sum_ctc_loss = 0
            sum_align_loss = 0
            sum_false = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                bert_input = dic['bert_input']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    bert_input = bert_input.to(torch.device("cuda"))
                
                bert_out_lens = bert_input.squeeze(-1).ne(self.bert_tokenizer.pad_token_id).sum(-1)
                encoder_out, encoder_out_lens, bert_out = self.model(audio_feat, feats_lengths, asr_target, target_lengths, bert_input, bert_out_lens)
                # choose_mask = self.model.align(encoder_out, encoder_out_lens, bert_out, bert_out_lens)
                # import pdb
                # pdb.set_trace()
                # sum_false += (encoder_out_lens<target_lengths).sum().item()
                loss, loss_ctc, loss_align = self.model.loss(encoder_out, encoder_out_lens, bert_out, asr_target, target_lengths, bert_out_lens, epoch)
                sum_loss += loss.item()
                if self.args.with_ctc_loss:
                    sum_ctc_loss += loss_ctc.item()
                else:
                    sum_ctc_loss += loss_ctc
                if self.args.with_align_loss:
                    sum_align_loss += loss_align.item()
                else:
                    sum_align_loss += loss_align
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if self.args.with_ctc_loss:
                        log_str += 'ctc loss {:.6f} '.format(loss_ctc.item())
                    else:
                        log_str += 'ctc loss {:.6f} '.format(loss_ctc)
                    if self.args.with_align_loss:
                        log_str += 'align loss {:.6f} '.format(loss_align.item())
                    else:
                        log_str += 'align loss {:.6f} '.format(loss_align)
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f} ctc loss: {(sum_ctc_loss/i):.6f} align loss: {(sum_align_loss/i):.6f} \n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()
            if epoch - best_epoch >= 10:
                break
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")
            
    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        sum_ctc_loss = 0
        sum_align_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            bert_input = dic['bert_input']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
                bert_input = bert_input.to(torch.device("cuda"))

            bert_out_lens = bert_input.squeeze(-1).ne(self.bert_tokenizer.pad_token_id).sum(-1)
            encoder_out, encoder_out_lens, bert_out = self.model(audio_feat, feats_lengths, asr_target, target_lengths, bert_input, bert_out_lens)
            loss, loss_ctc, loss_align = self.model.loss(encoder_out, encoder_out_lens, bert_out, asr_target, target_lengths, bert_out_lens)
            sum_dev_loss += loss.item()
            if self.args.with_ctc_loss:
                sum_ctc_loss += loss_ctc.item()
            else:
                sum_ctc_loss += loss_ctc
            if self.args.with_align_loss:
                sum_align_loss += loss_align.item()
            else:
                sum_align_loss += loss_align

        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}\n')
        logger.info(f'average dev ctc loss: {(sum_ctc_loss/i):.6f}\n')
        logger.info(f'average dev align loss: {(sum_align_loss/i):.6f}\n')
        return sum_dev_loss/i

    # @torch.no_grad()
    def eval_asr(self):
        self.model.eval()
        init_logger(logger)
        dataset = Dataset(self.args.input, self.args.char_dict, bert_tokenizer=self.bert_tokenizer,use_same_tokenizer=self.args.use_same_tokenizer, frame_shift=self.args.frame_shift)
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")
        
        if not self.args.use_same_tokenizer:
            char_dict = {v:k for k, v in dataset.char_dict.items()}
            eos = len(char_dict) - 1
        else:
            char_dict = {v:k for k, v in self.bert_tokenizer.get_vocab().items()}
            eos = 102

        bar = progress_bar(dataloader)
        with open(self.args.res, 'w') as fout:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                bert_input = dic['bert_input']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    bert_input = bert_input.to(torch.device("cuda"))

                bert_out_lens = bert_input.squeeze(-1).ne(self.bert_tokenizer.pad_token_id).sum(-1)
                encoder_out, encoder_out_lens, bert_out = self.model(audio_feat, feats_lengths, asr_target, target_lengths, bert_input, bert_out_lens)
                choose_mask = self.model.align(encoder_out, encoder_out_lens, bert_out, bert_out_lens)
                
                hyps, _ = self.model.ctc_model.ctc_greedy_search(
                    audio_feat,
                    feats_lengths,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    simulate_streaming=False)
                
                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    logger.info('{} {}'.format(key, ''.join(content)))
                    fout.write('{} {}\n'.format(key, ''.join(content)))
                import pdb
                pdb.set_trace()

class CTCBertNERParser(object):
    def __init__(self, args) -> None:
        self.args = args
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
        
        if self.args.use_same_tokenizer:
            vocab_size = len(self.bert_tokenizer.get_vocab())
            configs['blank_id'] = 1
            configs['sos'] = 101
            configs['eos'] = 102
        else:
            vocab_size = len(read_symbol_table(args.char_dict))
            configs['blank_id'] = 0
            configs['sos'] = None
            configs['eos'] = None

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        self.configs = configs

        # if not hasattr(args, 'fix_bert'):
        #     fix_bert = True
        # else:
        #     fix_bert = args.fix_bert
        
        # if not hasattr(args, 'with_ctc_loss'):
        #     with_ctc_loss = False
        # else:
        #     with_ctc_loss = args.with_ctc_loss
        
        train_set = NERDataset(self.args.train, self.args.char_dict, self.args.if_flat, if_build_ner_vocab=True)
        self.ner_vocab = train_set.ner_vocab
        
        self.model = CTCBertNERModel(len(self.ner_vocab), configs, args.ctc_path, args.bert, args.ctc_bert_path, if_fix_bert=args.fix_bert, with_ctc_loss=args.with_ctc_loss, with_align_loss=args.with_align_loss, use_speech=not args.text_only, use_tokenized=args.use_tokenized)

        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = NERDataset(self.args.train, self.args.char_dict, self.args.if_flat, if_build_ner_vocab=False, ner_vocab=self.ner_vocab, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer)
        train_loader = DataLoader(train_set, 
                                collate_fn=ner_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        
        dev_set = NERDataset(self.args.dev, self.args.char_dict, self.args.if_flat, if_build_ner_vocab=False, ner_vocab=self.ner_vocab, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=ner_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )

        test_set = NERDataset(self.args.test, self.args.char_dict, self.args.if_flat, if_build_ner_vocab=False, ner_vocab=self.ner_vocab, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer)
        test_loader = DataLoader(test_set, 
                                collate_fn=ner_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(test_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n test:{len(test_set):6}\n")

        steps = (len(train_set)//self.args.batch_size) * self.args.epochs // self.args.update_steps
        self.optimizer = AdamW(
                [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                 for n, c in self.model.named_parameters()],
                self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*self.args.warmup), steps)

        elapsed = timedelta()
        best_e, best_metric, min_loss = 1, ChartMetric(), 100000
        best_model_path, curr_model_path = os.path.join(self.args.path, 'best.model'), os.path.join(self.args.path, 'current.model')

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            sum_loss = 0
            bar = progress_bar(train_loader)
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                bert_input = dic['bert_input']
                ner_labels = dic['ner']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    bert_input = bert_input.to(torch.device("cuda"))
                    ner_labels = ner_labels.to(torch.device("cuda"))
                
                score, ctc_loss = self.model(audio_feat, feats_lengths, asr_target, target_lengths, bert_input)
                # [batch_size, t_len]
                pad_mask = bert_input.ne(0).any(-1)
                # [batch_size, t_len, t_len]
                char_pad_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
                lower_triangle_mask = torch.ones_like(char_pad_mask).bool()
                lower_triangle_mask = ~lower_triangle_mask.triu(1)
                loss = self.model.loss(score, ner_labels, char_pad_mask&lower_triangle_mask)
                loss = loss + 0.05 * ctc_loss
                sum_loss += loss.item()
                loss = loss / self.args.update_steps
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                if i % self.args.update_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )
            logger.info(f"{bar.postfix}")
            logger.info(f"avg train loss: {(sum_loss/i):.6f}")
        
            logger.info(f"dev: ")
            dev_metric = self.eval(dev_loader)

            logger.info(f"test: ")
            test_metric = self.eval(test_loader)
            t = datetime.now() - start
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), curr_model_path)
            torch.cuda.empty_cache()

            if epoch - best_e >= 10:
                break

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def eval(self, dataloader=None):
        self.model.eval()
        init_logger(logger)

        if dataloader is None:
            data_set = NERDataset(self.args.input, self.args.char_dict, self.args.if_flat, if_build_ner_vocab=False, ner_vocab=self.ner_vocab, bert_tokenizer=self.bert_tokenizer, use_same_tokenizer=self.args.use_same_tokenizer)
            dataloader = DataLoader(data_set, 
                                    collate_fn=ner_collate_fn,
                                    batch_sampler=BatchSampler(SequentialSampler(data_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False),
                                    num_workers=self.args.num_workers
                                    )


        metric = ChartMetric()
        bar = progress_bar(dataloader)
        start = datetime.now()
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            keys = dic['keys']
            bert_input = dic['bert_input']
            ner_labels = dic['ner']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
                bert_input = bert_input.to(torch.device("cuda"))
                ner_labels = ner_labels.to(torch.device("cuda"))
            
            score, _ = self.model(audio_feat, feats_lengths, asr_target, target_lengths, bert_input)
            # [batch_size, t_len]
            pad_mask = bert_input.ne(0).any(-1)
            # [batch_size, t_len, t_len]
            char_pad_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
            lower_triangle_mask = torch.ones_like(char_pad_mask).bool()
            lower_triangle_mask = ~lower_triangle_mask.triu(1)

            if self.args.if_flat:
                pred = self.model.decode(score, lower_triangle_mask&char_pad_mask, pad_mask)
            else:
                # pred = self.model.nested_decode(score, lower_triangle_mask&char_pad_mask, pad_mask)
                pred = self.model.fast_nested_decode(score, lower_triangle_mask&char_pad_mask, pad_mask)
            # pred = pred.masked_fill(pred.eq(len(self.ner_vocab)), -1)
            # pred = pred.masked_fill(~(lower_triangle_mask&char_pad_mask), -1)
            ner_labels = ner_labels.masked_fill(ner_labels.eq(len(self.ner_vocab)), -1)
            ner_labels = ner_labels.masked_fill(~(lower_triangle_mask&char_pad_mask), -1)
            metric(pred, ner_labels)
        t = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(f"{t}s elapsed\n")
        return metric

class CTCAttentionASRParser(object):
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

        symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(symbol_vocab)

        # if args.add_context:
        #     self.context_vocab = read_context_table(args.context_vocab)
        #     self.context_tensor = build_context_tensor(self.context_vocab, symbol_vocab)


        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        self.configs = configs
        
        self.model = init_asr_model(configs)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
            # self.context_tensor = self.context_tensor.cuda()

        if self.args.add_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        else:
            self.bert_tokenizer = None

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        train_set = Dataset(self.args.train, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        train_loader = DataLoader(train_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        
        dev_set = Dataset(self.args.dev, self.args.char_dict, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        num_epochs = self.configs.get('max_epoch', 100)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))

                loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

            if epoch - best_epoch > self.configs.get('patience', 10):
                break
        
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")
    
    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
            loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
            sum_dev_loss += loss.item()
        
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        dataset = Dataset(self.args.input, self.args.char_dict, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num)
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")

        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']']
        bar = progress_bar(dataloader)
        all_res_file = self.args.res + '.all'
        asr_res_file = self.args.res + '.asr'
        with open(all_res_file, 'w') as fout, open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                
                if self.args.decode_mode == 'attention':
                    hyps, _ = self.model.recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif self.args.decode_mode == 'ctc_greedy_search':
                    hyps, _ = self.model.ctc_greedy_search(
                    audio_feat,
                    feats_lengths,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    simulate_streaming=False)
                else:
                    raise NotImplementedError

                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    all_res = ''.join(content)
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')
                    logger.info('{}   {}   {}'.format(key, all_res, asr_res))
                    fout.write('{} {}\n'.format(key, all_res))
                    fasr.write('{} {}\n'.format(key, asr_res))

class CLASCTCAttentionASRParser(object):
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

        self.symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(self.symbol_vocab)

        # self.dev_con_vocab = read_context_table(args.dev_ne_dict)
        # self.dev_context_tensor = build_context_tensor(self.dev_con_vocab, self.symbol_vocab, pad_value=-1)

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        configs['symbol_table'] = self.symbol_vocab
        configs['att_type'] = args.att_type
        configs['raw_cba'] = args.raw_cba
        self.configs = configs
        
        self.model = init_asr_model(configs)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
            # self.dev_context_tensor = self.dev_context_tensor.cuda()

        if self.args.add_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        else:
            self.bert_tokenizer = None

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

        self.test_context_vocab = read_context_table(self.args.test_ne_dict)
        self.test_context_tensor = build_context_tensor(self.test_context_vocab, self.symbol_vocab, pad_value=-1)
        if use_cuda:
            self.test_context_tensor = self.test_context_tensor.cuda()


    def train(self):
        train_set = CLASDataset(self.args.train, self.args.char_dict, self.args.train_ne_dict, is_training=True, is_dev=False, is_test=False, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        train_loader = DataLoader(train_set, 
                                collate_fn=clas_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        # TODO: 这里我把dev的context也像train那样搞
        dev_set = CLASDataset(self.args.dev, self.args.char_dict, self.args.dev_ne_dict, is_training=True, is_dev=False, is_test=False, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=clas_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        num_epochs = self.configs.get('max_epoch', 100)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                # need_att_mask = dic['need_att_mask']
                context_tensor = dic['context_tensor']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    # need_att_mask = need_att_mask.to(torch.device("cuda"))
                    context_tensor = context_tensor.to(torch.device("cuda"))

                # here context_tensor do not contain 'none', cocoder should contain it
                context = self.model.concoder(context_tensor)
                loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

            if epoch - best_epoch > self.configs.get('patience', 10):
                break
        
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")
    
    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)

        # context = self.model.concoder(self.dev_context_tensor)

        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            # need_att_mask = dic['need_att_mask']
            context_tensor = dic['context_tensor']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
                context_tensor = context_tensor.to(torch.device("cuda"))
                # need_att_mask = need_att_mask.to(torch.device("cuda"))
            context = self.model.concoder(context_tensor)
            loss, loss_att, loss_ctc = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context)
            sum_dev_loss += loss.item()
        
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        is_train = False
        is_test = True
        dataset = CLASDataset(self.args.input, self.args.char_dict, self.args.test_ne_dict, is_training=is_train, is_dev=False, is_test=is_test, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dataloader = DataLoader(dataset,
                                collate_fn=clas_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")

        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']', '$']
        

        # context = self.model.concoder(self.test_context_tensor)
        # context = context[-1].unsqueeze(0)

        bar = progress_bar(dataloader)
        all_res_file = self.args.res + '.all'
        asr_res_file = self.args.res + '.asr'
        if is_test:
            context = self.model.concoder(self.test_context_tensor)
        with open(all_res_file, 'w') as fout, open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if is_train:
                    context_tensor = dic['context_tensor']
                # context_tensor = dic['context_tensor']
                # need_att_mask = dic['need_att_mask']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    if is_train:
                        context_tensor = context_tensor.to(torch.device("cuda"))
                    # need_att_mask = need_att_mask.to(torch.device("cuda"))
                if is_train:
                    context = self.model.concoder(context_tensor)

                if self.args.decode_mode == 'attention':
                    hyps, _ = self.model.recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif self.args.decode_mode == 'ctc_greedy_search':
                    hyps, _ = self.model.ctc_greedy_search(
                    audio_feat,
                    feats_lengths,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    simulate_streaming=False)
                else:
                    raise NotImplementedError

                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    all_res = ''.join(content)
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')

                    logger.info('{}   {}   {}'.format(key, all_res, asr_res))
                    fout.write('{} {}\n'.format(key, all_res))
                    fasr.write('{} {}\n'.format(key, asr_res))

class CopyNEASRParser(object):
    """
    use simpleattention, add attention loss
    """
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

        self.symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(self.symbol_vocab)

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        configs['symbol_table'] = self.symbol_vocab
        configs['att_type'] = args.att_type
        configs['add_copy_loss'] = args.add_copy_loss
        configs['no_concat'] = args.no_concat

        self.epochs = configs["max_epoch"]
        self.grad_clip = configs["grad_clip"]
        self.accum_grad = configs["accum_grad"]
        self.log_interval = configs["log_interval"]
        self.keep_nbest_models = configs["keep_nbest_models"]
        self.lr = configs["optim_conf"]["lr"]
        self.warm_steps = configs["scheduler_conf"]["warmup_steps"]
        self.configs = configs

        self.model = init_asr_model(configs)
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = self.model.to(self.gpu_id)

        if self.args.add_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        else:
            self.bert_tokenizer = None

        if self.args.mode == "train":
            init_logger(logger)

            train_set = CLASDataset(self.args.train, self.args.char_dict, self.args.train_ne_dict, is_training=True, is_dev=False, is_test=False, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
            self.train_loader = DataLoader(train_set, 
                                    batch_size=self.args.batch_size,
                                    sampler=DistributedSampler(train_set),
                                    collate_fn=copyne_collate_fn,
                                    num_workers=self.args.num_workers
                                    )
            logger.info(f"\n Rank:{self.gpu_id} train: {len(train_set):6}\n")

            steps = (len(train_set)//self.args.batch_size) * self.epochs // self.accum_grad
            self.optimizer = AdamW(
                    [{'params': c, 'lr': self.lr} for n, c in self.model.named_parameters()], self.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_steps, steps)

            self.start_epoch = 1
            if self.gpu_id == 0:
                self.best_epoch, self.best_dev_loss = 1, 100000
                self.best_model_path, self.curr_model_path = os.path.join(self.args.path, 'best.model'), os.path.join(self.args.path, 'current.model')
                if self.keep_nbest_models > 1:
                    # the best loss is not saved here
                    self.nbest_loss_lst = [self.best_dev_loss] * (self.keep_nbest_models - 1)
                    self.nbest_epoch_lst = [self.best_epoch] * (self.keep_nbest_models - 1)
                else:
                    self.nbest_loss_lst = None
                    self.nbest_epoch_lst = None
            
            # continue training
            if self.args.pre_model != "None":
                loc = f"cuda:{self.gpu_id}"
                checkpoint = torch.load(self.args.pre_model, map_location=loc)
                self.start_epoch = checkpoint['epoch'] + 1
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'best_dev_loss' in checkpoint and self.gpu_id == 0:
                    self.best_dev_loss = checkpoint['best_dev_loss']
                if 'best_epoch' in checkpoint and self.gpu_id == 0:
                    self.best_epoch = checkpoint['best_epoch']
                else:
                    self.best_epoch = self.start_epoch

                if "nbest_loss_lst" in checkpoint and self.gpu_id == 0:
                    self.nbest_loss_lst = checkpoint["nbest_loss_lst"]

                if "nbest_epoch_lst" in checkpoint and self.gpu_id == 0:
                    self.nbest_epoch_lst = checkpoint["nbest_epoch_lst"]

                if self.gpu_id == 0:
                    logger.info(f"Rank:{self.gpu_id} loading previous model from {self.args.pre_model}, start epoch {self.start_epoch}, best epoch {self.best_epoch}, best dev loss {self.best_dev_loss}")
                else:
                    logger.info(f"Rank:{self.gpu_id} loading previous model from {self.args.pre_model}, start epoch {self.start_epoch}")
                del checkpoint
                torch.cuda.empty_cache()
        elif self.args.mode == "evaluate":
            if self.args.use_avg:
                path = os.path.join(self.args.path, 'avg.model')
            else:
                path = os.path.join(self.args.path, 'best.model')
            self.load_model(path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def load_model(self, model_path):
        assert model_path != 'None'
        loc = f"cuda:{self.gpu_id}"
        logger.info(f"Rank:{self.gpu_id} loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=loc)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        if self.args.test_ne_dict != "None":
            self.test_context_vocab = read_context_table(self.args.test_ne_dict)
            self.test_context_tensor = build_context_tensor(self.test_context_vocab, self.symbol_vocab, pad_value=-1)
            self.test_context_tensor = self.test_context_tensor.to(loc)

    def train(self):
        if self.gpu_id == 0:
            # TODO: 这里我把dev的context也像train那样搞
            dev_set = CLASDataset(self.args.dev, self.args.char_dict, self.args.dev_ne_dict, is_training=True, is_dev=False, is_test=False, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
            dev_loader = DataLoader(dev_set, 
                                    collate_fn=copyne_collate_fn,
                                    batch_sampler=BatchSampler(RandomSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False),
                                    num_workers=self.args.num_workers
                                    )
            logger.info(f"\n dev:{len(dev_set):6}\n")

        elapsed = timedelta()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            start = datetime.now()
            logger.info(f"Rank:{self.gpu_id} Epoch {epoch} / {self.epochs}:")
            sum_loss = 0
            bar = progress_bar(self.train_loader)
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                context_tensor = dic['context_tensor']
                att_tgt = dic['att_tgt']
                # if self.args.device != '-1':
                audio_feat = audio_feat.to(self.gpu_id)
                asr_target = asr_target.to(self.gpu_id)
                feats_lengths = feats_lengths.to(self.gpu_id)
                target_lengths = target_lengths.to(self.gpu_id)
                context_tensor = context_tensor.to(self.gpu_id)
                att_tgt = att_tgt.to(self.gpu_id)

                # here context_tensor do not contain 'none', cocoder should contain it, and append none at last
                context = self.model.module.concoder(context_tensor)
                loss, loss_att, loss_ctc, loss_copy = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context=context, att_tgt=att_tgt)
                sum_loss += loss.item()
                loss = loss / self.accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if i % self.accum_grad == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if i % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * self.accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    if loss_copy is not None:
                        log_str += 'loss_copy {:.6f} '.format(loss_copy.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(f"Rank:{self.gpu_id} {log_str}")
            logger.info(f'Rank:{self.gpu_id} average train loss: {(sum_loss/i):.6f}\n')

            if self.gpu_id == 0:
                # evaluate on dev dataset
                this_dev_loss = self.loss_on_dev(dev_loader)
                t = datetime.now() - start
                if this_dev_loss < self.best_dev_loss:
                    self.best_dev_loss = this_dev_loss
                    self.best_epoch = epoch
                    logger.info(f"Rank:{self.gpu_id} {t}s elapsed (saved)\n")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_dev_loss': self.best_dev_loss,
                        'best_epoch': self.best_epoch,
                        'nbest_loss_lst': self.nbest_loss_lst,
                        'nbest_epoch_lst': self.nbest_epoch_lst,
                        }, self.best_model_path)
                else:
                    if self.nbest_loss_lst is not None and this_dev_loss < self.nbest_loss_lst[-1]:
                        # find the place to insert
                        idx = 0
                        for idx in range(self.keep_nbest_models - 1):
                            if this_dev_loss < self.nbest_loss_lst[idx]:
                                break
                        self.nbest_loss_lst.insert(idx, this_dev_loss)
                        self.nbest_epoch_lst.insert(idx, epoch)
                        to_del_loss = self.nbest_loss_lst.pop()
                        to_del_epoch = self.nbest_epoch_lst.pop()
                        save_model_path = os.path.join(self.args.path, f'nbest_epo_{epoch}.model')
                        del_model_path = os.path.join(self.args.path, f'nbest_epo_{to_del_epoch}.model')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_dev_loss': this_dev_loss,
                            'best_epoch': epoch,
                            'nbest_loss_lst': self.nbest_loss_lst,
                            'nbest_epoch_lst': self.nbest_epoch_lst,
                            }, save_model_path)
                        if os.path.exists(del_model_path):
                            os.remove(del_model_path)
                        logger.info(f"Rank:{self.gpu_id}: Epoch:{epoch} rankindex {idx} saved, Epoch:{to_del_epoch} deleted.\n")
                    logger.info(f"Rank:{self.gpu_id} {t}s elapsed\n")

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_dev_loss': self.best_dev_loss,
                        'best_epoch': self.best_epoch,
                        'nbest_loss_lst': self.nbest_loss_lst,
                        'nbest_epoch_lst': self.nbest_epoch_lst,
                        }, self.curr_model_path)
            
            torch.cuda.empty_cache()
        
        if self.gpu_id == 0:
            logger.info(f"Epoch {self.best_epoch} saved")
            logger.info(f"Best Average Dev loss: {self.best_dev_loss:.6f}")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            context_tensor = dic['context_tensor']
            att_tgt = dic['att_tgt']
            # if self.args.device != '-1':
            audio_feat = audio_feat.to(self.gpu_id)
            asr_target = asr_target.to(self.gpu_id)
            feats_lengths = feats_lengths.to(self.gpu_id)
            target_lengths = target_lengths.to(self.gpu_id)
            context_tensor = context_tensor.to(self.gpu_id)
            # need_att_mask = need_att_mask.to(torch.device("cuda"))
            att_tgt = att_tgt.to(self.gpu_id)
            context = self.model.module.concoder(context_tensor)
            loss, loss_att, loss_ctc, loss_copy = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context=context, att_tgt=att_tgt)
            sum_dev_loss += loss.item()

        logger.info(f'Rank:{self.gpu_id} average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)

        is_train = False
        is_dev = False
        is_test = True

        dataset = CLASDataset(self.args.input, self.args.char_dict, self.args.test_ne_dict, is_training=is_train, is_dev=is_dev, is_test=is_test, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dataloader = DataLoader(dataset,
                                collate_fn=copyne_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")

        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']', '$']
        bar = progress_bar(dataloader)
        # all_res_file = self.args.res + '.all'
        asr_res_file = self.args.res + '.asr'
        if is_test:
            context = self.model.module.concoder(self.test_context_tensor)
        start = datetime.now()
        with open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                # asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                # target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if is_train or is_dev:
                    context_tensor = dic['context_tensor']
                    att_tgt = dic['att_tgt']
                # need_att_mask = dic['need_att_mask']
                # if self.args.device != '-1':
                
                audio_feat = audio_feat.to(self.gpu_id)
                # asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(self.gpu_id)
                # target_lengths = target_lengths.to(torch.device("cuda"))
                if is_train or is_dev:
                    context_tensor = context_tensor.to(self.gpu_id)
                    att_tgt = att_tgt.to(self.gpu_id)
                
                if is_train or is_dev:
                    context = self.model.concoder(context_tensor)
                else:
                    context_tensor = self.test_context_tensor

                if self.args.decode_mode == 'attention':
                    hyps, _ = self.model.module.recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context,
                    context_tensor=context_tensor)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif self.args.decode_mode == 'copy_attention':
                    hyps, _ = self.model.module.copy_recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context, 
                    context_tensor=context_tensor,
                    copy_threshold=self.args.copy_threshold)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif self.args.decode_mode == 'ctc_greedy_search':
                    hyps, _ = self.model.module.ctc_greedy_search(
                    audio_feat,
                    feats_lengths,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    simulate_streaming=False)
                else:
                    raise NotImplementedError

                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        if char_dict[w] == '<unk>':
                            content.append("淦")
                        else:
                            content.append(char_dict[w])
                    # all_res = ''.join(content)
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')

                    logger.info('{}\t{}'.format(key, asr_res))
                    # fout.write('{} {}\n'.format(key, all_res))
                    fasr.write('{} {}\n'.format(key, asr_res))
        elapsed = datetime.now() - start
        print(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

    @torch.no_grad()
    def api(self, audio_file, ne_vocab_file, copy_threshold=0.9):
        self.model.eval()
        data_file_path = process_audio(audio_file, segment_length_sec=10, tgt_sample_rate=16000, tmp_dir="tmp_dir")

        is_train = False
        is_dev = False
        is_test = True

        if ne_vocab_file != "None":
            self.test_context_vocab = read_context_table(ne_vocab_file)
            self.test_context_tensor = build_context_tensor(ne_vocab_file, self.symbol_vocab, pad_value=-1)
            loc = f"cuda:{self.gpu_id}"
            self.test_context_tensor = self.test_context_tensor.to(loc)

        dataset = CLASDataset(data_file_path, self.args.char_dict, self.args.test_ne_dict, is_training=is_train, is_dev=is_dev, is_test=is_test,  frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dataloader = DataLoader(dataset,
                                collate_fn=copyne_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=16,
                                drop_last=False),
                                num_workers=3
        )
        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        bar = progress_bar(dataloader)

        all_res_s = ""
        if is_test:
            context = self.model.module.concoder(self.test_context_tensor)
        start = datetime.now()
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            feats_lengths = dic['audio_feat_length']
            keys = dic['keys']
            audio_feat = audio_feat.to(self.gpu_id)
            feats_lengths = feats_lengths.to(self.gpu_id)
            context_tensor = self.test_context_tensor

            hyps, _ = self.model.module.copy_recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context, 
                    context_tensor=context_tensor,
                    copy_threshold=copy_threshold)
            hyps = [hyp.tolist() for hyp in hyps]

            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    if char_dict[w] == '<unk>':
                        content.append("<unk>")
                    else:
                        content.append(char_dict[w])
                asr_res = ''.join(content)
                all_res_s += asr_res

        elapsed = datetime.now() - start
        print(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")
        os.removedirs("tmp_dir")
        return all_res_s


class CopyASRPaser(object):
    """_summary_

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
        
        self.symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(self.symbol_vocab)

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        configs['symbol_table'] = self.symbol_vocab
        configs['dot_dim'] = args.dot_dim
        self.configs = configs

        self.model = init_copyasr_model(configs)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        if self.args.add_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert)
        else:
            self.bert_tokenizer = None

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

        self.test_context_vocab = read_context_table(self.args.test_ne_dict)
        self.test_context_tensor = build_context_tensor(self.test_context_vocab, self.symbol_vocab, pad_value=-1)
        if use_cuda:
            self.test_context_tensor = self.test_context_tensor.cuda()

    def train(self):
        train_set = CopyASRDataset(self.args.train, self.args.char_dict, self.args.train_ne_dict, is_training=True, is_dev=False, is_test=False, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        train_loader = DataLoader(train_set, 
                                collate_fn=copyasr_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        dev_set = CopyASRDataset(self.args.dev, self.args.char_dict, self.args.dev_ne_dict, is_training=True, is_dev=False, is_test=False, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=copyasr_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        num_epochs = self.configs.get('max_epoch', 100)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        
        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                context_tensor = dic['context_tensor']
                info_tgt = dic['info_tgt']
                info_mask = dic['info_mask']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    context_tensor = context_tensor.to(torch.device("cuda"))
                    info_tgt = info_tgt.to(torch.device("cuda"))
                    info_mask = info_mask.to(torch.device("cuda"))
                # should not cover null 
                context = self.model.concoder(context_tensor)
                loss, loss_att, loss_ctc, loss_infonce = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context, info_tgt, info_mask)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    if loss_infonce is not None:
                        log_str += 'loss_infonce {:.6f} '.format(loss_infonce.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

            if epoch - best_epoch > self.configs.get('patience', 10):
                break
        
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            context_tensor = dic['context_tensor']
            info_tgt = dic['info_tgt']
            info_mask = dic['info_mask']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
                context_tensor = context_tensor.to(torch.device("cuda"))
                info_tgt = info_tgt.to(torch.device("cuda"))
                info_mask = info_mask.to(torch.device("cuda"))
            context = self.model.concoder(context_tensor)
            loss, loss_att, loss_ctc, loss_infonce = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, context, info_tgt, info_mask)
            sum_dev_loss += loss.item()
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        is_train = False
        is_test = True
        dataset = CopyASRDataset(self.args.input, self.args.char_dict, self.args.test_ne_dict, is_training=is_train, is_dev=False, is_test=is_test, bert_tokenizer=self.bert_tokenizer, e2ener=self.args.e2ener, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_context=True, pad_context=self.args.pad_context)
        dataloader = DataLoader(dataset,
                                collate_fn=copyasr_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
        )
        logger.info(f"\n dataset: {len(dataset):6}\n")

        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']', '$']
        bar = progress_bar(dataloader)
        all_res_file = self.args.res + '.all'
        asr_res_file = self.args.res + '.asr'
        if is_test:
            context = self.model.concoder(self.test_context_tensor)
        with open(all_res_file, 'w') as fout, open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if is_train:
                    context_tensor = dic['context_tensor']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                    if is_train:
                        context_tensor = context_tensor.to(torch.device("cuda"))
                
                if is_train:
                    context = self.model.concoder(context_tensor)
                else:
                    context_tensor = self.test_context_tensor
                
                if self.args.decode_mode == 'attention':
                    hyps, _ = self.model.recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context,
                    context_tensor=context_tensor)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif self.args.decode_mode == 'copy_attention':
                    hyps, _ = self.model.copy_recognize(
                    audio_feat,
                    feats_lengths,
                    beam_size=self.args.beam_size,
                    context=context, 
                    context_tensor=context_tensor)
                    hyps = [hyp.tolist() for hyp in hyps]
                
                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        if char_dict[w] == '<unk>':
                            content.append("淦")
                        else:
                            content.append(char_dict[w])
                    all_res = ''.join(content)
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')

                    logger.info('{}   {}   {}'.format(key, all_res, asr_res))
                    fout.write('{} {}\n'.format(key, all_res))
                    fasr.write('{} {}\n'.format(key, asr_res))

class BartSeq2SeqParser(object):
    def __init__(self, args) -> None:
        self.args = args
        self.bart_tokenizer = BertTokenizer.from_pretrained(self.args.bart)
        self.model = BartForEnd2EndSpeechNER.from_pretrained(self.args.bart)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        
        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
            self.configs = configs
    
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = BartSeq2SeqDataset(self.args.train, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        train_loader = DataLoader(train_set,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        dev_set = BartSeq2SeqDataset(self.args.dev, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        dev_loader = DataLoader(dev_set,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=4)
        
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        accum_grad = self.configs.get('accum_grad', 2)
        if self.args.first_step:
            num_epochs = self.configs.get('first_max_epoch', 80)
            steps = len(train_loader) * num_epochs // accum_grad
            update_group, name_lst = self.model.getparamgroup_in_first_step()
            optimizer = optim.Adam(params=[{"params": update_group, "lr": self.configs['first_optim_conf']["lr"]}])
            scheduler = PolynomialLR(optimizer, steps=steps, **self.configs['first_scheduler_conf'])
        else:
            num_epochs = self.configs.get('second_max_epoch', 20)
            steps = len(train_loader) * num_epochs // accum_grad
            optimizer = optim.Adam(self.model.parameters(), **self.configs['second_optim_conf'])
            scheduler = PolynomialLR(optimizer, steps=steps, **self.configs['second_scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                feats_lengths = dic['audio_feat_length']
                keys = dic['keys']
                # labels = bart_input_ids
                bart_input_ids = dic['bart_input'].squeeze(-1)
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    bart_input_ids = bart_input_ids.to(torch.device("cuda"))
                model_out = self.model(input_ids=bart_input_ids, labels=bart_input_ids, sp_encoder_inputs=audio_feat, sp_lengths=feats_lengths)
                loss = model_out.loss
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

            if epoch - best_epoch > self.configs.get('patience', 10):
                break

        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            feats_lengths = dic['audio_feat_length']
            keys = dic['keys']
            # labels = bart_input_ids
            bart_input_ids = dic['bart_input'].squeeze(-1)
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                bart_input_ids = bart_input_ids.to(torch.device("cuda"))
            model_out = self.model(input_ids=bart_input_ids, labels=bart_input_ids, sp_encoder_inputs=audio_feat, sp_lengths=feats_lengths)
            loss = model_out.loss
            sum_dev_loss += loss.item()
        
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        
        # TODO:
        bos_token_id = 101

        self.model.eval()
        init_logger(logger)
        dataset = BartSeq2SeqDataset(self.args.input, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        dataloader = DataLoader(dataset,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers)
        logger.info(f"\n dataset: {len(dataset):6}\n")

        bar = progress_bar(dataloader)
        with open(self.args.res, 'w') as fout:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                feats_lengths = dic['audio_feat_length']
                keys = dic['keys']
                # labels = bart_input_ids
                labels = dic['bart_input'].squeeze(-1)
                batch_size = audio_feat.size(0)
                input_ids = torch.ones((batch_size, 1), dtype=torch.long) * bos_token_id
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))
                    input_ids = input_ids.to(torch.device("cuda"))
                
                pred_ids = self.model.generate(input_ids=input_ids, sp_encoder_inputs=audio_feat, sp_lengths=feats_lengths, do_sample=False, num_beams=3)
                
class BartSpeechParser(object):
    def __init__(self, args) -> None:
        self.args = args
        self.bart_tokenizer = BertTokenizer.from_pretrained(self.args.bart_tokenizer)
        self.model = BartSpeechNER(self.args.bart_tokenizer, self.args.bart_asrcorr, self.args.config, add_ctc=self.args.add_ctc, device=args.device)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
            self.configs = configs
        
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = BartSeq2SeqDataset(self.args.train, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num, spec_aug=True, speed_perturb=True)
        train_loader = DataLoader(train_set,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )

        dev_set = BartSeq2SeqDataset(self.args.dev, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        dev_loader = DataLoader(dev_set,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=4)

        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        accum_grad = self.configs.get('accum_grad', 2)
        
        if self.args.bart_asrcorr != "None":
            if self.args.first_step:
                num_epochs = self.configs.get('first_max_epoch', 100)
                steps = len(train_loader) * num_epochs // accum_grad
                update_group, name_lst = self.model.getparamgroup_in_first_step()
                optimizer = optim.Adam(params=[{"params": update_group, "lr": self.configs['first_optim_conf']["lr"]}])
                # scheduler = WarmupLR(optimizer, **self.configs['first_scheduler_conf'])
                scheduler = WarmupLR(optimizer, **self.configs['first_scheduler_conf'])
            else:
                num_epochs = self.configs.get('second_max_epoch', 100)
                steps = len(train_loader) * num_epochs // accum_grad
                # sp_param_group, sp_name, inter_param_group, inter_name, other_param_group, other_name = self.model.getparamgroup_in_first_step(if_second_stage=True)
                # sp_param_group, sp_name, other_param_group, other_name = self.model.getparamgroup_in_first_step(if_second_stage=True)
                # optimizer = optim.Adam(params=[{"params": sp_param_group, "lr": self.configs['second_optim_conf']["sp_lr"]},
                #                                 {"params": other_param_group, "lr": self.configs['second_optim_conf']["other_lr"]}
                #                                 ])
                update_group = [p for p in self.model.parameters()]
                optimizer = optim.Adam(params=[{"params": update_group, "lr": self.configs['first_optim_conf']["lr"]}])
                # scheduler = PolynomialLR(optimizer, steps=steps, **self.configs['second_scheduler_conf'])
                scheduler = WarmupLR(optimizer, **self.configs['first_scheduler_conf'])
        else:
            num_epochs = self.configs.get('first_max_epoch', 100)
            steps = len(train_loader) * num_epochs // accum_grad
            update_group = [p for p in self.model.parameters()]
            optimizer = optim.Adam(params=[{"params": update_group, "lr": self.configs['first_optim_conf']["lr"]}])
            # optimizer = optim.Adam(self.model.parameters(), **self.configs['first_optim_conf'])
            scheduler = WarmupLR(optimizer, **self.configs['first_scheduler_conf'])
            # scheduler = PolynomialLR(optimizer, steps=steps, **self.configs['first_scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            logger.info('Epoch {} TRAIN'.format(epoch))
            bar = progress_bar(train_loader)
            sum_loss = 0
            sum_att_loss = 0
            sum_ctc_loss = 0
            start = datetime.now()
            
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                feats_lengths = dic['audio_feat_length']
                keys = dic['keys']
                # labels = bart_input_ids
                bart_input_ids = dic['bart_input'].squeeze(-1)
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    bart_input_ids = bart_input_ids.to(torch.device("cuda"))
                x, attention_mask = self.model(audio_feat, feats_lengths)
                loss, att_loss, ctc_loss = self.model.loss(x, bart_input_ids, attention_mask, bart_input_ids.ne(self.model.config.pad_token_id))
                sum_loss += loss.item()
                sum_att_loss += att_loss.item()
                if ctc_loss is not None:
                    sum_ctc_loss += ctc_loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % log_interval == 0:
                    sp_lr = optimizer.param_groups[0]['lr']
                    # if not self.args.first_step:
                    #     inter_lr = optimizer.param_groups[1]['lr']
                    #     other_lr = optimizer.param_groups[2]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if att_loss is not None:
                        log_str += 'loss_att {:.6f} '.format(att_loss.item())
                    if ctc_loss is not None:
                        log_str += 'loss_ctc {:.6f} '.format(ctc_loss.item())
                    log_str += 'sp_lr: {:.8f}  '.format(sp_lr)
                    # if not self.args.first_step:
                    #     log_str += 'inter_lr: {:.8f}  '.format(inter_lr)
                    #     log_str += 'other_lr: {:.8f}'.format(other_lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f} average att loss: {(sum_att_loss/i):.6f} average ctc loss: {(sum_ctc_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            this_train_loss = sum_loss/i
            t = datetime.now() - start
            # if this_train_loss < best_dev_loss:
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

            if epoch - best_epoch > self.configs.get('patience', 10):
                break

        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        sum_att_loss = 0
        sum_ctc_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            feats_lengths = dic['audio_feat_length']
            keys = dic['keys']
            # labels = bart_input_ids
            bart_input_ids = dic['bart_input'].squeeze(-1)
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                bart_input_ids = bart_input_ids.to(torch.device("cuda"))
            x, attention_mask = self.model(audio_feat, feats_lengths)
            loss, att_loss, ctc_loss = self.model.loss(x, bart_input_ids, attention_mask, bart_input_ids.ne(self.model.config.pad_token_id))
            sum_dev_loss += loss.item()
            sum_att_loss += att_loss.item()
            if ctc_loss is not None:
                sum_ctc_loss += ctc_loss.item()

        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f} average att loss: {(sum_att_loss/i):.6f} average ctc loss: {(sum_ctc_loss/i):.6f}')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        dataset = BartSeq2SeqDataset(self.args.input, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, bart_tokenizer=self.bart_tokenizer, e2ener=self.args.e2ener, max_frame_num=self.args.max_frame_num)
        dataloader = DataLoader(dataset,
                                collate_fn=bartseq2seq_collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers)
        logger.info(f"\n dataset: {len(dataset):6}\n")

        bar = progress_bar(dataloader)
        id2str = {v:k for k, v in self.bart_tokenizer.vocab.items()}
        metric = AishellNerMetric()
        with open(self.args.res, 'w') as fout:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                feats_lengths = dic['audio_feat_length']
                keys = dic['keys']
                # labels = bart_input_ids
                labels = dic['bart_input'].squeeze(-1)
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))

                x, attention_mask = self.model(audio_feat, feats_lengths)
                preds = self.model.decode(x, attention_mask.bool(), beam_size=self.args.beam_size)
                preds = preds.squeeze(1)
                pred_lens = preds.ne(self.model.config.pad_token_id).sum(-1).tolist()

                res = []
                for sent_lst, pred_len in zip(preds.tolist(), pred_lens):
                    this_res = []
                    for i in range(pred_len):
                        if sent_lst[i] == self.model.config.bos_token_id or sent_lst[i] == self.model.config.eos_token_id:
                            continue
                        this_res.append(id2str[sent_lst[i]])
                    res.append(this_res)

                gold_lens = labels.ne(self.model.config.pad_token_id).sum(-1).tolist()
                gold_res = []
                for sent_lst, gold_len in zip(labels.tolist(), gold_lens):
                    this_res = []
                    for i in range(gold_len):
                        if sent_lst[i] == self.model.config.bos_token_id or sent_lst[i] == self.model.config.eos_token_id:
                            continue
                        this_res.append(id2str[sent_lst[i]])
                    gold_res.append(this_res)

                i = 0
                metric.e2ecall([''.join(pred_lst) for pred_lst in res], [''.join(gold_lst) for gold_lst in gold_res])
                for gold_lst, pred_lst in zip(gold_res, res):
                    fout.write(keys[i]+' ')
                    fout.write('gold_res:'+' ')
                    fout.write(''.join(gold_lst)+' ')
                    fout.write('pred_res:'+' ')
                    fout.write(''.join(pred_lst)+'\n')
                    i += 1
            print(metric)
            return metric
                
class BartASRCorrectionParser(object):
    def __init__(self, args) -> None:
        self.args = args
        self.bart_tokenizer = BertTokenizer.from_pretrained(self.args.bart_config)
        if self.args.distill:
            self.model = BartASRCorrection(bart_path="None", bart_tokenizer_path=self.args.bart_config)
            self.teacher_model = BartASRCorrection(self.args.bart)
            assert self.args.teach_model != 'None'
            self.teacher_model.load_state_dict(torch.load(self.args.teach_model))
        else:
            # use the pre-trained bart
            self.model = BartASRCorrection(self.args.bart)

        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
            if self.args.distill:
                self.teacher_model = self.teacher_model.to(torch.device("cuda"))

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
            self.configs = configs

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)

    def train(self):
        init_logger(logger)
        train_set = BartTxtSeq2SeqDataset(self.args.train, self.bart_tokenizer, e2ener=self.args.e2ener)
        train_loader = DataLoader(train_set,
                                collate_fn=barttxtseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        dev_set = BartTxtSeq2SeqDataset(self.args.dev, self.bart_tokenizer, e2ener=self.args.e2ener)
        dev_loader = DataLoader(dev_set,
                                collate_fn=barttxtseq2seq_collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=4)

        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        accum_grad = self.configs.get('accum_grad', 2)
        num_epochs = self.configs.get('max_epoch', 60)
        steps = len(train_loader) * num_epochs // accum_grad
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = PolynomialLR(optimizer, steps=steps, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        # best_dev_loss = 100000.0
        best_metric = Metric()
        best_epoch = 1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()

        if self.args.distill:
            self.teacher_model.eval()

        for epoch in range(1, num_epochs+1):
            self.model.train()
            logger.info('Epoch {} TRAIN'.format(epoch))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()

            for i, dic in enumerate(bar, 1):
                keys = dic['keys']
                src = dic['src']
                tgt = dic['tgt']
                if self.args.device != '-1':
                    src = src.to(torch.device("cuda"))
                    tgt = tgt.to(torch.device("cuda"))
                
                tea_loss = None
                if not self.args.distill:
                    x = self.model(src)
                    src_mask = src.ne(self.model.config.pad_token_id)
                    tgt_mask = tgt.ne(self.model.config.pad_token_id)
                    loss = self.model.loss(x, tgt, src_mask, tgt_mask)
                else:
                    src_mask = src.ne(self.model.config.pad_token_id)
                    tgt_mask = tgt.ne(self.model.config.pad_token_id)
                    tea_logits, tea_loss = self.teacher_model.get_logits(src, tgt, src_mask, tgt_mask)
                    stu_x = self.model(src)
                    loss = self.model.distill_loss(stu_x, tea_logits, tgt, src_mask, tgt_mask, self.args.tem, self.args.hard_weight)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if tea_loss is not None:
                        log_str += f'teacher loss {tea_loss:.6f} '
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            # this_dev_loss = self.loss_on_dev(dev_loader)
            dev_metric = self.eval(dev_loader)
            logger.info(f"dev: {dev_metric}")
            t = datetime.now() - start
            # if this_dev_loss < best_dev_loss:
            if dev_metric > best_metric:
                best_metric = dev_metric
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()
            if epoch - best_epoch >= self.configs.get('patience', 10):
                break
        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Dev Metric: {best_metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            keys = dic['keys']
            src = dic['src']
            tgt = dic['tgt']
            if self.args.device != '-1':
                src = src.to(torch.device("cuda"))
                tgt = tgt.to(torch.device("cuda"))

            x = self.model(src)
            src_mask = src.ne(self.model.config.pad_token_id)
            tgt_mask = tgt.ne(self.model.config.pad_token_id)
            loss = self.model.loss(x, tgt, src_mask, tgt_mask)
            sum_dev_loss += loss.item()

        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}\n')
        return sum_dev_loss/i

    @torch.no_grad()
    def eval(self, dataloader=None):
        self.model.eval()
        init_logger(logger)
        if dataloader is None:
            dataset = BartTxtSeq2SeqDataset(self.args.input, self.bart_tokenizer, self.args.e2ener)
            dataloader = DataLoader(dataset,
                                    collate_fn=barttxtseq2seq_collate_fn,
                                    batch_sampler=BatchSampler(SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False),
                                    num_workers=self.args.num_workers)
            logger.info(f"\n dataset: {len(dataset):6}\n")

        bar = progress_bar(dataloader)
        id2str = {v:k for k, v in self.bart_tokenizer.vocab.items()}
        metric = AishellNerMetric()
        with open(self.args.res, 'w') as fout:
            for k, dic in enumerate(bar, 1):
                keys = dic['keys']
                src = dic['src']
                tgt = dic['tgt']
                if self.args.device != '-1':
                    src = src.to(torch.device("cuda"))
                    tgt = tgt.to(torch.device("cuda"))

                x = self.model(src)
                src_mask = src.ne(self.model.config.pad_token_id)
                preds = self.model.decode(x, src_mask, beam_size=self.args.beam_size)
                preds = preds.squeeze(1)
                pred_lens = preds.ne(self.model.config.pad_token_id).sum(-1).tolist()

                res = []
                for sent_lst, pred_len in zip(preds.tolist(), pred_lens):
                    this_res = []
                    for i in range(pred_len):
                        if sent_lst[i] == self.model.config.bos_token_id or sent_lst[i] == self.model.config.eos_token_id:
                            continue
                        this_res.append(id2str[sent_lst[i]])
                    res.append(this_res)

                gold_lens = tgt.ne(self.model.config.pad_token_id).sum(-1).tolist()
                gold_res = []
                for sent_lst, gold_len in zip(tgt.tolist(), gold_lens):
                    this_res = []
                    for i in range(gold_len):
                        if sent_lst[i] == self.model.config.bos_token_id or sent_lst[i] == self.model.config.eos_token_id:
                            continue
                        this_res.append(id2str[sent_lst[i]])
                    gold_res.append(this_res)

                i = 0
                metric.e2ecall([''.join(pred_lst) for pred_lst in res], [''.join(gold_lst) for gold_lst in gold_res])
                for gold_lst, pred_lst in zip(gold_res, res):
                    fout.write(keys[i]+' ')
                    fout.write('gold_res:'+' ')
                    fout.write(''.join(gold_lst)+' ')
                    fout.write('pred_res:'+' ')
                    fout.write(''.join(pred_lst)+'\n')
                    i += 1
            print(metric)
            return metric

class ParaformerASRParser(object):
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

        symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(symbol_vocab)

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        self.configs = configs

        self.model = init_paraformer_model(configs)
        if args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.path, 'best.model')))
        use_cuda = self.args.device != '-1' and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)
        
    def train(self):
        train_set = Dataset(self.args.train, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, max_frame_num=self.args.max_frame_num)
        train_loader = DataLoader(train_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(train_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        dev_set = Dataset(self.args.dev, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num)
        dev_loader = DataLoader(dev_set, 
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(RandomSampler(dev_set),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers
                                )
        logger.info(f"\n train: {len(train_set):6}\n dev:{len(dev_set):6}\n")

        num_epochs = self.configs.get('max_epoch', 100)
        optimizer = optim.Adam(self.model.parameters(), **self.configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **self.configs['scheduler_conf'])

        clip = self.configs.get('grad_clip', 5.0)
        log_interval = self.configs.get('log_interval', 10)
        accum_grad = self.configs.get('accum_grad', 2)
        logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        
        best_dev_loss = 100000.0
        best_epoch = -1
        best_model_path = os.path.join(self.args.path, 'best.model')
        current_model_path = os.path.join(self.args.path, 'current.model')
        elapsed = timedelta()
        for epoch in range(1, num_epochs+1):
            self.model.train()
            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            bar = progress_bar(train_loader)
            sum_loss = 0
            start = datetime.now()
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                
                loss, loss_att, loss_pre = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
                sum_loss += loss.item()
                loss = loss / accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if i % accum_grad == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if i % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_pre is not None:
                        log_str += 'loss_pre {:.6f} '.format(loss_pre.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(log_str)
            logger.info(f'average train loss: {(sum_loss/i):.6f}\n')

            # evaluate on dev dataset
            this_dev_loss = self.loss_on_dev(dev_loader)
            t = datetime.now() - start
            if this_dev_loss < best_dev_loss:
                best_dev_loss = this_dev_loss
                best_epoch = epoch
                logger.info(f"{t}s elapsed (saved)\n")
                torch.save(copy.deepcopy(self.model.state_dict()), best_model_path)
            else:
                logger.info(f"{t}s elapsed\n")
            torch.save(copy.deepcopy(self.model.state_dict()), current_model_path)
            torch.cuda.empty_cache()

        logger.info(f"Epoch {best_epoch} saved")
        logger.info(f"Best Average Dev loss: {best_dev_loss:.6f}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            if self.args.device != '-1':
                audio_feat = audio_feat.to(torch.device("cuda"))
                asr_target = asr_target.to(torch.device("cuda"))
                feats_lengths = feats_lengths.to(torch.device("cuda"))
                target_lengths = target_lengths.to(torch.device("cuda"))
            loss, loss_att, loss_pre = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths)
            sum_dev_loss += loss.item()
        
        logger.info(f'average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)
        dataset = Dataset(self.args.input, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num)
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers)
        logger.info(f"\n dataset: {len(dataset):6}\n")
        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']']
        bar = progress_bar(dataloader)
        asr_res_file = self.args.res + '.asr'
        with open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                if self.args.device != '-1':
                    audio_feat = audio_feat.to(torch.device("cuda"))
                    asr_target = asr_target.to(torch.device("cuda"))
                    feats_lengths = feats_lengths.to(torch.device("cuda"))
                    target_lengths = target_lengths.to(torch.device("cuda"))
                
                hyps, _ = self.model.recognize(audio_feat, feats_lengths)
                hyps = [hyp.tolist() for hyp in hyps]

                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')
                    logger.info('{}    {}'.format(key, asr_res))
                    fasr.write('{} {}\n'.format(key, asr_res))
                    fasr.flush()

class nParaformerASRParser(object):
    def __init__(self, args) -> None:
        self.args = args

        with open(args.config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

        symbol_vocab = read_symbol_table(args.char_dict)
        vocab_size = len(symbol_vocab)

        configs['input_dim'] = input_dim
        configs['output_dim'] = vocab_size
        configs['cmvn_file'] = args.cmvn
        configs['is_json_cmvn'] = True
        configs['decoder_conf']['add_ne_feat'] = args.add_ne_feat
        configs['model_conf']['add_ne_feat'] = args.add_ne_feat
        configs['model_conf']['O_idx'] = 0
        configs['decoder_conf']['add_bert_feat'] = args.add_bert_feat
        configs['decoder_conf']['bert_path'] = args.bert_path
        configs['model_conf']['add_bert_feat'] = args.add_bert_feat
        configs['model_conf']['e2e_ner'] = args.e2e_ner
        configs['decoder_conf']['e2e_ner'] = args.e2e_ner
        configs['model_conf']['mysampler'] = args.mysampler
        self.epochs = configs["max_epoch"]
        self.grad_clip = configs["grad_clip"]
        self.accum_grad = configs["accum_grad"]
        self.log_interval = configs["log_interval"]
        self.keep_nbest_models = configs["keep_nbest_models"]
        self.lr = configs["optim_conf"]["lr"]
        self.warm_steps = configs["scheduler_conf"]["warmup_steps"]
        self.configs = configs

        self.model = init_paraformer_model(configs)
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = self.model.to(self.gpu_id)

        if self.args.mode == "train":
            init_logger(logger)

            if self.args.add_bert_feat:
                bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert_path)
            else:
                bert_tokenizer = None
            train_set = Dataset(self.args.train, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, speed_perturb=True, spec_aug=True, max_frame_num=self.args.max_frame_num, add_ne_feat=self.args.add_ne_feat or self.args.e2e_ner, nelabel_dict_file=self.args.ne_dict_file, add_bert_feat=self.args.add_bert_feat, bert_tokenizer=bert_tokenizer)
            self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, sampler=DistributedSampler(train_set), collate_fn=collate_fn, num_workers=self.args.num_workers)
            logger.info(f"\n Rank:{self.gpu_id} train: {len(train_set):6}\n")

            steps = (len(train_set)//self.args.batch_size) * self.epochs // self.accum_grad
            self.optimizer = AdamW(
                    [{'params': c, 'lr': self.lr} for n, c in self.model.named_parameters()], self.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_steps, steps)

            self.start_epoch = 1
            if self.gpu_id == 0:
                self.best_epoch, self.best_dev_loss = 1, 100000
                self.best_model_path, self.curr_model_path = os.path.join(self.args.path, 'best.model'), os.path.join(self.args.path, 'current.model')
                if self.keep_nbest_models > 1:
                    # the best loss is not saved here
                    self.nbest_loss_lst = [self.best_dev_loss] * (self.keep_nbest_models - 1)
                    self.nbest_epoch_lst = [self.best_epoch] * (self.keep_nbest_models - 1)
                else:
                    self.nbest_loss_lst = None
                    self.nbest_epoch_lst = None

            # continue training
            if self.args.pre_model != "None":
                loc = f"cuda:{self.gpu_id}"
                checkpoint = torch.load(self.args.pre_model, map_location=loc)
                self.start_epoch = checkpoint['epoch'] + 1
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'best_dev_loss' in checkpoint and self.gpu_id == 0:
                    self.best_dev_loss = checkpoint['best_dev_loss']
                if 'best_epoch' in checkpoint and self.gpu_id == 0:
                    self.best_epoch = checkpoint['best_epoch']
                else:
                    self.best_epoch = self.start_epoch
                
                if "nbest_loss_lst" in checkpoint and self.gpu_id == 0:
                    self.nbest_loss_lst = checkpoint["nbest_loss_lst"]

                if "nbest_epoch_lst" in checkpoint and self.gpu_id == 0:
                    self.nbest_epoch_lst = checkpoint["nbest_epoch_lst"]

                if self.gpu_id == 0:
                    logger.info(f"Rank:{self.gpu_id} loading previous model from {self.args.pre_model}, start epoch {self.start_epoch}, best epoch {self.best_epoch}, best dev loss {self.best_dev_loss}")
                else:
                    logger.info(f"Rank:{self.gpu_id} loading previous model from {self.args.pre_model}, start epoch {self.start_epoch}")
                del checkpoint
                torch.cuda.empty_cache()

        elif self.args.mode == "evaluate":
            if self.args.use_avg:
                path = os.path.join(self.args.path, 'avg.model')
            else:
                path = os.path.join(self.args.path, 'best.model')
            self.load_model(path)
            
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
                
    def load_model(self, model_path):
        assert model_path != 'None'
        loc = f"cuda:{self.gpu_id}"
        logger.info(f"Rank:{self.gpu_id} loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=loc)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
    def train(self):
        if self.gpu_id == 0:
            
            if self.args.add_bert_feat:
                bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert_path)
            else:
                bert_tokenizer = None

            dev_set = Dataset(self.args.dev, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_ne_feat=self.args.add_ne_feat or self.args.e2e_ner, nelabel_dict_file=self.args.ne_dict_file, add_bert_feat=self.args.add_bert_feat, bert_tokenizer=bert_tokenizer)
            dev_loader = DataLoader(dev_set, 
                                    collate_fn=collate_fn,
                                    batch_sampler=BatchSampler(RandomSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False),
                                    num_workers=self.args.num_workers
                                    )
            logger.info(f"\n dev:{len(dev_set):6}\n")

        
        elapsed = timedelta()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            start = datetime.now()
            logger.info(f"Rank:{self.gpu_id} Epoch {epoch} / {self.epochs}:")
            sum_loss = 0
            bar = progress_bar(self.train_loader)
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                feats_lengths = dic['audio_feat_length']
                target_lengths = dic['asr_target_length']
                keys = dic['keys']
                bio_ne = dic['bio_ne']
                bert_input = dic['bert_input']
                audio_feat = audio_feat.to(self.gpu_id)
                asr_target = asr_target.to(self.gpu_id)
                feats_lengths = feats_lengths.to(self.gpu_id)
                target_lengths = target_lengths.to(self.gpu_id)
                if bio_ne is not None:
                    bio_ne = bio_ne.to(self.gpu_id)
                if bert_input is not None:
                    bert_input = bert_input.to(self.gpu_id)
                loss, loss_att, loss_pre, loss_ctc, pre_loss_att, loss_ner, pre_loss_ner = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, bio_ne, bert_input)
                sum_loss += loss.item()
                loss = loss / self.accum_grad
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if i % self.accum_grad == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if i % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, i,
                        loss.item() * self.accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if pre_loss_att is not None:
                        log_str += 'pre_loss_att {:.6f} '.format(pre_loss_att.item())
                    if loss_ner is not None:
                        log_str += 'loss_ner {:.6f} '.format(loss_ner.item())
                    if pre_loss_ner is not None:
                        log_str += 'pre_loss_ner {:.6f} '.format(pre_loss_ner.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    if loss_pre is not None:
                        log_str += 'loss_pre {:.6f} '.format(loss_pre.item())
                    log_str += 'lr {:.8f}'.format(lr)
                    logger.info(f"Rank:{self.gpu_id} {log_str}")
            logger.info(f'Rank:{self.gpu_id} average train loss: {(sum_loss/i):.6f}\n')

            if self.gpu_id == 0:
                # evaluate on dev dataset
                this_dev_loss = self.loss_on_dev(dev_loader)
                t = datetime.now() - start
                if this_dev_loss < self.best_dev_loss:
                    self.best_dev_loss = this_dev_loss
                    self.best_epoch = epoch
                    logger.info(f"Rank:{self.gpu_id} {t}s elapsed (saved)\n")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_dev_loss': self.best_dev_loss,
                        'best_epoch': self.best_epoch,
                        'nbest_loss_lst': self.nbest_loss_lst,
                        'nbest_epoch_lst': self.nbest_epoch_lst,
                        }, self.best_model_path)
                else:
                    if self.nbest_loss_lst is not None and this_dev_loss < self.nbest_loss_lst[-1]:
                        # find the place to insert
                        idx = 0
                        for idx in range(self.keep_nbest_models - 1):
                            if this_dev_loss < self.nbest_loss_lst[idx]:
                                break
                        self.nbest_loss_lst.insert(idx, this_dev_loss)
                        self.nbest_epoch_lst.insert(idx, epoch)
                        to_del_loss = self.nbest_loss_lst.pop()
                        to_del_epoch = self.nbest_epoch_lst.pop()
                        save_model_path = os.path.join(self.args.path, f'nbest_epo_{epoch}.model')
                        del_model_path = os.path.join(self.args.path, f'nbest_epo_{to_del_epoch}.model')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_dev_loss': this_dev_loss,
                            'best_epoch': epoch,
                            'nbest_loss_lst': self.nbest_loss_lst,
                            'nbest_epoch_lst': self.nbest_epoch_lst,
                            }, save_model_path)
                        if os.path.exists(del_model_path):
                            os.remove(del_model_path)
                        logger.info(f"Rank:{self.gpu_id}: Epoch:{epoch} rankindex {idx} saved, Epoch:{to_del_epoch} deleted.\n")
                    logger.info(f"Rank:{self.gpu_id} {t}s elapsed\n")

                torch.save({
                    'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_dev_loss': self.best_dev_loss,
                        'best_epoch': self.best_epoch,
                        'nbest_loss_lst': self.nbest_loss_lst,
                        'nbest_epoch_lst': self.nbest_epoch_lst,
                        }, self.curr_model_path)
            torch.cuda.empty_cache()

        if self.gpu_id == 0:
            logger.info(f"Epoch {self.best_epoch} saved")
            logger.info(f"Best Average Dev loss: {self.best_dev_loss:.6f}")

    @torch.no_grad()
    def loss_on_dev(self, dataloader):
        self.model.eval()
        init_logger(logger)
        sum_dev_loss = 0
        bar = progress_bar(dataloader)
        for i, dic in enumerate(bar, 1):
            audio_feat = dic['audio_feat']
            asr_target = dic['asr_target']
            feats_lengths = dic['audio_feat_length']
            target_lengths = dic['asr_target_length']
            bio_ne = dic['bio_ne']
            bert_input = dic['bert_input']
            if bio_ne is not None:
                bio_ne = bio_ne.to(self.gpu_id)
            if bert_input is not None:
                bert_input = bert_input.to(self.gpu_id)
            audio_feat = audio_feat.to(self.gpu_id)
            asr_target = asr_target.to(self.gpu_id)
            feats_lengths = feats_lengths.to(self.gpu_id)
            target_lengths = target_lengths.to(self.gpu_id)
            loss, loss_att, loss_pre, loss_ctc, pre_loss_att, loss_ner, pre_loss_ner = self.model(
                            audio_feat, feats_lengths, asr_target, target_lengths, bio_ne, bert_input)
            sum_dev_loss += loss.item()
        
        logger.info(f'Rank:{self.gpu_id} average dev loss: {(sum_dev_loss/i):.6f}')
        return sum_dev_loss/i
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        init_logger(logger)

        if self.args.add_bert_feat:
            bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert_path)
        else:
            bert_tokenizer = None

        dataset = Dataset(self.args.input, self.args.char_dict, frame_length=self.args.frame_length, frame_shift=self.args.frame_shift, max_frame_num=self.args.max_frame_num, add_ne_feat=self.args.add_ne_feat or self.args.e2e_ner, nelabel_dict_file=self.args.ne_dict_file, add_bert_feat=self.args.add_bert_feat, bert_tokenizer=bert_tokenizer)
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                batch_size=self.args.batch_size,
                                drop_last=False),
                                num_workers=self.args.num_workers)
        logger.info(f"\n dataset: {len(dataset):6}\n")
        char_dict = {v:k for k, v in dataset.char_dict.items()}
        eos = len(char_dict) - 1
        special_brackets = ['(', ')', '<', '>', '[', ']']

        if self.args.e2e_ner:
            from supar.utils.metric import LevenNERMetric
            ne_label_itos = {v:k for k, v in dataset.nelabel_dict.items()}
            metric = LevenNERMetric(ne_label_itos)

        bar = progress_bar(dataloader)
        asr_res_file = self.args.res + '.asr'
        with open(asr_res_file, 'w') as fasr:
            for i, dic in enumerate(bar, 1):
                audio_feat = dic['audio_feat']
                asr_target = dic['asr_target']
                bio_ne = dic['bio_ne']
                feats_lengths = dic['audio_feat_length']
                # target_lengths = dic['asr_target_length']
                keys = dic['keys']
                
                audio_feat = audio_feat.to(self.gpu_id)
                # asr_target = asr_target.to(self.gpu_id)
                feats_lengths = feats_lengths.to(self.gpu_id)
                # target_lengths = target_lengths.to(self.gpu_id)
                
                hyps, _, ner_pred = self.model.module.recognize(audio_feat, feats_lengths)
                hyps = [hyp.tolist() for hyp in hyps]

                if self.args.e2e_ner:
                    # get_str
                    gold_txt = []
                    asr_target = [asr.tolist() for asr in asr_target]
                    for i, key in enumerate(keys):
                        content = []
                        for w in asr_target[i]:
                            if w == -1:
                                break
                            content.append(char_dict[w])
                        gold_res = ''.join(content)
                        for bracket in special_brackets:
                            gold_res = gold_res.replace(bracket, '')
                        gold_txt.append(gold_res)

                pred_txt = []
                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                    asr_res = ''.join(content)
                    for bracket in special_brackets:
                        asr_res = asr_res.replace(bracket, '')
                    pred_txt.append(asr_res)
                    logger.info('{}    {}'.format(key, asr_res))
                    fasr.write('{} {}\n'.format(key, asr_res))
                    fasr.flush()
                
                if self.args.e2e_ner:
                    bio_ne = bio_ne.masked_fill(bio_ne.eq(-1), dataset.nelabel_dict["O"])
                    metric(ner_pred, bio_ne, pred_txt, gold_txt)
        
        if self.args.e2e_ner:
            logger.info(f"{'metric:':5} {metric}")


if __name__ == "__main__":
    train_file = 'data/sp_ner/new_train.json'
    char_dict_file = 'data/sp_ner/chinese_char.txt'
    # a = torch.rand((3, 80))
    # b = torch.rand((4, 80))
    # pdb.set_trace()

    parser = Parser(train_file, char_dict_file)
    parser.train()
