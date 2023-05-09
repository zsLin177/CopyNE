# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

from distutils.command.config import config
import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_

from wenet.utils.metric import SeqTagMetric
from wenet.utils.compute_evaluation_indicators import call_as_metric

import os

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc = model(
                            feats, feats_lengths, target, target_lengths)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts

    def train_basener(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, bert_tokenids, ner_seq, _ = batch
                bert_tokenids = bert_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = ner_seq.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        mask = bert_tokenids.ne(bert_pad_idx) & bert_tokenids.ne(bert_bos_idx)
                        score = model(bert_tokenids)
                        loss = model.loss(score, ner_seq, mask)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)

    def eval_basener(self, model, data_loader, device, args, ner_label_dict):
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']

        dev_pred_file = os.path.join(args['model_dir'], 'pred_ner')
        dev_gold_file = args['eval_conf']['gold_file']

        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        metric = SeqTagMetric(ner_label_dict)
        f_w = open(dev_pred_file, 'w')
        ner_dict = args['ner_dict']
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys, bert_tokenids, ner_seq, _ = batch
                bert_tokenids = bert_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = ner_seq.size(0)
                if num_utts == 0:
                    continue
                score = model(bert_tokenids)
                mask = bert_tokenids.ne(bert_pad_idx) & bert_tokenids.ne(bert_bos_idx)
                loss = model.loss(score, ner_seq, mask)
                ner_preds = model.decode(score, mask)
                mask = mask[:, 1:]
                metric(ner_preds.masked_fill(~mask, -1), ner_seq.masked_fill(~mask, -1))
                lengths = mask.sum(-1).tolist()
                ner_preds = ner_preds.tolist()

                for i, key in enumerate(keys):
                    content2 = []
                    for w in ner_preds[i][0:lengths[i]]:
                        content2.append(ner_dict[w])
                    f_w.write('{} [{}]\n'.format(key, ','.join(content2)))

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
            logging.debug(f'SeqTagMetric: {metric}')
        f_w.close()
        nested_ner_F, _ = call_as_metric(dev_pred_file, dev_gold_file)
        logging.debug(f'Nested Ner Metric: {nested_ner_F:6.2%}')
        return total_loss, num_seen_utts, nested_ner_F

    def train_spner(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        ctc_bos_eos_idx = args['model_conf']['ctc_conf']['bos_eos_idx']
        txt_model_type = args['model_conf']['txt_type']
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths, bert_tokenid, ner_seq = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                bert_tokenid = bert_tokenid.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None

                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        chars = torch.cat((torch.ones_like(target[:, 0].unsqueeze(-1))* ctc_bos_eos_idx, target), -1)
                        chars = chars.masked_fill(chars.eq(-1), ctc_bos_eos_idx)
                        chars_mask = chars.eq(ctc_bos_eos_idx)
                        if txt_model_type == 'bert':
                            score, encoder_out, encoder_out_lens = model(feats, feats_lengths, bert_tokenid, bert_tokenid.ne(bert_pad_idx), chars, chars_mask)
                            mask = bert_tokenid.ne(bert_pad_idx) & bert_tokenid.ne(bert_bos_idx)
                        elif txt_model_type == 'lstm':
                            score, encoder_out, encoder_out_lens = model(feats, feats_lengths, chars, bert_tokenid.ne(bert_pad_idx), chars, chars_mask)
                            mask = ~chars_mask
                        loss = model.loss(score, ner_seq, mask, encoder_out, encoder_out_lens, target, target_lengths)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)

    def eval_spner(self, model, data_loader, device, args, ner_label_dict):
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        ctc_bos_eos_idx = args['model_conf']['ctc_conf']['bos_eos_idx']
        txt_model_type = args['model_conf']['txt_type']

        dev_pred_file = os.path.join(args['model_dir'], 'pred_ner')
        dev_gold_file = args['eval_conf']['gold_file']

        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        metric = SeqTagMetric(ner_label_dict)
        f_w = open(dev_pred_file, 'w')
        ner_dict = args['ner_dict']
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys, feats, target, feats_lengths, target_lengths, bert_tokenids, ner_seq = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                bert_tokenids = bert_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                chars = torch.cat((torch.ones_like(target[:, 0].unsqueeze(-1))* ctc_bos_eos_idx, target), -1)
                chars = chars.masked_fill(chars.eq(-1), ctc_bos_eos_idx)
                chars_mask = chars.eq(ctc_bos_eos_idx)
                if txt_model_type == 'bert':
                    mask = bert_tokenids.ne(bert_pad_idx) & bert_tokenids.ne(bert_bos_idx)
                    score, encoder_out, encoder_out_lens = model(feats, feats_lengths, bert_tokenids, bert_tokenids.ne(bert_pad_idx), chars, chars_mask)
                elif txt_model_type == 'lstm':
                    score, encoder_out, encoder_out_lens = model(feats, feats_lengths, chars, bert_tokenids.ne(bert_pad_idx), chars, chars_mask)
                    mask = ~chars_mask
                loss = model.loss(score, ner_seq, mask, encoder_out, encoder_out_lens, target, target_lengths)
                ner_preds = model.decode(score, mask)
                mask = mask[:, 1:]
                metric(ner_preds.masked_fill(~mask, -1), ner_seq.masked_fill(~mask, -1))
                lengths = mask.sum(-1).tolist()
                ner_preds = ner_preds.tolist()
                
                for i, key in enumerate(keys):
                    content2 = []
                    for w in ner_preds[i][0:lengths[i]]:
                        content2.append(ner_dict[w])
                    f_w.write('{} [{}]\n'.format(key, ','.join(content2)))


                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
            logging.debug(f'SeqTagMetric: {metric}')
        f_w.close()
        nested_ner_F, _ = call_as_metric(dev_pred_file, dev_gold_file)
        logging.debug(f'Nested Ner Metric: {nested_ner_F:6.2%}')

        return total_loss, num_seen_utts, nested_ner_F

    def train_lstmbasener(self, model, optimizer, scheduler, data_loader, device, writer,
                args, scaler):
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        char_pad_idx = args['lstm_conf']['char_pad_idx']
        char_bos_idx = args['lstm_conf']['char_bos_idx']

        logging.info('using accumulate grad, new batch size is {} times'
                    ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, bert_tokenids, ner_seq, char_tokenids = batch
                # bert_tokenids = bert_tokenids.to(device)
                char_tokenids = char_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = ner_seq.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        mask = char_tokenids.ne(char_pad_idx) & char_tokenids.ne(char_bos_idx)
                        score = model(char_tokenids)
                        loss = model.loss(score, ner_seq, mask)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)
        logging.debug(f'sum train sentences:{num_seen_utts}')

    def eval_lstmbasener(self, model, data_loader, device, args, ner_label_dict):
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        char_pad_idx = args['lstm_conf']['char_pad_idx']
        char_bos_idx = args['lstm_conf']['char_bos_idx']


        dev_pred_file = os.path.join(args['model_dir'], 'pred_ner')
        dev_gold_file = args['eval_conf']['gold_file']

        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        metric = SeqTagMetric(ner_label_dict)
        f_w = open(dev_pred_file, 'w')
        ner_dict = args['ner_dict']
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys, bert_tokenids, ner_seq, char_tokenids = batch
                # bert_tokenids = bert_tokenids.to(device)
                char_tokenids = char_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = ner_seq.size(0)
                if num_utts == 0:
                    continue
                score = model(char_tokenids)
                mask = char_tokenids.ne(char_pad_idx) & char_tokenids.ne(char_bos_idx)
                loss = model.loss(score, ner_seq, mask)
                ner_preds = model.decode(score, mask)
                mask = mask[:, 1:]
                metric(ner_preds.masked_fill(~mask, -1), ner_seq.masked_fill(~mask, -1))
                lengths = mask.sum(-1).tolist()
                ner_preds = ner_preds.tolist()

                for i, key in enumerate(keys):
                    content2 = []
                    for w in ner_preds[i][0:lengths[i]]:
                        content2.append(ner_dict[w])
                    f_w.write('{} [{}]\n'.format(key, ','.join(content2)))

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
            logging.debug(f'SeqTagMetric: {metric}')
        f_w.close()
        nested_ner_F, _ = call_as_metric(dev_pred_file, dev_gold_file)
        logging.debug(f'Nested Ner Metric: {nested_ner_F:6.2%}')
        return total_loss, num_seen_utts, nested_ner_F

    def train_lstmspner(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        char_pad_idx = args['lstm_conf']['char_pad_idx']
        char_bos_idx = args['lstm_conf']['char_bos_idx']
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths, _, ner_seq, char_tokenids = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                char_tokenids = char_tokenids.to(device)
                # bert_tokenid = bert_tokenid.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None

                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        mask = char_tokenids.ne(char_pad_idx) & char_tokenids.ne(char_bos_idx)
                        score, encoder_out, encoder_out_lens = model(feats, feats_lengths, char_tokenids)
                        loss = model.loss(score, ner_seq, mask, encoder_out, encoder_out_lens, target, target_lengths)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)
        logging.debug(f'sum train sentences:{num_seen_utts}')

    def eval_lstmspner(self, model, data_loader, device, args, ner_label_dict):
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        bert_pad_idx = args['bert_conf']['pad_idx']
        bert_bos_idx = args['bert_conf']['bos_idx']
        char_pad_idx = args['lstm_conf']['char_pad_idx']
        char_bos_idx = args['lstm_conf']['char_bos_idx']

        dev_pred_file = os.path.join(args['model_dir'], 'pred_ner')
        dev_gold_file = args['eval_conf']['gold_file']

        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        metric = SeqTagMetric(ner_label_dict)
        f_w = open(dev_pred_file, 'w')
        ner_dict = args['ner_dict']
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys, feats, target, feats_lengths, target_lengths, _, ner_seq, char_tokenids = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                # bert_tokenids = bert_tokenids.to(device)
                char_tokenids = char_tokenids.to(device)
                ner_seq = ner_seq.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                mask = char_tokenids.ne(char_pad_idx) & char_tokenids.ne(char_bos_idx)
                score, encoder_out, encoder_out_lens = model(feats, feats_lengths, char_tokenids)
                loss = model.loss(score, ner_seq, mask, encoder_out, encoder_out_lens, target, target_lengths)
                ner_preds = model.decode(score, mask)
                mask = mask[:, 1:]
                metric(ner_preds.masked_fill(~mask, -1), ner_seq.masked_fill(~mask, -1))
                lengths = mask.sum(-1).tolist()
                ner_preds = ner_preds.tolist()
                
                for i, key in enumerate(keys):
                    content2 = []
                    for w in ner_preds[i][0:lengths[i]]:
                        content2.append(ner_dict[w])
                    f_w.write('{} [{}]\n'.format(key, ','.join(content2)))


                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
            logging.debug(f'SeqTagMetric: {metric}')
        f_w.close()
        nested_ner_F, _ = call_as_metric(dev_pred_file, dev_gold_file)
        logging.debug(f'Nested Ner Metric: {nested_ner_F:6.2%}')
        logging.debug(f'sum dev sentences:{num_seen_utts}')
        return total_loss, num_seen_utts, nested_ner_F