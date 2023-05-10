import random
import math
import torch
import torchaudio as ta
import re
from utils.process import read_json, read_symbol_table, build_ner_vocab, read_context_table
from supar.utils.fn import pad
from supar.utils.field import SubwordField

per_regex = '\[.+?\]'
loc_regex = '\(.+?\)'
org_regex = '\<.+?\>'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, char_dict_file, num_mel_bins=80, frame_length=25, frame_shift=10, max_frame_num=100000, speed_perturb=False, spec_aug=False, bert_tokenizer=None, e2ener=False, use_same_tokenizer=False, add_context=False) -> None:
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.max_frame_num = max_frame_num
        self.data = read_json(json_file)
        # <black> = 0, <unk> = 1, <sos/eos> = 4232
        self.char_dict = read_symbol_table(char_dict_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bert_tokenizer = bert_tokenizer
        self.e2ener = e2ener
        self.use_same_tokenizer = use_same_tokenizer
        self.add_context = add_context
        if self.bert_tokenizer is not None:
            self.word_field = SubwordField('bert',
                                    pad=self.bert_tokenizer.pad_token,
                                    unk=self.bert_tokenizer.unk_token,
                                    bos=self.bert_tokenizer.cls_token,
                                    eos=self.bert_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bert_tokenizer.tokenize)
            self.word_field.vocab = self.bert_tokenizer.get_vocab()

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)

        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]
        label, tokens = self.tokenize_for_asr(sentence)
        if not self.use_same_tokenizer:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = self.word_field.transform([tokens])[0][1:-1].squeeze(-1)
        
        key = self.data[index]["key"]

        if self.bert_tokenizer is not None:
            bert_input = self.word_field.transform([tokens])[0]
        else:
            bert_input = None

        if self.add_context:
            need_att_mask = self.build_need_att_mask(sentence)
        else:
            need_att_mask = None

        return y, label, key, bert_input, need_att_mask

    def build_need_att_mask(self, sentence, add_bos=True):
        if add_bos:
            mask = [0] * (len(sentence)+1)
            sentence = '#' + sentence
        else:
            mask = [0] * len(sentence)
        mask = torch.tensor(mask)

        per_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(per_regex, sentence)]
        loc_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(loc_regex, sentence)]
        org_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(org_regex, sentence)]

        for st, ed in per_intervals+loc_intervals+org_intervals:
            mask[st:ed+1] = 1
        
        return mask

    def tokenize_for_asr(self, sentence):
        label = []
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        for ch in tokens:
            if ch in self.char_dict:
                label.append(self.char_dict[ch])
            elif '<unk>' in self.char_dict:
                label.append(self.char_dict['<unk>'])
            else:
                raise KeyError
        return label, tokens

def collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    asr_target = pad([instance[1] for instance in batch], padding_value=-1)
    asr_target_length = torch.tensor([instance[1].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[2] for instance in batch]
    
    curr_bert = batch[0][3]

    if batch[0][4] is not None:
        need_att_mask = pad([instance[4] for instance in batch], padding_value=0)
        need_att_mask = need_att_mask.bool()
    else:
        need_att_mask = None

    if curr_bert is not None:
        bert_pad_idx = 0
        bert_input = pad([instance[3] for instance in batch], padding_value=bert_pad_idx)
        return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'bert_input': bert_input,
                'need_att_mask': need_att_mask}
    
    return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'need_att_mask': need_att_mask}


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, char_dict_file, if_flat, if_build_ner_vocab=True, ner_vocab=None, num_mel_bins=80, speed_perturb=False, spec_aug=False, bert_tokenizer=None, use_same_tokenizer=False) -> None:
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.data = read_json(json_file, if_flat)
        # <black> = 0, <unk> = 1, <sos/eos> = 4232
        self.char_dict = read_symbol_table(char_dict_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bert_tokenizer = bert_tokenizer
        self.use_same_tokenizer = use_same_tokenizer
        if self.bert_tokenizer is not None:
            self.word_field = SubwordField('bert',
                                    pad=self.bert_tokenizer.pad_token,
                                    unk=self.bert_tokenizer.unk_token,
                                    bos=self.bert_tokenizer.cls_token,
                                    eos=self.bert_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bert_tokenizer.tokenize)
            self.word_field.vocab = self.bert_tokenizer.get_vocab()

        if if_build_ner_vocab:
            self.ner_vocab = build_ner_vocab(self.data)
        else:
            self.ner_vocab = ner_vocab

    def __len__(self):
        return len(self.data)

    def get_ner(self, index):
        # based on words
        """
        sentence: 我是苏州人 -> [BOS] 我 是 苏 州 人 [EOS]
        ner_labels: [[None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, LOC, None],
                     [None, None, None, LOC, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                    ]
        """
        real_char_num = len(self.data[index]['sentence'])
        # plus bos and eos
        char_num = real_char_num + 2
        null_label_idx = len(self.ner_vocab)
        # row: end; column: start
        ner_labels = [[null_label_idx] * char_num for i in range(char_num)]

        dic = self.data[index]
        for ner in dic['entity']:
            st, ed, label = ner[0], ner[1], ner[3]
            ner_labels[ed][st+1] = self.ner_vocab[label]
        # lower triangular
        return ner_labels
    
    def tokenize_for_asr(self, sentence):
        label = []
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        for ch in tokens:
            if ch in self.char_dict:
                label.append(self.char_dict[ch])
            elif '<unk>' in self.char_dict:
                label.append(self.char_dict['<unk>'])
            else:
                raise KeyError
        return label, tokens

    def __getitem__(self, index):
        try:
            audio_path = self.data[index]["wav"]
        except KeyError:
            audio_path = self.data[index]["audio"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=25, frame_shift=10,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        try:
            sentence = self.data[index]["txt"]
        except KeyError:
            sentence = self.data[index]["sentence"]
        label, tokens = self.tokenize_for_asr(sentence)
        if not self.use_same_tokenizer:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = self.word_field.transform([tokens])[0][1:-1].squeeze(-1)
        
        try:
            key = self.data[index]["key"]
        except KeyError:
            key = None

        if self.bert_tokenizer is not None:
            bert_input = self.word_field.transform([tokens])[0]
        else:
            bert_input = None
        
        # get ner labels
        ner_tensor = torch.tensor(self.get_ner(index))
        return y, label, key, bert_input, ner_tensor

def ner_collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    asr_target = pad([instance[1] for instance in batch], padding_value=-1)
    asr_target_length = torch.tensor([instance[1].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[2] for instance in batch]
    ner_labels = pad([instance[4] for instance in batch], padding_value=-1)
    
    curr_bert = batch[0][3]
    if curr_bert is not None:
        bert_pad_idx = 0
        bert_input = pad([instance[3] for instance in batch], padding_value=bert_pad_idx)
        return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'bert_input': bert_input,
                'ner': ner_labels}
    
    return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'ner': ner_labels}
 
class BartSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                json_file,
                num_mel_bins=80, 
                frame_length=25,
                frame_shift=10,
                max_frame_num=512,
                speed_perturb=False, 
                spec_aug=False, 
                bart_tokenizer=None, 
                e2ener=False) -> None:
        """
        in fnlp/bart-base-chinese, the tokenizer is same as bert-base-chinese
        """
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.max_frame_num = max_frame_num
        self.data = read_json(json_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bart_tokenizer = bart_tokenizer
        self.e2ener = e2ener
        if self.bart_tokenizer is not None:
            self.word_field = SubwordField('bart',
                                    pad=self.bart_tokenizer.pad_token,
                                    unk=self.bart_tokenizer.unk_token,
                                    bos=self.bart_tokenizer.cls_token,
                                    eos=self.bart_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bart_tokenizer.tokenize)
            self.word_field.vocab = self.bart_tokenizer.get_vocab()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)
        
        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]

        tokens = self.tokenize(sentence)
        key = self.data[index]["key"]
        if self.bart_tokenizer is not None:
            bart_input = self.word_field.transform([tokens])[0]
        else:
            bart_input = None
        
        return y, key, bart_input

    def tokenize(self, sentence):
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        return tokens
        
def bartseq2seq_collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[1] for instance in batch]
    curr_bart = batch[0][2]
    if curr_bart is not None:
        bart_pad_idx = 0
        bart_input = pad([instance[2] for instance in batch], padding_value=bart_pad_idx)
        return {'audio_feat': audio_feat,
                'audio_feat_length': audio_feat_length,
                'keys': keys,
                'bart_input': bart_input}
    else:
        raise KeyError("need bart input")

class BartTxtSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                json_file,
                bart_tokenizer=None, 
                e2ener=False,
                add_noise=False,
                mask_rate=0.35,
                poisson_avg=3) -> None:
        """
        in fnlp/bart-base-chinese, the tokenizer is same as bert-base-chinese
        """
        super().__init__()
        self.data = read_json(json_file)
        self.bart_tokenizer = bart_tokenizer
        self.e2ener = e2ener
        self.add_noise = add_noise
        self.mask_rate = mask_rate
        self.poisson_avg = poisson_avg
        self.poisson = torch.distributions.Poisson(poisson_avg)
        if self.bart_tokenizer is not None:
            self.word_field = SubwordField('bart',
                                    pad=self.bart_tokenizer.pad_token,
                                    unk=self.bart_tokenizer.unk_token,
                                    bos=self.bart_tokenizer.cls_token,
                                    eos=self.bart_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bart_tokenizer.tokenize)
            self.word_field.vocab = self.bart_tokenizer.get_vocab()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.e2ener:
            tgt = self.data[index]["txt"]
        else:
            tgt = self.data[index]["ner_txt"]
        tgt_tokens = self.my_tokenize(tgt)

        # src = self.data[index]["asrout"]
        src = self.data[index]["txt"]
        src_tokens = self.my_tokenize(src)
        if self.add_noise:
            # TODO
            pass
        key = self.data[index]["key"]

        return self.word_field.transform([src_tokens])[0].squeeze(-1), self.word_field.transform([tgt_tokens])[0].squeeze(-1), key

    def my_tokenize(self, sentence):
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        return tokens

    def add_noise_for_text_infilling(self, src_tokens_lst, max_span_len=10):
        src_token_num = len(src_tokens_lst)
        n_masked_num = math.ceil(src_token_num * self.mask_rate)
        if n_masked_num == 0:
            return src_tokens_lst
        
        span_masked_res = []
        # containing the character masked and the [MASK] added
        masked_num = 0
        masked_mask = torch.tensor([1] * src_token_num, dtype=torch.long)
        # the num of [MASK] added
        added_mask_num = 0

        while masked_num < n_masked_num:
            span_len = min(self.poisson.sample(sample_shape=(1,)).item(), max_span_len)
            if span_len > 0:
                span_len = min()
            else:
                added_mask_num += 1
                masked_num += 1

def barttxtseq2seq_collate_fn(batch):
    bart_pad_idx = 0
    keys = [instance[2] for instance in batch]
    src = pad([instance[0] for instance in batch], padding_value=bart_pad_idx)
    tgt = pad([instance[1] for instance in batch], padding_value=bart_pad_idx)
    return {'keys': keys,
            'src': src,
            'tgt': tgt}

class CLASDataset(Dataset):
    def __init__(self, json_file, char_dict_file, ne_dict_file, is_training, is_dev, is_test, num_mel_bins=80, frame_length=25, frame_shift=10, max_frame_num=100000, speed_perturb=False, spec_aug=False, bert_tokenizer=None, e2ener=False, use_same_tokenizer=False, add_context=True, pad_context=2) -> None:
        super().__init__(json_file, char_dict_file, num_mel_bins, frame_length, frame_shift, max_frame_num, speed_perturb, spec_aug, bert_tokenizer, e2ener, use_same_tokenizer, add_context)
        self.ne_stoi = read_context_table(ne_dict_file)
        self.ne_itos = {v:k for k, v in self.ne_stoi.items()}
        self.is_training = is_training
        self.is_dev = is_dev
        self.is_test = is_test
        self.pad_context = pad_context

    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])

        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)

        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat

        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)

        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]

        ne_set = set([sentence[st: ed+1] for ne_label, st, ed in self.data[index]["ne_lst"] if ed > st])

        key = self.data[index]["key"]

        return y, sentence, ne_set, key, self.ne_itos, self.is_training, self.is_dev, self.is_test, self.char_dict, self.pad_context

def add_match_symbol(sentence, ne_set):
        """
        add '$' at the end of nes
        """
        # if len(ne_lst) == 0:
        #     return sentence
        # new_sentence = sentence
        # for i in range(len(ne_lst)):
        #     raw_st, raw_ed = ne_lst[i][1], ne_lst[i][2]
        #     new_sentence = new_sentence[0: raw_ed+1+i] + '$' + new_sentence[raw_ed+1+i:]
        # return new_sentence
        new_s = sentence
        for item in ne_set:
            new_s = re.sub(r'({})'.format(item), r'\1$', new_s)
        return new_s

def clas_collate_fn(batch):
    """
    ne_vocab: is_training==true, the ne_vocab is all the nes in train set.
              or, it is the supported vocabulary in the dev or test time.
    is_training: is_training==true during training and dev, we need to generate context vocabulary for every training batch.
    if_sample_nes: whhether to sample nes from ne_vocab for batch
    note: ne_vocab do not contain the NULL, we add NULL in the model

    """
    ne_vocab, is_training, is_dev, is_test, char_vocab = batch[0][4], batch[0][5], batch[0][6], batch[0][7], batch[0][8]

    batch_size = len(batch)
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)

    raw_sentence_lst = [instance[1] for instance in batch]

    sample_num = None
    if is_training:
        batch_ne_set = set()
        for instance in batch:
            this_ne_set = instance[2]
            if len(this_ne_set) > 0:
                batch_ne_set = batch_ne_set | this_ne_set
            else:
                # sample n-grams as ne
                this_s = instance[1]
                k = torch.randint(1, 3, (1, )).item()
                n_order = min(torch.randint(2, 5, (1, )).item(), len(this_s))
                if n_order == len(this_s):
                    k = 1
                while k > 0:
                    st_idx = torch.randint(0, len(this_s)-n_order+1, (1, )).item()
                    batch_ne_set.add(this_s[st_idx:st_idx+n_order])
                    k -= 1
        
        max_bei = batch[0][9]
        # min_ne_num = batch_size
        # max_ne_num = batch_size * max_bei
        sample_num = int(torch.rand(1).item() * batch_size + batch_size * (max_bei-1))
        if len(batch_ne_set) < sample_num:
            # sample nes from the training_ne_vocab
            while len(batch_ne_set) < sample_num:
                sample_idx = random.randint(0, len(ne_vocab)-1)
                if len(ne_vocab[sample_idx]) > 1:
                    batch_ne_set.add(ne_vocab[sample_idx])
        
        batch_ne_lst = list(batch_ne_set)
        new_sentence_lst = [add_match_symbol(sentence, batch_ne_lst) for sentence in raw_sentence_lst]
    elif is_dev:
        # 因为要计算dev loss, 所以dev也要重新处理一下句子, dev用的是提供的全部的ne_vocab
        batch_ne_lst = [ne_vocab[i] for i in range(len(ne_vocab))]
        new_sentence_lst = [add_match_symbol(sentence, batch_ne_lst) for sentence in raw_sentence_lst]
    else:
        # 在测试的的时候并不需要用到sentence, 但是为了统一, 还是将这个输出
        batch_ne_lst = [ne_vocab[i] for i in range(len(ne_vocab))]
        new_sentence_lst = raw_sentence_lst

    # TODO: build batch_ne_lst tensor
    if is_training:
        # 在dev和test的时候context都是固定的，所以可以只创建一次, 如果是None, 则说明没有创建, 需要创建
        context_tensor = build_ne_vocab_tensor(batch_ne_lst, char_vocab)
    else:
        context_tensor = None

    # TODO: build new_sentence_lst tensor
    asr_target, asr_target_length = build_new_s_tensor(new_sentence_lst, char_vocab)

    keys = [instance[3] for instance in batch]

    return {'audio_feat': audio_feat,
            'asr_target': asr_target,
            'audio_feat_length': audio_feat_length,
            'asr_target_length': asr_target_length,
            'keys': keys,
            'need_att_mask': None,
            'context_tensor': context_tensor,
            'sample_num': sample_num}

def copyne_collate_fn(batch):
    """
    for instances without nes, we sample n-grams,
    and can choose to pad ne dict with other nes
    """
    ne_vocab, is_training, is_dev, is_test, char_vocab = batch[0][4], batch[0][5], batch[0][6], batch[0][7], batch[0][8]

    batch_size = len(batch)
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)

    raw_sentence_lst = [instance[1] for instance in batch]
    sample_num = None
    if is_training:
        batch_ne_set = set()
        for instance in batch:
            this_ne_set = instance[2]
            if len(this_ne_set) > 0:
                batch_ne_set = batch_ne_set | this_ne_set
            else:
                # sample n-grams as ne
                this_s = instance[1]
                k = torch.randint(1, 3, (1, )).item()
                n_order = min(torch.randint(2, 5, (1, )).item(), len(this_s))
                if n_order == len(this_s):
                    k = 1
                while k > 0:
                    st_idx = torch.randint(0, len(this_s)-n_order+1, (1, )).item()
                    batch_ne_set.add(this_s[st_idx:st_idx+n_order])
                    k -= 1
        
        # pad distractors
        max_bei = batch[0][9]
        sample_num = int(torch.rand(1).item() * batch_size + batch_size * (max_bei-1))
        if len(batch_ne_set) < sample_num:
            # sample nes from the training_ne_vocab
            while len(batch_ne_set) < sample_num:
                sample_idx = torch.randint(0, len(ne_vocab), (1, )).item()
                if len(ne_vocab[sample_idx]) > 1:
                    batch_ne_set.add(ne_vocab[sample_idx])
        batch_ne_lst = list(batch_ne_set)
        # 从大到小排序
        batch_ne_lst = sorted(batch_ne_lst, key=lambda x:len(x), reverse=True)
    elif is_dev:
        # dev is used to analyse the effect of the size of the vocab
        # we use the _min_len file, so the set will not be empty
        batch_ne_set = set()
        for instance in batch:
            this_ne_set = instance[2]
            if len(this_ne_set) > 0:
                batch_ne_set = batch_ne_set | this_ne_set
        # only use the ne in the sentence, without n-grams and other nes
        batch_ne_lst = list(batch_ne_set)
        # 从大到小排序
        batch_ne_lst = sorted(batch_ne_lst, key=lambda x:len(x), reverse=True)
    else:
        batch_ne_lst = [ne_vocab[i] for i in range(len(ne_vocab))]
        # batch_ne_lst = sorted(batch_ne_lst, key=lambda x:len(x), reverse=True)
    
    if is_training:
        context_tensor = build_ne_vocab_tensor(batch_ne_lst, char_vocab)
        att_tgt = pad([build_copy_tgt(sent, batch_ne_lst) for sent in raw_sentence_lst], padding_value=-1)
    elif is_dev:
        context_tensor = build_ne_vocab_tensor(batch_ne_lst, char_vocab)
        att_tgt = pad([build_copy_tgt(sent, batch_ne_lst) for sent in raw_sentence_lst], padding_value=-1)
    else:
        context_tensor = None
        att_tgt = None
    asr_target, asr_target_length = build_new_s_tensor(raw_sentence_lst, char_vocab)
    keys = [instance[3] for instance in batch]

    return {'audio_feat': audio_feat,
            'asr_target': asr_target,
            'audio_feat_length': audio_feat_length,
            'asr_target_length': asr_target_length,
            'keys': keys,
            'need_att_mask': None,
            'context_tensor': context_tensor,
            'sample_num': sample_num,
            'ne_lst': batch_ne_lst,
            'att_tgt': att_tgt}

def build_copy_tgt(sentence, ne_lst):
    res = [len(ne_lst)] * (len(sentence)+1)
    for idx, ne in enumerate(ne_lst):
        lst = [(item.span()[0], item.span()[1]-1) for item in re.finditer(ne, sentence)]
        for st, ed in lst:
            if res[st] != len(ne_lst):
                continue
            res[st] = idx
    return torch.tensor(res, dtype=torch.int64)
    
def build_ne_vocab_tensor(ne_lst, char_vocab):
    res = []
    for ne in ne_lst:
        label = []
        for ch in ne:
            if ch == ' ':
                ch = '_'
            if ch in char_vocab:
                label.append(char_vocab[ch])
            elif '<unk>' in char_vocab:
                label.append(char_vocab['<unk>'])
            else:
                raise KeyError
        res.append(torch.tensor(label, dtype=torch.long))
    return pad(res, padding_value=-1)
    
def build_new_s_tensor(new_s_lst, char_vocab):
    s_tensor = build_ne_vocab_tensor(new_s_lst, char_vocab)
    return s_tensor, s_tensor.ne(-1).sum(-1)

class CopyASRDataset(Dataset):
    def __init__(self, json_file, char_dict_file, ne_dict_file, is_training, is_dev, is_test, num_mel_bins=80, frame_length=25, frame_shift=10, max_frame_num=100000, speed_perturb=False, spec_aug=False, bert_tokenizer=None, e2ener=False, use_same_tokenizer=False, pad_context=2, add_context=True):
        super().__init__(json_file, char_dict_file, num_mel_bins, frame_length, frame_shift, max_frame_num, speed_perturb, spec_aug, bert_tokenizer, e2ener, use_same_tokenizer, add_context)
        self.ne_stoi = read_context_table(ne_dict_file)
        self.ne_itos = {v:k for k, v in self.ne_stoi.items()}
        assert min(len(k) for k in self.ne_stoi.keys()) > 1
        self.is_training = is_training
        self.is_dev = is_dev
        self.is_test = is_test
        self.pad_context = pad_context
        char_itos_dict = {v:k for k, v in self.char_dict.items()}
        self.char_itos = [char_itos_dict[i] for i in range(len(char_itos_dict))]

    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])

        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)

        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat

        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)

        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]
        
        # only contain nes that length > 1
        ne_set = set([sentence[st: ed+1] for ne_label, st, ed in self.data[index]["ne_lst"] if ed>st])

        key = self.data[index]["key"]

        return y, sentence, ne_set, key, self.ne_itos, self.is_training, self.is_dev, self.is_test, self.char_dict, self.pad_context, self.char_itos

def copyasr_collate_fn(batch):
    """_summary_
        for instances without nes, we sample n-grams,
        and can choose to pad ne dict with other nes.
        use max match 
    """
    ne_vocab, is_training, is_dev, is_test, char_vocab = batch[0][4], batch[0][5], batch[0][6], batch[0][7], batch[0][8]
    char_itos = batch[0][10]
    unk_token = "<unk>"
    unk_id = char_vocab[unk_token]
    eos_token = "<sos/eos>"
    eos_id = char_vocab[eos_token]
    batch_size = len(batch)
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)

    raw_sentence_lst = [instance[1] for instance in batch]
    sample_num = None
    if is_training:
        batch_ne_set = set()
        for instance in batch:
            this_ne_set = instance[2]
            if len(this_ne_set) > 0:
                batch_ne_set = batch_ne_set | this_ne_set
            else:
                # sample n-grams as ne
                this_s = instance[1]
                if len(this_s) <= 1:
                    continue
                k = torch.randint(1, 3, (1, )).item()
                n_order = min(torch.randint(2, 5, (1, )).item(), len(this_s))
                if n_order == len(this_s):
                    k = 1
                while k > 0:
                    # may have the same st_idx, but its ok
                    st_idx = torch.randint(0, len(this_s)-n_order+1, (1, )).item()
                    batch_ne_set.add(this_s[st_idx:st_idx+n_order])
                    k -= 1
        # pad distractors
        # sample nes from the training_ne_vocab
        num_inbatch_nums = len(batch_ne_set)
        sample_num = (batch[0][9]-1) * num_inbatch_nums
        while sample_num > 0:
            sample_idx = torch.randint(0, len(ne_vocab), (1, )).item()
            if ne_vocab[sample_idx] not in batch_ne_set:
                batch_ne_set.add(ne_vocab[sample_idx])
                sample_num -= 1
        batch_ne_lst = list(batch_ne_set)
        # 从大到小排序
        batch_ne_lst = sorted(batch_ne_lst, key=lambda x:len(x), reverse=True)
        assert min(len(ne) for ne in batch_ne_lst) > 1
    elif is_dev:
        batch_ne_lst = [ne_vocab[i] for i in range(len(ne_vocab))]
    else:
        batch_ne_lst = [ne_vocab[i] for i in range(len(ne_vocab))]
    
    if is_training:
        context_tensor = build_ne_vocab_tensor(batch_ne_lst, char_vocab)
        batch_info_tgt = []
        batch_info_mask = []
        for sent in raw_sentence_lst:
            info_tgt, info_mask = build_infonce_tgt(sent, batch_ne_lst, char_itos, unk_id, eos_id)
            batch_info_tgt.append(info_tgt)
            batch_info_mask.append(info_mask)
        batch_info_tgt = pad(batch_info_tgt, padding_value=-1)
        batch_info_mask = pad(batch_info_mask, padding_value=0).bool()
    else:
        context_tensor = None
        batch_info_tgt = None
        batch_info_mask = None
    
    asr_target, asr_target_length = build_new_s_tensor(raw_sentence_lst, char_vocab)
    keys = [instance[3] for instance in batch]
    return {'audio_feat': audio_feat,
            'asr_target': asr_target,
            'audio_feat_length': audio_feat_length,
            'asr_target_length': asr_target_length,
            'keys': keys,
            'need_att_mask': None,
            'context_tensor': context_tensor,
            'sample_num': sample_num,
            'ne_lst': batch_ne_lst,
            'info_tgt': batch_info_tgt,
            'info_mask': batch_info_mask}

def build_infonce_tgt(s, ne_lst, char_itos, unk_id, eos_id):
    """
    use max match
    """
    match_lst = char_itos + ne_lst
    res = [unk_id] * len(s)  # 初始化res列表， 每个字符属于match_lst中的哪个词，全部填充为unk_id
    mask = [False] * len(s)  # 初始化mask列表, 每个字符是否是一个切分的首个字符，全部填充为False
    i = 0
    len_max = min(len(s), max(len(ne) for ne in match_lst))
    while i < len(s):
        match = False
        for length in range(len_max, 0, -1):
            if s[i:i+length] in match_lst:
                res[i:i+length] = [match_lst.index(s[i:i+length])] * length
                if length == 1:
                    mask[i] = True
                else:
                    mask[i] = True
                    mask[i+1:i+length] = [False] * (length - 1)
                i += length
                match = True
                break
        if not match:
            mask[i] = True
            i += 1
        len_max = min(len_max, len(s) - i)
    # shift
    res.append(eos_id)
    mask.append(True)
    return torch.tensor(res, dtype=torch.int64), torch.tensor(mask, dtype=torch.long)



    
         
        

            

