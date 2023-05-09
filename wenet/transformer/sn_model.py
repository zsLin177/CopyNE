# Speech and NER model

import pdb
import torch
import torch.nn as nn

from wenet.transformer.bert import BertEmbedding
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.ctc import CTC, MaskedCTC
from wenet.transformer.cma import CrossModalityAttention
from supar.structs import CRFLinearChain
from supar.modules import VariationalLSTM
from supar.modules import SharedDropout
from wenet.utils.mask import make_pad_mask
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import remove_duplicates_and_blank
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaseNerModel(nn.Module):
    '''
    base ner model just uses text with bert as encoder
    '''
    def __init__(
        self,
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )

        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, words):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, n_out]
        x = self.bert_embed(words.unsqueeze(-1))
        # [batch_size, seq_len, n_labels]
        score = self.scorer(x)
        return score

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, seq_len, n_labels]
        gold_labels: [batch_size, seq_len-1]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        batch_size, seq_len = mask.shape
        loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        return loss
        
def init_base_ner_model(configs):
    n_labels = configs['num_ner_labels']
    bert = configs['bert_conf']['bert_path']
    bert_out_dim = configs['bert_conf']['out_dim']
    bert_n_layers = configs['bert_conf']['used_layers']
    bert_dropout = configs['bert_conf']['dropout']
    bert_pad_idx = configs['bert_conf']['pad_idx']
    model = BaseNerModel(n_labels, 
                        bert, 
                        bert_out_dim,
                        bert_n_layers,
                        bert_dropout,
                        bert_pad_idx=bert_pad_idx)
    return model

class SANModel(nn.Module):
    '''
    Speech and NER model, fused with cross modality attention.
    speech part use ctc along
    text part use bert
    fuse with one layer cross modality attention
    ---
    bert (string): 
        bert name or path
    bert_out_dim (int): 
        None for the default dim, others add a linear to transform
    bert_n_layers (int): 
        The number of layers from the model to use. If 0, uses all layers.
    bert_dropout (float): 
        The dropout ratio of BERT layers.
    requires_grad (bool):
        If ``True``, the model parameters will be updated together with the downstream task.
        Default: ``True``
    sp_encoder: the speech encoder
    '''
    def __init__(
        self,
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        sp_encoder: TransformerEncoder,
        ctc: CTC,
        ctc_weight=0.1,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        
        self.sp_encoder = sp_encoder
        # currently not using masked ctc
        self.ctc = ctc
        self.ctc_weight = ctc_weight
        
        self.cma = CrossModalityAttention(bert_out_dim)

        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, 
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bert_tokenid: torch.Tensor):
        '''
        bert_tokenid: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, d_model]
        bert_repr = self.bert_embed(bert_tokenid.unsqueeze(-1))
        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        encoder_out, encoder_mask = self.sp_encoder(speech, speech_lengths)
        s_max_len = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        mask = make_pad_mask(encoder_out_lens, s_max_len)
        # [batch_size, seq_len, d_model]
        h = self.cma(bert_repr, encoder_out, encoder_out, mask)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(h)
        return score, encoder_out, encoder_out_lens

    def decode(self, score, mask):
        '''
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        '''
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, 
        ner_score, 
        gold_ner, 
        ner_mask,
        s_encoder_out,
        s_encoder_lens,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        batch_size, seq_len = ner_mask.shape
        ner_loss = -CRFLinearChain(ner_score[:, 1:], ner_mask[:, 1:], self.trans).log_prob(gold_ner).sum() / seq_len
        loss_ctc = self.ctc(s_encoder_out, s_encoder_lens, text,
                                text_lengths)
        loss = self.ctc_weight * loss_ctc + (1-self.ctc_weight) * ner_loss
        return loss

    def ctc_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ):
        """ Apply CTC greedy search

        Args:
        """
        assert decoding_chunk_size != 0
        batch_size = encoder_out.shape[0]
        # Let's assume B = batch_size
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.ctc.ctc_lo.out_features-1)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def __repr__(self):
        s = ''
        s += f'ctc_weight: {self.ctc_weight}\n'
        return s + super().__repr__()

def init_speech_ner_model(configs):
    n_labels = configs['num_ner_labels']
    bert = configs['bert_conf']['bert_path']
    bert_out_dim = configs['bert_conf']['out_dim']
    bert_n_layers = configs['bert_conf']['used_layers']
    bert_dropout = configs['bert_conf']['dropout']
    bert_pad_idx = configs['bert_conf']['pad_idx']

    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    
    vocab_size = configs['output_dim']
    if_masked_ctc = configs['model_conf'].get('masked_ctc', False)
    if if_masked_ctc == True:
        ctc = MaskedCTC(vocab_size, encoder.output_size())
    else:
        ctc = CTC(vocab_size, encoder.output_size())

    model = SANModel(n_labels,
                    bert,
                    bert_out_dim,
                    bert_n_layers,
                    bert_dropout,
                    encoder,
                    ctc,
                    bert_pad_idx=bert_pad_idx,
                    ctc_weight=configs['ctcw'])
    
    return model

    
class CMAModel(nn.Module):
    '''
    Speech and NER model, fused with cross modality attention.
    speech part use ctc along
    text part use bert
    fuse with one layer cross modality attention
    ---
    bert (string): 
        bert name or path
    bert_out_dim (int): 
        None for the default dim, others add a linear to transform
    bert_n_layers (int): 
        The number of layers from the model to use. If 0, uses all layers.
    bert_dropout (float): 
        The dropout ratio of BERT layers.
    requires_grad (bool):
        If ``True``, the model parameters will be updated together with the downstream task.
        Default: ``True``
    sp_encoder: the speech encoder
    '''
    def __init__(
        self,
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        sp_encoder: TransformerEncoder,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        
        self.sp_encoder = sp_encoder
        # currently not using masked ctc
        
        self.cma = CrossModalityAttention(bert_out_dim)

        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, 
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bert_tokenid: torch.Tensor):
        '''
        bert_tokenid: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, d_model]
        bert_repr = self.bert_embed(bert_tokenid.unsqueeze(-1))
        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        encoder_out, encoder_mask = self.sp_encoder(speech, speech_lengths)
        s_max_len = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        mask = make_pad_mask(encoder_out_lens, s_max_len)
        # [batch_size, seq_len, d_model]
        h = self.cma(bert_repr, encoder_out, encoder_out, mask)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(h)
        return score, encoder_out, encoder_out_lens

    def decode(self, score, mask):
        '''
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        '''
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, 
        ner_score, 
        gold_ner, 
        ner_mask,
        s_encoder_out,
        s_encoder_lens,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        batch_size, seq_len = ner_mask.shape
        ner_loss = -CRFLinearChain(ner_score[:, 1:], ner_mask[:, 1:], self.trans).log_prob(gold_ner).sum() / seq_len
        return ner_loss


    def ctc_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ):
        """ Apply CTC greedy search

        Args:
        """
        assert decoding_chunk_size != 0
        batch_size = encoder_out.shape[0]
        # Let's assume B = batch_size
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.ctc.ctc_lo.out_features-1)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores


def init_cma_model(configs):
    n_labels = configs['num_ner_labels']
    bert = configs['bert_conf']['bert_path']
    bert_out_dim = configs['bert_conf']['out_dim']
    bert_n_layers = configs['bert_conf']['used_layers']
    bert_dropout = configs['bert_conf']['dropout']
    bert_pad_idx = configs['bert_conf']['pad_idx']
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    model = CMAModel(n_labels,
                    bert,
                    bert_out_dim,
                    bert_n_layers,
                    bert_dropout,
                    encoder,
                    bert_pad_idx=bert_pad_idx)
    
    return model

class LstmBaseNerModel(nn.Module):
    '''
    base ner model just uses text with lstm as encoder
    '''
    def __init__(
        self,
        n_words,
        n_labels,
        n_embed=50,  
        n_lstm_hidden=300, 
        n_lstm_layers=4, 
        lstm_dropout=0.33,
        char_pad_idx=-1):
        super().__init__()
        # 0:<blank>, 1:<unk>, n_words-1:<eos>
        self.char_embed = nn.Embedding(num_embeddings=n_words,
                                    embedding_dim=n_embed)
        self.n_input = n_embed
        self.txt_encoder = VariationalLSTM(input_size=self.n_input,
                                        hidden_size=n_lstm_hidden,
                                        num_layers=n_lstm_layers,
                                        bidirectional=True,
                                        dropout=lstm_dropout)
        self.encoder_dropout = SharedDropout(p=lstm_dropout)
        self.scorer = nn.Sequential(
                        nn.Linear(n_lstm_hidden*2, n_lstm_hidden),
                        nn.ReLU(),
                        nn.Linear(n_lstm_hidden, n_labels)
        )
        self.n_words = n_words
        self.pad_index = char_pad_idx
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, words):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        _, seq_len = words.shape
        # default pad_index is -1, map to <eos>
        if self.pad_index == -1:
            pad_mask = words.eq(self.pad_index)
            words.masked_fill_(pad_mask, self.n_words-1)
        mask = words.ne(self.pad_index)

        # [batch_size, seq_len, n_embed]
        char_embed = self.char_embed(words)
        x = pack_padded_sequence(char_embed, mask.sum(1).tolist(), True, False)
        x, _ = self.txt_encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # [batch_size, sqe_len, n_lstm_hidden*2]
        x = self.encoder_dropout(x)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(x)
        return score

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, seq_len, n_labels]
        gold_labels: [batch_size, seq_len-1]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        batch_size, seq_len = mask.shape
        loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        return loss

def init_lstmbase_model(configs):
    n_labels = configs['num_ner_labels']
    n_words = configs['lstm_conf']['num_chars']
    n_embed = configs['lstm_conf']['n_char_embed']
    n_lstm_hidden = configs['lstm_conf']['n_lstm_hidden']
    n_lstm_layers = configs['lstm_conf']['n_lstm_layers']
    lstm_dropout = configs['lstm_conf']['lstm_dropout']
    char_pad_idx = configs['lstm_conf']['char_pad_idx']
    return LstmBaseNerModel(n_words,
                            n_labels,
                            n_embed,
                            n_lstm_hidden,
                            n_lstm_layers,
                            lstm_dropout=lstm_dropout,
                            char_pad_idx=char_pad_idx)



class LstmCmaModel(nn.Module):
    '''
    speech ner model just uses text with lstm as encoder
    '''
    def __init__(
        self,
        n_words,
        n_labels,
        sp_encoder: TransformerEncoder,
        n_embed=50,  
        n_lstm_hidden=300, 
        n_lstm_layers=4, 
        lstm_dropout=0.33,
        char_pad_idx=-1):
        super().__init__()
        # 0:<blank>, 1:<unk>, n_words-1:<eos>
        self.char_embed = nn.Embedding(num_embeddings=n_words,
                                    embedding_dim=n_embed)
        self.n_input = n_embed
        self.txt_encoder = VariationalLSTM(input_size=self.n_input,
                                        hidden_size=n_lstm_hidden,
                                        num_layers=n_lstm_layers,
                                        bidirectional=True,
                                        dropout=lstm_dropout)
        self.encoder_dropout = SharedDropout(p=lstm_dropout)
        self.scorer = nn.Sequential(
                        nn.Linear(n_lstm_hidden*2, n_lstm_hidden),
                        nn.ReLU(),
                        nn.Linear(n_lstm_hidden, n_labels)
        )
        self.n_words = n_words
        self.pad_index = char_pad_idx
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

        self.sp_encoder = sp_encoder
        self.cma = CrossModalityAttention(n_lstm_hidden*2)

    def forward(self, 
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                words: torch.Tensor):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        _, seq_len = words.shape
        # default pad_index is -1, map to <eos>
        if self.pad_index == -1:
            pad_mask = words.eq(self.pad_index)
            words.masked_fill_(pad_mask, self.n_words-1)
        mask = words.ne(self.pad_index)

        # [batch_size, seq_len, n_embed]
        char_embed = self.char_embed(words)
        x = pack_padded_sequence(char_embed, mask.sum(1).tolist(), True, False)
        x, _ = self.txt_encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # [batch_size, sqe_len, n_lstm_hidden*2]
        x = self.encoder_dropout(x)

        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        encoder_out, encoder_mask = self.sp_encoder(speech, speech_lengths)
        s_max_len = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        mask = make_pad_mask(encoder_out_lens, s_max_len)
        # [batch_size, seq_len, d_model]
        h = self.cma(x, encoder_out, encoder_out, mask)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(h)
        return score, encoder_out, encoder_out_lens

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, 
        ner_score, 
        gold_ner, 
        ner_mask,
        s_encoder_out,
        s_encoder_lens,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        batch_size, seq_len = ner_mask.shape
        ner_loss = -CRFLinearChain(ner_score[:, 1:], ner_mask[:, 1:], self.trans).log_prob(gold_ner).sum() / seq_len
        return ner_loss

def init_lstmcma_model(configs):
    n_labels = configs['num_ner_labels']
    n_words = configs['lstm_conf']['num_chars']
    n_embed = configs['lstm_conf']['n_char_embed']
    n_lstm_hidden = configs['lstm_conf']['n_lstm_hidden']
    n_lstm_layers = configs['lstm_conf']['n_lstm_layers']
    lstm_dropout = configs['lstm_conf']['lstm_dropout']
    char_pad_idx = configs['lstm_conf']['char_pad_idx']

    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    return LstmCmaModel(n_words,
                        n_labels,
                        encoder,
                        n_embed,
                        n_lstm_hidden,
                        n_lstm_layers,
                        lstm_dropout=lstm_dropout,
                        char_pad_idx=char_pad_idx)


class LstmM3tModel(nn.Module):
    '''
    speech ner model just uses text with lstm as encoder
    '''
    def __init__(
        self,
        n_words,
        n_labels,
        sp_encoder: TransformerEncoder,
        ctc,
        n_embed=50,  
        n_lstm_hidden=300, 
        n_lstm_layers=4, 
        lstm_dropout=0.33,
        char_pad_idx=-1,
        ctc_weight=0.1):
        super().__init__()
        # 0:<blank>, 1:<unk>, n_words-1:<eos>
        self.char_embed = nn.Embedding(num_embeddings=n_words,
                                    embedding_dim=n_embed)
        self.n_input = n_embed
        self.txt_encoder = VariationalLSTM(input_size=self.n_input,
                                        hidden_size=n_lstm_hidden,
                                        num_layers=n_lstm_layers,
                                        bidirectional=True,
                                        dropout=lstm_dropout)
        self.encoder_dropout = SharedDropout(p=lstm_dropout)
        self.scorer = nn.Sequential(
                        nn.Linear(n_lstm_hidden*2, n_lstm_hidden),
                        nn.ReLU(),
                        nn.Linear(n_lstm_hidden, n_labels)
        )
        self.n_words = n_words
        self.pad_index = char_pad_idx
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

        self.sp_encoder = sp_encoder
        self.cma = CrossModalityAttention(n_lstm_hidden*2)

        self.ctc_weight = ctc_weight
        self.ctc = ctc


    def forward(self, 
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                words: torch.Tensor):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        _, seq_len = words.shape
        # default pad_index is -1, map to <eos>
        if self.pad_index == -1:
            pad_mask = words.eq(self.pad_index)
            words.masked_fill_(pad_mask, self.n_words-1)
        mask = words.ne(self.pad_index)

        # [batch_size, seq_len, n_embed]
        char_embed = self.char_embed(words)
        x = pack_padded_sequence(char_embed, mask.sum(1).tolist(), True, False)
        x, _ = self.txt_encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # [batch_size, sqe_len, n_lstm_hidden*2]
        x = self.encoder_dropout(x)

        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        encoder_out, encoder_mask = self.sp_encoder(speech, speech_lengths)
        s_max_len = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        mask = make_pad_mask(encoder_out_lens, s_max_len)
        # [batch_size, seq_len, d_model]
        h = self.cma(x, encoder_out, encoder_out, mask)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(h)
        return score, encoder_out, encoder_out_lens

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, 
        ner_score, 
        gold_ner, 
        ner_mask,
        s_encoder_out,
        s_encoder_lens,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        batch_size, seq_len = ner_mask.shape
        ner_loss = -CRFLinearChain(ner_score[:, 1:], ner_mask[:, 1:], self.trans).log_prob(gold_ner).sum() / seq_len
        loss_ctc = self.ctc(s_encoder_out, s_encoder_lens, text,
                                text_lengths)
        loss = self.ctc_weight * loss_ctc + (1-self.ctc_weight) * ner_loss
        return loss

    def ctc_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ):
        """ Apply CTC greedy search

        Args:
        """
        assert decoding_chunk_size != 0
        batch_size = encoder_out.shape[0]
        # Let's assume B = batch_size
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.ctc.ctc_lo.out_features-1)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def __repr__(self):
        s = ''
        s += f'ctc_weight: {self.ctc_weight}\n'
        return s + super().__repr__()
    
def init_lstmm3t_model(configs):
    n_labels = configs['num_ner_labels']
    n_words = configs['lstm_conf']['num_chars']
    n_embed = configs['lstm_conf']['n_char_embed']
    n_lstm_hidden = configs['lstm_conf']['n_lstm_hidden']
    n_lstm_layers = configs['lstm_conf']['n_lstm_layers']
    lstm_dropout = configs['lstm_conf']['lstm_dropout']
    char_pad_idx = configs['lstm_conf']['char_pad_idx']

    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    vocab_size = configs['output_dim']
    if_masked_ctc = configs['model_conf'].get('masked_ctc', False)
    if if_masked_ctc == True:
        ctc = MaskedCTC(vocab_size, encoder.output_size())
    else:
        ctc = CTC(vocab_size, encoder.output_size())

    return LstmM3tModel(n_words,
                        n_labels,
                        encoder,
                        ctc,
                        n_embed,
                        n_lstm_hidden,
                        n_lstm_layers,
                        lstm_dropout=lstm_dropout,
                        char_pad_idx=char_pad_idx,
                        ctc_weight=configs['ctcw'])


class BaseSimpleNerModel(nn.Module):
    '''
    base ner model just uses text with bert as encoder
    without crf
    '''
    def __init__(
        self,
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, n_out]
        x = self.bert_embed(words.unsqueeze(-1))
        # [batch_size, seq_len, n_labels]
        score = self.scorer(x)
        return score

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        return score[:, 1:].argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, seq_len, n_labels]
        gold_labels: [batch_size, seq_len-1]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        loss = self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])
        return loss

def init_base_simple_ner_model(configs):
    n_labels = configs['num_ner_labels']
    bert = configs['bert_conf']['bert_path']
    bert_out_dim = configs['bert_conf']['out_dim']
    bert_n_layers = configs['bert_conf']['used_layers']
    bert_dropout = configs['bert_conf']['dropout']
    bert_pad_idx = configs['bert_conf']['pad_idx']
    model = BaseSimpleNerModel(n_labels, 
                        bert, 
                        bert_out_dim,
                        bert_n_layers,
                        bert_dropout,
                        bert_pad_idx=bert_pad_idx)
    return model


class LstmBaseSimpleNerModel(nn.Module):
    '''
    base ner model just uses text with lstm as encoder
    '''
    def __init__(
        self,
        n_words,
        n_labels,
        n_embed=50,  
        n_lstm_hidden=300, 
        n_lstm_layers=4, 
        lstm_dropout=0.33,
        char_pad_idx=-1):
        super().__init__()
        # 0:<blank>, 1:<unk>, n_words-1:<eos>
        self.char_embed = nn.Embedding(num_embeddings=n_words,
                                    embedding_dim=n_embed)
        self.n_input = n_embed
        self.txt_encoder = VariationalLSTM(input_size=self.n_input,
                                        hidden_size=n_lstm_hidden,
                                        num_layers=n_lstm_layers,
                                        bidirectional=True,
                                        dropout=lstm_dropout)
        self.encoder_dropout = SharedDropout(p=lstm_dropout)
        self.scorer = nn.Sequential(
                        nn.Linear(n_lstm_hidden*2, n_lstm_hidden),
                        nn.ReLU(),
                        nn.Linear(n_lstm_hidden, n_labels)
        )
        self.n_words = n_words
        self.pad_index = char_pad_idx
        self.ce_criterion = nn.CrossEntropyLoss()
        

    def forward(self, words):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        _, seq_len = words.shape
        # default pad_index is -1, map to <eos>
        if self.pad_index == -1:
            pad_mask = words.eq(self.pad_index)
            words.masked_fill_(pad_mask, self.n_words-1)
        mask = words.ne(self.pad_index)

        # [batch_size, seq_len, n_embed]
        char_embed = self.char_embed(words)
        x = pack_padded_sequence(char_embed, mask.sum(1).tolist(), True, False)
        x, _ = self.txt_encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # [batch_size, sqe_len, n_lstm_hidden*2]
        x = self.encoder_dropout(x)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(x)
        return score

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        return score[:, 1:].argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, seq_len, n_labels]
        gold_labels: [batch_size, seq_len-1]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        loss = self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])
        return loss

def init_lstmbasesimple_model(configs):
    n_labels = configs['num_ner_labels']
    n_words = configs['lstm_conf']['num_chars']
    n_embed = configs['lstm_conf']['n_char_embed']
    n_lstm_hidden = configs['lstm_conf']['n_lstm_hidden']
    n_lstm_layers = configs['lstm_conf']['n_lstm_layers']
    lstm_dropout = configs['lstm_conf']['lstm_dropout']
    char_pad_idx = configs['lstm_conf']['char_pad_idx']
    return LstmBaseSimpleNerModel(n_words,
                            n_labels,
                            n_embed,
                            n_lstm_hidden,
                            n_lstm_layers,
                            lstm_dropout=lstm_dropout,
                            char_pad_idx=char_pad_idx)
