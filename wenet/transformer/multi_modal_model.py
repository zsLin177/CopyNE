
import torch
import torch.nn as nn

from wenet.transformer.cma import CrossModalityAttention, GateAttentionFusionLayer, CTCAlignFusion, CTCAlignCMAFusion
from wenet.utils.mask import make_pad_mask
from supar.structs import CRFLinearChain
from wenet.transformer.bert import BertEmbedding
from wenet.transformer.lstm import LstmEncoder
from wenet.transformer.asr_model import init_ctc_model, CTCModel
from wenet.utils.checkpoint import load_checkpoint

class MultiModalModelEncoder(nn.Module):
    '''
    cover text model and speech model
    '''
    def __init__(self,
                txt_model,
                sp_model,
                fusion_type):
        super().__init__()
        self.txt_model = txt_model
        self.sp_model = sp_model
        # fusion layer
        self.fusion_type = fusion_type
        if fusion_type == 'cma':
            self.fusion_layer = CrossModalityAttention(sp_model.encoder.output_size())
        elif fusion_type == 'mat':
            # multi-head attention
            self.fusion_layer = GateAttentionFusionLayer(d_model=self.txt_model.n_out, audio_hidden_dim=sp_model.encoder.output_size())
        elif fusion_type == 'ctcalign':
            assert isinstance(sp_model, CTCModel)
            self.fusion_layer = CTCAlignFusion(sp_model.encoder.output_size())
        elif fusion_type == 'ctcaligncma':
            assert isinstance(sp_model, CTCModel)
            self.fusion_layer = CTCAlignCMAFusion(sp_model.encoder.output_size())
        self.out_dim = self.fusion_layer.n_out

    def forward(self, 
        speech,
        speech_lengths,
        words,
        text_mask=None,
        chars=None,
        char_mask=None):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        if isinstance(self.txt_model, BertEmbedding):
            txt_repr = self.txt_model(words.unsqueeze(-1))
        else:
            txt_repr = self.txt_model(words)
        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        sp_repr, sp_mask = self.sp_model._forward_encoder(speech, speech_lengths)
        s_max_len = sp_repr.size(1)
        sp_encoder_out_lens = sp_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        # mask = make_pad_mask(encoder_out_lens, s_max_len)
        # fusion
        if self.fusion_type == 'cma':
            mask = ~sp_mask.squeeze(1)
            h = self.fusion_layer(txt_repr, sp_repr, sp_repr, mask)
        elif self.fusion_type == 'mat':
            h = self.fusion_layer(txt_repr, text_mask.float(), sp_repr, sp_mask.squeeze(1).float())
        elif self.fusion_type == 'ctcalign':
            probs = self.sp_model.ctc.probs(sp_repr)
            h = self.fusion_layer(txt_repr, sp_repr, chars, probs, char_mask)
        elif self.fusion_type == 'ctcaligncma':
            probs = self.sp_model.ctc.probs(sp_repr)
            h = self.fusion_layer(txt_repr, sp_repr, chars, probs, char_mask)
        return h, sp_repr, sp_encoder_out_lens

class MultiModalNerModel(nn.Module):
    def __init__(self,
                n_labels,
                encoder,
                sp_weight=0.1):
        super().__init__()
        self.sp_weight = sp_weight
        print('sp_weight:', self.sp_weight)
        self.encoder = encoder
        self.decoder = nn.Sequential(
                        nn.Linear(self.encoder.out_dim, self.encoder.out_dim//2),
                        nn.ReLU(),
                        nn.Linear(self.encoder.out_dim//2, n_labels)
        )
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self,
                speech,
                speech_lengths,
                words,
                mask=None,
                chars=None,
                char_mask=None):
        h, sp_repr, sp_encoder_out_lens = self.encoder(speech, speech_lengths, words, mask, chars, char_mask)
        score = self.decoder(h)
        
        return score, sp_repr, sp_encoder_out_lens

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
        sp_loss = self.encoder.sp_model.loss(s_encoder_out, s_encoder_lens, text, text_lengths)
        return ner_loss + self.sp_weight * sp_loss
    
    def decode(self, score, mask):
        '''
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        '''
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)


def init_mmner_model(configs):
    txt_encoder_type = configs['model_conf']['txt_type']
    if txt_encoder_type == 'bert':
        txt_model = BertEmbedding(model=configs['model_conf']['bert_conf']['bert_path'],
                                  n_layers=configs['model_conf']['bert_conf']['used_layers'],
                                  n_out=configs['model_conf']['bert_conf']['out_dim'],
                                  dropout=configs['model_conf']['bert_conf']['dropout'],
                                  requires_grad=True)
    
    elif txt_encoder_type == 'lstm':
        txt_model = LstmEncoder(n_words=configs['model_conf']['lstm_conf']['n_words'],
                                n_embed=configs['model_conf']['lstm_conf']['n_embed'],
                                n_lstm_hidden=configs['model_conf']['lstm_conf']['n_lstm_hidden'],
                                n_lstm_layers=configs['model_conf']['lstm_conf']['n_lstm_layers'],
                                lstm_dropout=configs['model_conf']['lstm_conf']['lstm_dropout'],
                                char_pad_idx=configs['model_conf']['lstm_conf']['char_pad_idx'])

    sp_encoder_type = configs['model_conf']['speech_type']
    if sp_encoder_type == 'ctc':
        sp_model = init_ctc_model(configs['model_conf']['ctc_conf'])
        pre_train_ctc = configs.get('pre_train_ctc', None)
        if pre_train_ctc is not None:
            print('loading pre-trained ctc model from: '+ configs['pre_train_ctc'] + '\n')
            load_checkpoint(sp_model, configs['pre_train_ctc'])
        else:
            print('without pre-trained ctc model and init a new one \n')
        
    fusion_type = configs['model_conf']['fusion_type']
    encoder = MultiModalModelEncoder(txt_model, sp_model, fusion_type=fusion_type)
    n_labels = configs['num_ner_labels']
    sp_weight = configs.get('sp_weight', 0.1)

    model = MultiModalNerModel(n_labels, encoder, sp_weight)
    return model

    

