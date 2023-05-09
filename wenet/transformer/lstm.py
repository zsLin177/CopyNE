
import torch
import torch.nn as nn

from supar.modules import VariationalLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supar.modules import SharedDropout

class LstmEncoder(nn.Module):
    def __init__(self,
                n_words,
                n_embed=50,  
                n_lstm_hidden=300, 
                n_lstm_layers=3, 
                lstm_dropout=0.33,
                char_pad_idx=-1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_words,
                                    embedding_dim=n_embed)
        self.n_input = n_embed
        self.lstm = VariationalLSTM(input_size=self.n_input,
                                        hidden_size=n_lstm_hidden,
                                        num_layers=n_lstm_layers,
                                        bidirectional=True,
                                        dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.n_words = n_words
        self.pad_index = char_pad_idx

    def forward(self, words):
        _, seq_len = words.shape
        # default pad_index is -1, map to <eos>
        if self.pad_index == -1:
            pad_mask = words.eq(self.pad_index)
            words.masked_fill_(pad_mask, self.n_words-1)
        mask = words.ne(self.pad_index)

        embed = self.embed(words)
        x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # [batch_size, sqe_len, n_lstm_hidden*2]
        x = self.lstm_dropout(x)
        return x