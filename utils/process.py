import json
import torch
from collections import Counter
from utils.vocab import Vocab
from supar.utils.fn import pad

def read_json(file_name, if_flat=False):
    """
    json_line:{"sentence": "而此刻日本选手孤山信竟做出无理挑衅在拍照环节", "audio": "/opt/data/private/slzhou/datas/speech/downloads/data_aishell/wav/train/S0224/BAC009S0224W0351.wav", "entity": [[3, 5, "日本", "LOC"], [7, 10, "孤山信", "PER"]], "speaker_info": "F"}
    """
    res = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f.readlines():
            res.append(json.loads(line))

    if if_flat:
        for i in range(len(res)):
            res[i]['entity'] = get_flatted_ner(res[i]['entity'], len(res[i]['sentence']))
    return res

def get_flatted_ner(ner_lsts, char_lens):
    if len(ner_lsts) <= 1:
        return ner_lsts

    spans = {}
    for ner_lst in ner_lsts:
        spans[(ner_lst[0], ner_lst[1] - 1)] = [ner_lst[2], ner_lst[3]]
    if len(spans) <= 1:
        return ner_lsts
    dp = [0] * char_lens
    res = [None] * char_lens
    if (0, 0) in spans:
        dp[0] = 1
        res[0] = -1

    for i in range(1, char_lens):
        tmp_lst = [span_tup for span_tup in spans.keys() if span_tup[1] == i]
        if len(tmp_lst) == 0:
            dp[i] = dp[i-1]
        else:
            max_dpi = 0
            resi = None
            for span_tup in tmp_lst:
                st = span_tup[0]
                if st == 0:
                    max_dpi = span_tup[1] - span_tup[0] + 1
                    resi = -1
                    break
                else:
                    tmp_dpi = dp[st-1] + span_tup[1] - span_tup[0] + 1
                    if tmp_dpi > max_dpi:
                        max_dpi = tmp_dpi
                        resi = st - 1

            if max_dpi > dp[i-1]:
                dp[i] = max_dpi
                res[i] = resi
            else:
                dp[i] = dp[i-1]

    new_ner_lsts = []
    curr = char_lens - 1
    while curr >= 0:
        if res[curr] == None:
            curr -= 1
            continue
        next_curr = res[curr]
        new_ner_lsts.append([next_curr+1, curr+1] + spans[(next_curr+1, curr)])
        curr = next_curr

    return new_ner_lsts

def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table

def read_context_table(context_table_file):
    """
    json file
    """
    with open(context_table_file, 'r', encoding='utf8') as f:
        vocab = json.load(f)
    return vocab

def build_context_tensor(conetxt_vocab, symbol_vocab, pad_value=-1):
    def tokenize(char_vocab, s):
        label = []
        tokens = []
        for ch in s:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        for ch in tokens:
            if ch in char_vocab:
                label.append(char_vocab[ch])
            elif '<unk>' in char_vocab:
                label.append(char_vocab['<unk>'])
            else:
                raise KeyError
        return label

    lst = []
    itos = {v:k for k, v in conetxt_vocab.items()}
    # itos_lst = [itos[i] for i in range(len(itos))]
    for i in range(len(conetxt_vocab)):
        lst.append(torch.tensor(tokenize(symbol_vocab, itos[i])))
    
    return pad(lst, padding_value=pad_value)
    
def build_ner_vocab(data_lst):
    label_counter = Counter(en_lst[3]
                            for dic in data_lst
                            for en_lst in dic['entity']
                            )
    label_vocab = Vocab(label_counter)
    return label_vocab

    