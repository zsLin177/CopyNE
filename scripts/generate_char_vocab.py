import json

def read_jsonl(file_name):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def generate_char_vocab(dict_lsts):
    char_vocab = {}
    char_vocab['<blank>'] = 0
    char_vocab['<unk>'] = 1
    for dict_lst in dict_lsts:
        sent = dict_lst['txt']
        for char in sent:
            if char not in char_vocab:
                char_vocab[char] = len(char_vocab)
    char_vocab['<sos/eos>'] = len(char_vocab)
    return char_vocab

def write_vocab(char_vocab, file_name):
    with open(file_name, 'w') as f:
        for char, idx in char_vocab.items():
            f.write(f'{char} {idx}\n')

if __name__ == '__main__':
    data = []
    data += read_jsonl('/public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/data/multi/train.jsonl')
    data += read_jsonl('/public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/data/multi/dev.jsonl')
    char_vocab = generate_char_vocab(data)
    write_vocab(char_vocab, '/public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/data/multi/char.vocab')