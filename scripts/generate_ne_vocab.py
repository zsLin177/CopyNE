import json

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def generate_ne_vocab(data, filename):
    ne_vocab = {}
    for dic in data:
        sent = dic['txt']
        for ne in dic['ne_lst']:
            s = sent[ne[1]:ne[2]+1]
            if s not in ne_vocab and len(s) > 1:
                ne_vocab[s] = len(ne_vocab)
    with open(filename, 'w') as f:
        json.dump(ne_vocab, f, ensure_ascii=False)

if __name__ == "__main__":
    train = read_jsonl('data/multi/train_pred_normal.jsonl')
    dev = read_jsonl('data/multi/dev_plus_aishell1dev.jsonl')
    generate_ne_vocab(train, 'data/multi/train_ne.vocab')
    generate_ne_vocab(dev, 'data/multi/dev_ne.vocab')