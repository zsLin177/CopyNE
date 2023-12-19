# Description: Change the path of the wav files in the json file

import json
import os
import sys

def get_abs_path(path):
    return os.path.abspath(os.path.expanduser(path))

def change_wav_path(jsonl_file, father_abs_path):
    data = []
    with open(jsonl_file, 'r', encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))
    for i in range(len(data)):
        raw_wav_path = data[i]['wav']
        wav_path_lst = raw_wav_path.split('/')[-5:]
        wav_path = os.path.join(father_abs_path, *wav_path_lst)
        data[i]['wav'] = wav_path
    with open(jsonl_file, 'w', encoding="utf8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    jsonl_file = sys.argv[1]
    father_dir_path = sys.argv[2]
    father_abs_path = get_abs_path(father_dir_path)
    change_wav_path(jsonl_file, father_abs_path)

