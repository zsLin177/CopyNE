import json
import sys

def read_json(json_file):
    dic = {}
    with open(json_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            c_dic = json.loads(line)
            key = c_dic['key']
            dic[key] = c_dic
    return dic

def read_wer(wer_file):
    with open(wer_file, 'r', encoding='utf8') as f:
        s = f.read()
        lst = s.split('\n\n')
    return lst[0:-1]


def build_file(json_res, wer_res):
    gold_res_lst = []
    pred_res_lst = []
    for instance_s in wer_res:
        lst = instance_s.strip().split('\n')
        key = lst[0].split(' ')[1]
        gold_s_lst = lst[2].strip().split(' ')[1:]
        pred_s_lst = lst[3].strip().split(' ')[1:]
        ne_lst = json_res[key]['ne_lst']
        gold_s = json_res[key]['txt']

        for i, ne in enumerate(ne_lst, 1):
            n_key = key+'_'+str(i)
            st, ed = ne[1], ne[2]
            this_s = gold_s[st:ed+1]
            gold_res_lst.append((n_key, this_s))


        n_ne_lst = proj_idx(gold_s_lst, ne_lst)

        for i, ne in enumerate(n_ne_lst, 1):
            n_key = key+'_'+str(i)
            st, ed = ne[1], ne[2]
            p_s = ''.join(pred_s_lst[st:ed+1]).replace('__', '')
            pred_res_lst.append((n_key, p_s))

    return gold_res_lst, pred_res_lst
            
def proj_idx(gold_s_lst, gold_ne_lst):
    count = 0
    lst = [0] * len(gold_s_lst)
    for i in range(len(gold_s_lst)):
        lst[i] = count
        if gold_s_lst[i] == '__':
            count += 1
    for i in range(len(lst)-1, -1, -1):
        if gold_s_lst[i] == '__':
            lst.pop(i)
    for ne in gold_ne_lst:
        ne[1] = ne[1] + lst[ne[1]]
        ne[2] = ne[2] + lst[ne[2]]
    return gold_ne_lst

def write(gold_res, pred_res, gold_file, pred_file):
    with open(gold_file, 'w', encoding='utf8') as fg, open(pred_file, 'w', encoding='utf8') as fp:
        for key, res in gold_res:
            fg.write(key+' '+res+'\n')

        for key, res in pred_res:
            fp.write(key+' '+res+'\n')



if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 3
    # json_file = 'test_addne-min_ne_len2.json'
    json_file = args[1]
    json_res = read_json(json_file)
    # wer_file = 'netest-nocat-aishell-cp0.9.pred.asr.wer'
    wer_file = args[2]
    wer_res = read_wer(wer_file)
    gold_res_lst, pred_res_lst = build_file(json_res, wer_res)

    gold_file = 'ne_gold.text'
    pred_file = 'ne_pred.text'
    write(gold_res_lst, pred_res_lst, gold_file, pred_file)