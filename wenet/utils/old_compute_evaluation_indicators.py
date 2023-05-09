import pdb


def compute_matrix_align(align_ner_seq: dict):
    """
    计算预测的p,r,f
    :param align_ner_seq: 对齐的字典
    :return: [[p_nested,r_nested,f1_nested], [p_flat,r_flat,f1_flat]]
    """
    nested_predict_sum_cnt = 0  # TP + FP
    nested_predict_correct_cnt = 0  # TP
    nested_gold_sum_cnt = 0  # TP + FN
    flat_predict_sum_cnt = 0  # TP + FP
    flat_predict_correct_cnt = 0  # TP
    flat_gold_sum_cnt = 0  # TP + FN
    for predict_ner_seq, gold_ner_seq in align_ner_seq.values():
        nested_predict_ne_str = get_entity_by_BILOU_nested_only_valid(predict_ner_seq)
        nested_gold_ne_str = get_entity_by_BILOU_nested_only_valid(gold_ner_seq)

        nested_predict_sum_cnt += nested_predict_ne_str.__len__()
        nested_gold_sum_cnt += nested_gold_ne_str.__len__()
        nested_predict_correct_cnt += set(nested_predict_ne_str).intersection(set(nested_gold_ne_str)).__len__()

        flat_predict_ne_str = get_entity_by_BILOU_flat_only_valid(predict_ner_seq)
        flat_gold_ne_str = get_entity_by_BILOU_flat_only_valid(gold_ner_seq)

        flat_predict_sum_cnt += flat_predict_ne_str.__len__()
        flat_gold_sum_cnt += flat_gold_ne_str.__len__()
        flat_predict_correct_cnt += set(flat_predict_ne_str).intersection(set(flat_gold_ne_str)).__len__()

    p_nested = nested_predict_correct_cnt / nested_predict_sum_cnt
    r_nested = nested_predict_correct_cnt / nested_gold_sum_cnt
    f1_nested = (2 * p_nested * r_nested) / (p_nested + r_nested)

    p_flat = flat_predict_correct_cnt / flat_predict_sum_cnt
    r_flat = flat_predict_correct_cnt / flat_gold_sum_cnt
    f1_flat = (2 * p_flat * r_flat) / (p_flat + r_flat)
    return [[p_nested, r_nested, f1_nested], [p_flat, r_flat, f1_flat]]


def align_files(predict_file, gold_file):
    align_ner_seq = dict()
    for _line in predict_file:
        if _line.strip() == "":
            continue
        _key, _pre_ner_seq = _line.split(' ')
        align_ner_seq[_key] = [eval(_pre_ner_seq.replace("[", "['").replace("]", "']").replace(",", "','")), ]
    for _line in gold_file:
        if _line.strip() == "":
            continue
        gold_line_dict = eval(_line)
        align_ner_seq[gold_line_dict['key']].append(gold_line_dict['ner_seq'])
    return align_ner_seq


def get_entity_by_BILOU_nested_only_valid(_ner_seq: list):
    """
    由ner_seq获取ne entity.
    nested方式
    只抽取其中符合BILOU的ne
    :param _ner_seq: BILOU的序列
    :return: entity列表t
    """
    seq_len = len(_ner_seq)
    _entity = []
    # 只找B-和U-，然后验证是否符合。
    for _idx in range(seq_len):
        if _ner_seq[_idx] == 'O':
            continue
        _tags_list = _ner_seq[_idx].split("|")
        for _idy, _tag in enumerate(_tags_list):
            if _tag[0] == 'B':
                for _idz in range(_idx + 1, seq_len):
                    _tmp_tags_list = _ner_seq[_idz].split("|")
                    if _tmp_tags_list.__len__() <= _idy:
                        break
                    if _tmp_tags_list[_idy][0] == 'I':
                        if _tmp_tags_list[_idy][2:] != _tag[2:]:
                            break
                        else:
                            continue
                    if _tmp_tags_list[_idy][0] == 'L':
                        if _tmp_tags_list[_idy][2:] != _tag[2:]:
                            break
                        _entity.append([_idx, _idz + 1, _tag[2:]])
                        break
                    break
            if _tag[0] == 'U':
                _entity.append([_idx, _idx + 1, _tag[2:]])
    _entity.sort(key=lambda _item: (_item[0], _item[0] - _item[1]))
    _entity_str = [str(_item) for _item in _entity]
    return _entity_str


def get_entity_by_BILOU_flat_only_valid(_ner_seq: list):
    """
    从nested中删除细粒度的
    :param _ner_seq:
    :return:
    """
    _nested_ner_seq = [eval(_) for _ in get_entity_by_BILOU_nested_only_valid(_ner_seq)]
    _nested_ner_seq.sort(key=lambda _item: (_item[0], _item[0] - _item[1]))
    _tmp = [0 for _ in range(_ner_seq.__len__())]
    _del = []
    for ne in _nested_ner_seq:
        if _tmp[ne[0]] != 0:
            _del.append(ne)
            continue
        for _idx in range(ne[0], ne[1]):
            _tmp[_idx] = 1
    _entity = _nested_ner_seq
    for item in _del:
        _entity.remove(item)

    _entity.sort(key=lambda _item: (_item[0], _item[0] - _item[1]))
    _entity_str = [str(_item) for _item in _entity]
    return _entity_str

def call_as_metric(pred_name, gold_name):
    predict_file = open(pred_name)
    gold_file = open(gold_name)
    align_file = align_files(predict_file, gold_file)
    indicators = compute_matrix_align(align_file)
    return indicators[0][2], indicators

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='pred file')
    parser.add_argument('-g', required=True, help='gold file')
    args = parser.parse_args()
    predict_file = open(args.p)
    gold_file = open(args.g)
    # predict_file = open("../中文ner数据处理_aishell/output_aishell_ner/test.list")
    align_file = align_files(predict_file, gold_file)
    indicators = compute_matrix_align(align_file)
    print(f'nested: P: {indicators[0][0]:6.2%} R: {indicators[0][1]:6.2%} F: {indicators[0][2]:6.2%}')
    print(f'  flat: P: {indicators[1][0]:6.2%} R: {indicators[1][1]:6.2%} F: {indicators[1][2]:6.2%}')
