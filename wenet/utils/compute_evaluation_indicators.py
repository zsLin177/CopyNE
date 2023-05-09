import pdb


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BILOU(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'L-'
    single_label = 'U-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)
    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)
    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def get_flat_ner_fmeasure(golden_lists, predict_lists, label_type="BILOU"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BILOU":
            gold_matrix = get_ner_BILOU(golden_list)
            pred_matrix = get_ner_BILOU(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))

        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def get_nested_ner_fmeasure(golden_lists, predict_lists, label_type="BILOU"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        splited_gold_list = split_nested_tag(golden_list)
        splited_pred_list = split_nested_tag(predict_list)
        gold_matrix = []
        pred_matrix = []
        for i in range(len(splited_gold_list[0])):
            golden_layer_i = [ele[i] for ele in splited_gold_list]
            if label_type == "BILOU":
                gold_matrix = gold_matrix + get_ner_BILOU(golden_layer_i)
            else:
                gold_matrix = gold_matrix + get_ner_BIO(golden_layer_i)
        for j in range(len(splited_pred_list[0])):
            pred_layer_j = [ele[j] for ele in splited_pred_list]
            if label_type == "BILOU":
                pred_matrix = pred_matrix + get_ner_BILOU(pred_layer_j)
            else:
                pred_matrix = pred_matrix + get_ner_BIO(pred_layer_j)
        gold_matrix = list(set(gold_matrix))
        pred_matrix = list(set(pred_matrix))
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        # print(gold_matrix)
        # print(pred_matrix)
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    # print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def split_nested_tag(label_list):
    label_list = [ele.split("|") for ele in label_list]
    max_tag_layer = max([len(ele) for ele in label_list])
    label_list = [pad(ele, max_tag_layer) for ele in label_list]
    return label_list


def pad(ele, max_tag_layer):
    while len(ele) < max_tag_layer:
        ele.append("O")
    return ele


def get_flat_ner(golden_lists, predict_lists, label_type="BILOU"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BILOU":
            gold_matrix = get_ner_BILOU(golden_list)
            pred_matrix = get_ner_BILOU(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))

        golden_full.append(gold_matrix)
        predict_full.append(pred_matrix)
        right_full.append(right_ner)

    return golden_full, predict_full, right_full


def get_nested_ner(golden_lists, predict_lists, label_type="BILOU"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    # breakpoint()
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        splited_gold_list = split_nested_tag(golden_list)
        splited_pred_list = split_nested_tag(predict_list)
        gold_matrix = []
        pred_matrix = []
        for i in range(len(splited_gold_list[0])):
            golden_layer_i = [ele[i] for ele in splited_gold_list]
            if label_type == "BILOU":
                gold_matrix = gold_matrix + get_ner_BILOU(golden_layer_i)
            else:
                gold_matrix = gold_matrix + get_ner_BIO(golden_layer_i)
        for j in range(len(splited_pred_list[0])):
            pred_layer_j = [ele[j] for ele in splited_pred_list]
            if label_type == "BILOU":
                pred_matrix = pred_matrix + get_ner_BILOU(pred_layer_j)
            else:
                pred_matrix = pred_matrix + get_ner_BIO(pred_layer_j)
        gold_matrix = list(set(gold_matrix))
        pred_matrix = list(set(pred_matrix))
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        # print(gold_matrix)
        # print(pred_matrix)
        golden_full.append(gold_matrix)
        predict_full.append(pred_matrix)
        right_full.append(right_ner)

    return golden_full, predict_full, right_full


def compute_matrix_align(align_ner_seq: dict, dataset_type: str = "nested"):
    """
    计算预测的p,r,f
    dataset_type : nested or flat
    :param align_ner_seq: 对齐的字典
    :return: [p,r,f]
    """
    predict_bilou_list = []
    gold_bilou_list = []
    for predict_ner_seq, gold_ner_seq in align_ner_seq.values():
        predict_bilou_list.append(predict_ner_seq)
        gold_bilou_list.append(gold_ner_seq)
    if dataset_type == 'nested':
        accuracy, precision, recall, f_measure = get_nested_ner_fmeasure(golden_lists=gold_bilou_list,
                                                                         predict_lists=predict_bilou_list)
    if dataset_type == 'flat':
        accuracy, precision, recall, f_measure = get_flat_ner_fmeasure(golden_lists=gold_bilou_list,
                                                                       predict_lists=predict_bilou_list)
    return accuracy, precision, recall, f_measure


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


def call_as_metric(pred_name, gold_name, dataset_type="nested"):
    predict_file = open(pred_name)
    gold_file = open(gold_name)
    align_file = align_files(predict_file, gold_file)
    indicators = compute_matrix_align(align_file, dataset_type)
    # indicators : accuracy, precision, recall, f_measure
    return indicators[-1], indicators[1:]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='pred file')
    parser.add_argument('-g', required=True, help='gold file')
    args = parser.parse_args()
    predict_file = open(args.p)
    gold_file = open(args.g)
    align_file = align_files(predict_file, gold_file)
    dataset_type = 'flat'
    indicators = compute_matrix_align(align_file, dataset_type=dataset_type)
    print(
        f'{dataset_type:6}: ACC:  {indicators[0]:6.2%}  P: {indicators[1]:6.2%} R: {indicators[2]:6.2%} F: {indicators[3]:6.2%}')
