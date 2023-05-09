from utils.process import construct_json

if __name__ == "__main__":
    # pred_file = 'baseline_train.pred.asr'
    # gold_file = 'data/end2end/train.text'
    # new_file = 'tuing_glm_train.json'

    pred_file = 'baseline_dev.pred.asr'
    gold_file = 'data/end2end/dev.text'
    new_file = 'tuing_glm_dev.json'

    construct_json(pred_file, gold_file, new_file, correct_bei=3, max_wrong_num=300)
