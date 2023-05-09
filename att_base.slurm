#!/bin/bash
#SBATCH -J base_att
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-baseatt.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-baseatt.%j.error
#SBATCH --gres=gpu:1
#SBATCH -n 1

source /public/home/zhli13/anaconda3/bin/activate
conda activate wenet

python -m main_ctcattasr train --train data/end2end/train_addne.json \
                     --dev data/end2end/dev_addne.json \
                     --test data/end2end/test_addne.json \
                     --char_dict data/end2end/CLAS_ner_char.vocab \
                     --path exp/debug_attasr/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/attasr.yaml \
                     --num_workers 8 \
                     --device 0
