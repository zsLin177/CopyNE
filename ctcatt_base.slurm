#!/bin/bash
#SBATCH -J base_ctcatt-stcmd
#SBATCH -p long
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-basectcatt-stcmd.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-basectcatt-stcmd.%j.error
#SBATCH --gres=gpu:1
#SBATCH -n 6

source /public/home/gongchen18/anaconda3/bin/activate
conda activate wenet

python -m main_ctcattasr train --train data/st-cmd/train.json \
                     --dev data/st-cmd/dev.json \
                     --test data/st-cmd/test.json \
                     --path exp/ctcattasr_st-cmd/ \
                     --char_dict data/st-cmd/st-cmd-char.vocab \
                     --cmvn data/st-cmd/global_cmvn \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --device 4
