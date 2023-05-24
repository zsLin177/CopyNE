#!/bin/bash
#SBATCH -J copyloss_ctcatt_st-cmd
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne_st-cmd.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne_st-cmd.%j.error
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH -n 6

source /public/home/gongchen18/anaconda3/bin/activate
conda activate wenet

python -m main train --train data/st-cmd/train_addne.json \
                     --dev data/st-cmd/dev_addne.json \
                     --test data/st-cmd/test_addne.json \
                     --add_context \
                     --pad_context 3 \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --train_ne_dict data/st-cmd/train_ner_minlen2.vocab \
                     --dev_ne_dict data/st-cmd/dev_ner_minlen2.vocab \
                     --no_concat \
                     --path exp/copyne_ctcatt_st-cmd_nocat/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/copyne.yaml \
                     --char_dict data/st-cmd/st-cmd-char.vocab \
                     --cmvn data/st-cmd/global_cmvn \
                     --num_workers 6 \
                     --device 0