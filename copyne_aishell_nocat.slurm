#!/bin/bash
#SBATCH -J copyne_aishell_beta0
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne_aishell.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-job-copyne_aishell.%j.error
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH -c 6

source /public/home/gongchen18/anaconda3/bin/activate
conda activate wenet

python -m main train --train data/end2end/train_addne.json \
                     --dev data/end2end/dev_addne.json \
                     --test data/end2end/test_addne.json \
                     --add_context \
                     --pad_context 1 \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --train_ne_dict data/end2end/vocab/aishell_train_ner_minlen2_most-all.vocab \
                     --dev_ne_dict data/end2end/vocab/aishell_dev_ner_minlen2_most-all.vocab \
                     --no_concat \
                     --path exp/copyne_aishell_nocat_beta0/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/copyne.yaml \
                     --char_dict data/end2end/vocab/raw_plus_ne.vocab \
                     --num_workers 6 \
                     --device 0