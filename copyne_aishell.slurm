#!/bin/bash
#SBATCH -J copyne_aishell
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne_aishell.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-job-copyne_aishellt.%j.error
#SBATCH --gres=gpu:1
#SBATCH -n 6

source /public/home/gongchen18/anaconda3/bin/activate
conda activate wenet

python -m main train --train data/end2end/train_addne.json \
                     --dev data/end2end/dev_addne.json \
                     --test data/end2end/test_addne.json \
                     --add_context \
                     --pad_context 3 \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --train_ne_dict data/end2end/vocab/aishell_train_ner_minlen2_most-all.vocab \
                     --dev_ne_dict data/end2end/vocab/aishell_dev_ner_minlen2_most-all.vocab \
                     --path exp/copyne_aishell/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/copyne.yaml \
                     --char_dict data/end2end/vocab/raw_plus_ne.vocab \
                     --num_workers 6 \
                     --device 0