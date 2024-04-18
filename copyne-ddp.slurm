#!/bin/bash
#SBATCH -J copyne
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-copyne.%j.error
#SBATCH --gres=gpu:4
#SBATCH -n 24
#SBATCH -t 3-00:00:00

source /public/home/gongchen18/anaconda3/bin/activate
conda activate whisper

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 main.py train --train data/end2end/train_addne.json \
#                      --dev data/end2end/dev_addne.json \
#                      --test data/end2end/test_addne.json \
#                      --add_context \
#                      --pad_context 2 \
#                      --att_type simpleatt \
#                      --add_copy_loss \
#                      --train_ne_dict data_to_upload/aishell_vocab/train_ne.vocab \
#                      --dev_ne_dict data_to_upload/aishell_vocab/dev_ne.vocab \
#                      --path exp/copyne_aishell_beta1-large/ \
#                      --pre_model exp/copyne_aishell_beta1-large/current.model \
#                      --batch_size 32 \
#                      --seed 777 \
#                      --config conf/copyne-conformer.yaml \
#                      --char_dict data_to_upload/aishell_vocab/char.vocab \
#                      --cmvn data_to_upload/aishell1_global_cmvn_mel80 \
#                      --num_workers 6

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 main.py train --train data/st-cmd/train_addne.json \
                     --dev data/st-cmd/dev_addne.json \
                     --test data/st-cmd/test_addne.json \
                     --add_context \
                     --pad_context 3 \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --train_ne_dict data/st-cmd/train_ner_minlen2.vocab \
                     --dev_ne_dict data/st-cmd/dev_ner_minlen2.vocab \
                     --path exp/copyne_st-cmd-large/ \
                     --batch_size 32 \
                     --seed 777 \
                     --config conf/copyne-conformer.yaml \
                     --char_dict data/st-cmd/st-cmd-char.vocab \
                     --cmvn data/st-cmd/global_cmvn \
                     --num_workers 6