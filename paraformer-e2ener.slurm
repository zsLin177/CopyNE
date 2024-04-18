#!/bin/bash
#SBATCH -J paraformer
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -o /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-paraformer.%j.out
#SBATCH -e /public/home/gongchen18/slzhou/speech_help_ner/asr_help_ner/job-paraformer.%j.error
#SBATCH --gres=gpu:4
#SBATCH -n 24
#SBATCH -t 3-00:00:00

source /public/home/gongchen18/anaconda3/bin/activate
conda activate whisper

torchrun --nnodes=1 --nproc_per_node=4 n_main.py train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/paraformer-ctc-e2ener \
                     --e2e_ner \
                     --batch_size 32 \
                     --seed 777 \
                     --config conf/paraformer.yaml \
                     --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --cmvn data_to_upload/aishell1_global_cmvn_mel80 \
                     --num_workers 12