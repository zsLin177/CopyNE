# CopyNE: Better Contextual ASR by Copying Named Entities
This is the repo for CopyNE, a novel approach for contextual ASR. The paper can be found at [here](https://arxiv.org/abs/2305.12839).

## Abstract
End-to-end automatic speech recognition (ASR) systems have made significant progress in general scenarios. 
However, it remains challenging to transcribe contextual named entities (NEs) in the contextual ASR scenario.
Previous approaches have attempted to address this by utilizing the NE dictionary.
These approaches treat entities as individual tokens and generate them token-by-token, which may result in incomplete transcriptions of entities.
In this paper, we treat entities as indivisible wholes and introduce the idea of copying into ASR. 
We design a systematic mechanism called CopyNE, which can copy entities from the NE dictionary.
By copying all tokens of an entity at once, we can reduce errors during entity transcription, ensuring the completeness of the entity. 
Experiments demonstrate that CopyNE consistently improves the accuracy of transcribing entities compared to previous approaches.
Even when based on the strong Whisper, CopyNE still achieves notable improvements.

## Installation
```
pip install -r requirements.txt
```

## Data Preparation
```shell
cd data_to_upload
# download and unzip the the tgz file
wget -c https://us.openslr.org/resources/33/data_aishell.tgz
tar -zxvf data_aishell.tgz
# change the wav path in the json file
cd ..
python scripts/change_wav_path.py data_to_upload/aishell_dataset/train_addne.json data_to_upload
python scripts/change_wav_path.py data_to_upload/aishell_dataset/dev_addne.json data_to_upload
python scripts/change_wav_path.py data_to_upload/aishell_dataset/test_addne.json data_to_upload
python scripts/change_wav_path.py data_to_upload/aishell_dataset/dev-ne.json data_to_upload
python scripts/change_wav_path.py data_to_upload/aishell_dataset/test-ne.json data_to_upload
```

## Training
```
python -m main train --train data_to_upload/aishell_dataset/train_addne.json \
                     --dev data_to_upload/aishell_dataset/dev_addne.json \
                     --test data_to_upload/aishell_dataset/test_addne.json \
                     --add_context \
                     --pad_context 2 \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --train_ne_dict data_to_upload/aishell_vocab/train_ne.vocab \
                     --dev_ne_dict data_to_upload/aishell_vocab/dev_ne.vocab \
                     --path exp/copyne_aishell_beta1/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/copyne.yaml \
                     --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --num_workers 6 \
                     --device 0
```

## Predict and Evaluation
```shell
# predict on the test-ne set
python -m main evaluate --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data_to_upload/aishell_dataset/test-ne.json \
                     --test_ne_dict data_to_upload/aishell_vocab/test_ne.vocab \
                     --path exp/copyne_aishell_beta1/ \
                     --res test-ne.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

# compute CER and NE-CER
./compute-necer.sh data_to_upload/aishell_dataset/test-ne.text test-ne.pred.asr
```

## Acknowledge
1. We borrowed some code from [wenet](https://github.com/wenet-e2e/wenet) for speech processing and modeling.
2. We borrowed some code from [supar](https://github.com/yzhangcs/parser) for the some tensor operations.

## Citation
If you find this repo helpful, please cite the following paper:
```bibtex
@article{zhou2023copyne,
  title={CopyNE: Better Contextual ASR by Copying Named Entities},
  author={Zhou, Shilin and Li, Zhenghua and Hong, Yu and Zhang, Min and Wang, Zhefeng and Huai, Baoxing},
  journal={arXiv preprint arXiv:2305.12839},
  year={2023}
}
```

