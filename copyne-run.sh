CUDA_VISIBLE_DEVICES=0 python -m main evaluate --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/test_addne.json \
                     --test_ne_dict data_to_upload/aishell_vocab/test_ne.vocab \
                     --path exp/copyne_aishell_beta1/ \
                     --res aishell-test.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

./compute-necer.sh data_to_upload/aishell_dataset/test-ne.text aishell-test.pred.asr

CUDA_VISIBLE_DEVICES=0 python -m main evaluate --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/dev_addne-min_ne_len2.json \
                     --test_ne_dict data_to_upload/aishell_vocab/dev_ne.vocab \
                     --path exp/copyne_aishell_nocat_beta1/ \
                     --no_concat \
                     --res aishell-dev-nocat.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

./compute-necer.sh data_to_upload/aishell_dataset/dev-ne.text aishell-dev-nocat.pred.asr

CUDA_VISIBLE_DEVICES=0 python -m main evaluate --char_dict data_to_upload/aishell_vocab/char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/dev_addne-min_ne_len2.json \
                     --test_ne_dict data_to_upload/aishell_vocab/dev_ne.vocab \
                     --path exp/copyne_aishell_nocat_beta1/ \
                     --no_concat \
                     --res aishell-dev-nocat.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0