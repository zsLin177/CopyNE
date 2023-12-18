
for k in 0.85 0.9 0.95 0.99
do
    python -m main evaluate --char_dict data/end2end/vocab/CLAS_ner_char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/dev_addne-min_ne_len2.json \
                     --test_ne_dict data/end2end/vocab/dev-ne_iv$k.vocab \
                     --path exp/copyne_aishell_old/ \
                     --res results-oov/dev-ne_iv$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0
    ./compute-necer.sh data/end2end/dev_addne-min_ne_len2.txt results-oov/dev-ne_iv$k.pred.asr

    python -m main evaluate --char_dict data/end2end/vocab/CLAS_ner_char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/test_addne-min_ne_len2.json \
                     --test_ne_dict data/end2end/vocab/test-ne_iv$k.vocab \
                     --path exp/copyne_aishell_old/ \
                     --res results-oov/test-ne_iv$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

    ./compute-necer.sh data/end2end/test_addne-min_ne_len2.txt results-oov/test-ne_iv$k.pred.asr

done
