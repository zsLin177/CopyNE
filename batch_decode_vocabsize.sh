
for k in 6000 7000 8000 9000 10000
do
    python -m main evaluate --char_dict data/end2end/vocab/CLAS_ner_char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/dev_addne-min_ne_len2.json \
                     --test_ne_dict data/end2end/vocab/dev-plus-train$k.vocab \
                     --path exp/copyne_aishell_old/ \
                     --res ne-dev-plus$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

    python tools/compute-wer.py --char=1 --v=1 \
        data/end2end/dev_addne-min_ne_len2.txt ne-dev-plus$k.pred.asr > ne-dev-plus$k.pred.asr.wer
done
