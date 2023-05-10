
for k in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python -m main evaluate --char_dict data/end2end/vocab/CLAS_ner_char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/end2end/dev_addne.json \
                     --test_ne_dict data/end2end/vocab/aishell_dev_ner_minlen2_most-all.vocab \
                     --path exp/copyne_aishell_old/ \
                     --res dev-cp$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold $k \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

    python tools/compute-wer.py --char=1 --v=1 \
        data/end2end/dev.text dev-cp$k.pred.asr > dev-cp$k.pred.asr.wer
done
