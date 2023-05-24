
for k in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do 
    python -m main evaluate --char_dict data/st-cmd/st-cmd-char.vocab \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/st-cmd/dev_addne.json \
                     --test_ne_dict data/st-cmd/dev_ner_minlen2.vocab \
                     --no_concat \
                     --path exp/copyne_ctcatt_st-cmd_nocat/ \
                     --res dev-nocat-stcmd-cp$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold $k \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

    # ./compute-necer.sh data/st-cmd/dev.text dev-nocat-stcmd-cp$k.pred.asr

    python tools/compute-wer.py --char=1 --v=1 \
        data/st-cmd/dev.text dev-nocat-stcmd-cp$k.pred.asr > dev-nocat-stcmd-cp$k.pred.asr.cer
done
