
for k in -1 1000 2000 3000 4000
do
    # python -m main evaluate --char_dict data/end2end/vocab/CLAS_ner_char.vocab \
    #                  --add_context \
    #                  --att_type simpleatt \
    #                  --add_copy_loss \
    #                  --config conf/copyne.yaml \
    #                  --input data/end2end/dev_addne-min_ne_len2.json \
    #                  --test_ne_dict data/end2end/vocab/dev-plus-train$k.vocab \
    #                  --path exp/copyne_aishell_old/ \
    #                  --res cat-dev-plus$k.pred \
    #                  --decode_mode copy_attention \
    #                  --copy_threshold 0.9 \
    #                  --batch_size 64 \
    #                  --beam_size 10 \
    #                  --device 0
    # ./compute-necer.sh data/end2end/dev_addne-min_ne_len2.txt nocat-ne-dev-plus$k.pred.asr

    # python tools/compute-wer.py --char=1 --v=1 \
    #     data/end2end/dev.text cat-dev-plus$k.pred.asr > cat-dev-plus$k.pred.asr.cer

    python -m main evaluate --char_dict data/st-cmd/st-cmd-char.vocab \
                     --cmvn data/st-cmd/global_cmvn \
                     --add_context \
                     --att_type simpleatt \
                     --add_copy_loss \
                     --config conf/copyne.yaml \
                     --input data/st-cmd/dev_addne.json \
                     --test_ne_dict data/st-cmd/dev-plus-train$k.vocab \
                     --no_concat \
                     --path exp/copyne_ctcatt_st-cmd_nocat/ \
                     --res stcmd_dev-plustrain$k.pred \
                     --decode_mode copy_attention \
                     --copy_threshold 0.9 \
                     --batch_size 64 \
                     --beam_size 10 \
                     --device 0

    python tools/compute-wer.py --char=1 --v=1 --padding-symbol=underline \
        data/st-cmd/dev.text stcmd_dev-plustrain$k.pred.asr > stcmd_dev-plustrain$k.pred.asr.cer

done
