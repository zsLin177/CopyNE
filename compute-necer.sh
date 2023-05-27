#!/bin/bash

# gold be like data/end2end/test_addne-min_ne_len2.txt
# pred be like netest-nocat-aishell-cp0.9.pred.asr
gold=$1
pred=$2

# first compute cer
python tools/compute-wer.py --char=1 --v=1 --padding-symbol=underline \
        $gold $pred > $pred.cer
echo "CER:"
tail -n 7 $pred.cer
echo "CER computed. Results are in $pred.cer"
echo ""

# delete the last 8 lines of $pred.cer
touch temp.txt
head -n -7 $pred.cer > temp.txt
# echo "Last 7 lines deleted"

# generate ne-gold and ne-pred
filename=$(basename "$gold")
extension="${filename##*.}"
path=${gold%"$filename"}
path_without_extension="$path${filename%.$extension}"
# echo "$path_without_extension"
python build_file_for_ne_cer.py $path_without_extension.json temp.txt
rm temp.txt

# compute ne-cer
python tools/compute-wer.py --char=1 --v=1 \
        ne_gold.text ne_pred.text > $pred.ne-cer
echo "NE-CER:"
tail -n 7 $pred.ne-cer
echo "NE-CER computed. Results are in $pred.ne-cer"