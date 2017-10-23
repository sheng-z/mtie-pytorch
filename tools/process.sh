#!/bin/bash

DATA_DIR=$1
OUTPUT_DIR=$2

SCRIPT=$HOME/ssd/projects/lrlp/experiments/script/preprocess.sh

for filename in `ls $DATA_DIR`
do
    if [[ $filename == *"-"*"."eng ]]; then
        corpus=${filename%.*}
        source_lang=${corpus%-*}
        target_lang=${corpus#*-}
        echo "$filename"
        mkdir -p $OUTPUT_DIR/$corpus
        if [ -d "$OUTPUT_DIR/$corpus" ]; then
            printf "\n=========================\nProcessing: $corpus\n"
            # mkdir -p $OUTPUT_DIR/$corpus
            $SCRIPT $DATA_DIR/$corpus $source_lang $target_lang $OUTPUT_DIR/$corpus
            printf "\n\n"
        fi
    fi
done
