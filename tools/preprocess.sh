#!/bin/bash

# This script generate data for sequence-to-sequence learning. It reads
# bitext, parses the target text into linear predpatt forms,
# and splits data into train/dev/test.
#
# Bitext should have the same corpus name with their language as the file
# suffix (e.g., "zh-en.zh" and "zh-en.en").
#
# Example usage:
#   ./preprocess.sh corpus_dir/zh-en zh en output_dir/
#
#
# Author: Sheng Zhang <zhangsheng.zero@gmail.com>


# Moses
MOSES=$HOME/projects/mosesdecoder

# Parsey variables
PMP_DIR=$HOME/projects/models/syntaxnet
MODEL=$PMP_DIR/../ud-models/English

# Personal scripts
SUTILS=$HOME/ssd/projects/sutils/sutils


SOURCE_CORPUS=$(realpath $1)
SRC_LANG=$2
TGT_LANG=$3
OUTPUT_DIR=$(realpath $4)
CORPUS_NAME=$(basename $1)
CORPUS=${OUTPUT_DIR}/intermediate/${CORPUS_NAME}

mkdir -p ${OUTPUT_DIR}/intermediate


# printf "\nTokenizing...\t`date`\n" >&2
# TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
# $TOKENIZER -l ${SRC_LANG} < ${SOURCE_CORPUS}.${SRC_LANG} > ${CORPUS}.tok.${SRC_LANG}
# $TOKENIZER -l ${TGT_LANG} < ${SOURCE_CORPUS}.${TGT_LANG} > ${CORPUS}.tok.${TGT_LANG}
CORPUS=${CORPUS}.tok

# printf "\nNormalizing punctuation symbols...\t`date`\n" >&2
# cat ${CORPUS}.${SRC_LANG} | perl $SUTILS/normalize/normalize.perl \
#     > ${CORPUS}.norm.${SRC_LANG}
# # If you don't want to do normalization:
# # cp ${CORPUS}.${SRC_LANG} ${CORPUS}.norm.${SRC_LANG}
# cat ${CORPUS}.${TGT_LANG} | perl $SUTILS/normalize/normalize.perl \
#     > ${CORPUS}.norm.${TGT_LANG}
CORPUS=${CORPUS}.norm

# printf "\nParsing...\t`date`\n" >&2
# $SUTILS/parsey/parse.sh ${CORPUS}.${TGT_LANG} ${CORPUS}.conll $PMP_DIR $MODEL

# printf "\nPairing...\t`date`\n" >&2
# python $SUTILS/mt/pair_conll.py -src ${CORPUS}.${SRC_LANG} \
#     -tgt ${CORPUS}.${TGT_LANG} -conll ${CORPUS}.conll \
#     -src_output ${CORPUS}.paired.${SRC_LANG} \
#     -tgt_output ${CORPUS}.paired.${TGT_LANG} \
#     -conll_output ${CORPUS}.paired.conll
CORPUS=${CORPUS}.paired

# printf "\nRunning predpatt...\t`date`\n" >&2
# python $SUTILS/predpatt/run_predpatt_from_conll.py --conll ${CORPUS}.conll \
#     --mode linear > ${CORPUS}.pp


# printf "\nCleaning empty lines...\t`date`\n" >&2
# python $SUTILS/mt/clean.py ${CORPUS}.${SRC_LANG} ${CORPUS}.${TGT_LANG}  \
#     ${CORPUS}.pp    \
#     ${CORPUS}.clean.


printf "\nSplitting data...\t`date`\n" >&2
DATASET_DIR=${OUTPUT_DIR}/dataset
mkdir -p $DATASET_DIR
cp ${CORPUS}.clean.${SRC_LANG} ${DATASET_DIR}/corpus.${SRC_LANG}
cp ${CORPUS}.clean.${TGT_LANG} ${DATASET_DIR}/corpus.${TGT_LANG}
cp ${CORPUS}.clean.pp ${DATASET_DIR}/corpus.pp
python $SUTILS/mt/split.py --data_dir=$DATASET_DIR \
    --langs ${SRC_LANG} ${TGT_LANG} pp --corpus_name=corpus

printf "\ndone\t`date`\n" >&2
