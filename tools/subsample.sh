dataset=$(realpath $1)
src_lang=$2
tgt_lang=$3
output_root=$(realpath $4)

SUTILS=$HOME/ssd/projects/sutils/sutils

for size in 500000 100000 50000 10000
do
    output=${output_root}/$size
    mkdir -p ${output}
    src=$dataset/corpus.train.${src_lang}
    tgt=$dataset/corpus.train.${tgt_lang}
    pp=$dataset/corpus.train.pp
    src_o=$output/corpus.train.${src_lang}
    tgt_o=$output/corpus.train.${tgt_lang}
    pp_o=$output/corpus.train.pp
    python $SUTILS/mt/subsample.py -src=$src -tgt=$tgt -pp=$pp \
        -src_output=${src_o} -tgt_output=${tgt_o} \
        -pp_output=${pp_o} -size=$size
    cp $dataset/corpus.{dev,test}.{${src_lang},${tgt_lang},pp} $output/
done
