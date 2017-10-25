# Cross-lingual Open IE Project

## Prerequisites
Python 2.7

[PyTorch](http://pytorch.org/) 0.1.12 (`conda install pytorch=0.1.12 cuda80 -c soumith`)

Cuda 8.0

[PredPatt](https://github.com/hltcoe/PredPatt)

## Demo
To run the `demo`,

1. Put your `man-eng` data_dir in `./data/`:
```
ln -s ${man-eng} ./data/man-eng
```

2. Download the GloVe Vectors [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip):
```
wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.zip ./data/word_vecs/eng.txt
```

3. Run the `demo`:
```
./demo
```
The BLEU score on the test set should be around 22 in greedy decoding.
