#!/usr/bin/env python
# encoding: utf-8


from gensim.models import KeyedVectors as word2vec

w2v_path = '/export/ssd/apoliak/GoogleNews-vectors-negative300.bin.gz'
model = word2vec.load_word2vec_format(w2v_path, binary=True)
model.save_word2vec_format('./GoogleNews-vectors-negative300.txt', binary=False)
