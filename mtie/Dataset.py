from __future__ import division
import mtie
import torch
from torch.autograd import Variable
from predpatt.util import linear


def splitLabel(words):
    labels = [mtie.Constants.PRED]
    tokens = []
    for word in words:
        if word in linear.PRED_ENC or word in linear.ARGPRED_ENC:
            tokens.append(word)
            labels.append(mtie.Constants.PRED)
        elif word in linear.ARG_ENC:
            tokens.append(word)
            labels.append(mtie.Constants.ARGU)
        elif word == linear.SOMETHING:
            tokens.append(word)
            labels.append(mtie.Constants.ARGU)
        elif word.endswith(linear.HEADER_SUF):
            if word.endswith(linear.PRED_HEADER):
                labels.append(mtie.Constants.PRED)
            else:
                labels.append(mtie.Constants.ARGU)
            word = word.rsplit(':', 1)[0]
            word += ':' + linear.HEADER_SUF
            tokens.append(word)
        else:
            if word.endswith(linear.PRED_SUF):
                labels.append(mtie.Constants.PRED)
            else:
                labels.append(mtie.Constants.ARGU)
            word, label = word.rsplit(":", 1)
            tokens.append(word)
    labels.append(mtie.Constants.PRED)
    return tokens, torch.LongTensor(labels)


class DatasetwithLabel(object):

    def __init__(self, srcData, tgtData, srlData, batchSize, cuda):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            self.srl = srlData
            assert(len(self.src) == len(self.tgt) and len(self.tgt) == len(self.srl))
        else:
            self.tgt = None
            self.srl = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = (len(self.src) + batchSize - 1) // batchSize

    def _batchify(self, data, pad_idx, align_right=False):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(pad_idx)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        out = out.contiguous()
        if self.cuda:
            out = out.cuda()

        v = Variable(out)
        return v

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            mtie.Constants.PAD, align_right=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize], mtie.Constants.PAD)
            srlBatch = self._batchify(
                self.srl[index*self.batchSize:(index+1)*self.batchSize], mtie.Constants.NULL)
        else:
            tgtBatch = None
            srlBatch = None
        return srcBatch, tgtBatch, srlBatch

    def __len__(self):
        return self.numBatches


class Dataset(object):

    def __init__(self, srcData, tgtData, batchSize, cuda):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = (len(self.src) + batchSize - 1) // batchSize

    def _batchify(self, data, align_right=False):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(mtie.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        out = out.t().contiguous()
        if self.cuda:
            out = out.cuda()

        v = Variable(out)
        return v

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize], align_right=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None
        return srcBatch, tgtBatch

    def __len__(self):
        return self.numBatches
