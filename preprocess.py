from __future__ import print_function
import sys
import mtie

import argparse
import torch

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the test source data")
parser.add_argument('-test_tgt', required=True,
                    help="Path to the test target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-load_src_emb', default=1, type=int,
                    help="Load pretrained embedding on the source side.")
parser.add_argument('-pre_word_vecs_enc',
                    help="Path to an existing pretrained word vectors enc")
parser.add_argument('-load_tgt_emb', default=1, type=int,
                    help="Load pretrained embedding on the source side.")
parser.add_argument('-pre_word_vecs_dec',
                    help="Path to an existing pretrained word vectors dec")

parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

mtie.utils.set_seed(opt.seed)
log = mtie.utils.get_logging()


def makeVocabulary(filenames, size):
    vocab = mtie.Dict([mtie.Constants.PAD_WORD, mtie.Constants.UNK_WORD,
                       mtie.Constants.BOS_WORD, mtie.Constants.EOS_WORD])

    for filename in filenames:
        with open(filename) as f:
            for sent in f.readlines():
                for word in sent.split():
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    log.info('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        log.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = mtie.Dict()
        vocab.loadFile(vocabFile)
        log.info('Loaded ' + vocab.size() + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        log.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    log.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    log.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        srcWords = srcF.readline().split()
        tgtWords = tgtF.readline().split()

        if not srcWords or not tgtWords:
            if srcWords and not tgtWords or not srcWords and tgtWords:
                log.info('WARNING: source and target do not have the same number of sentences')
            break

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:

            src += [srcDicts.convertToIdx(srcWords,
                                          mtie.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          mtie.Constants.UNK_WORD,
                                          mtie.Constants.BOS_WORD,
                                          mtie.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            log.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        log.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    log.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    log.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt


def is_labeled(token):
    return (token.endswith(":a") or token.endswith(":a_h") or token.endswith(":p")
            or token.endswith(":p_h"))


def make_word_vecs(filepath, vocab, pp=False):
    token2vec = {}
    log.info("Loading pretrained vecs from %s" %filepath)
    progress_bar = mtie.ProgressBar("Loading", input_file=filepath)
    progress_bar.start_progress_bar()
    for i, line in enumerate(open(filepath), 1):
        progress_bar.increment_step()
        fields = line.strip().split()
        if i == 1 and len(fields) == 2:
            continue
        token = fields[0]
        vec = list(map(float, fields[1:]))
        token2vec[token] = torch.Tensor(vec)
    progress_bar.stop_progress_bar()
    if "<unk>" in token2vec:
        unk_vec = token2vec["<unk>"]
    elif "unk" in token2vec:
        unk_vec = token2vec["unk"]
    else:
        unk_vec = token2vec.values()[0]
        unk_vec = torch.zeros(unk_vec.size()).uniform_(-opt.param_init, opt.param_init)
    unk_count = 0
    ret = []
    for idx in xrange(vocab.size()):
        token = vocab.idxToLabel[idx]
        if token == mtie.Constants.PAD_WORD:
            ret.append(torch.zeros(unk_vec.size()))
            continue
        if token == mtie.Constants.UNK_WORD:
            ret.append(unk_vec)
            continue
        if not pp and token in token2vec:
            vec = token2vec[token]
        elif pp and is_labeled(token) and token.rsplit(":", 1)[0] in token2vec:
            vec = token2vec[token.rsplit(":", 1)[0]]
        else:
            unk_count += 1
            vec = torch.zeros(unk_vec.size()).uniform_(-opt.param_init, opt.param_init)
        ret.append(vec)
    ret = torch.stack(ret)
    log.info("Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    log.info("OOV words %d" %unk_count)
    return ret


def main():

    dicts = {}
    data_files_src = [opt.train_src, opt.valid_src, opt.test_src]
    data_files_tgt = [opt.train_tgt, opt.valid_tgt, opt.test_tgt]
    dicts['src'] = initVocabulary('source', data_files_src, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = initVocabulary('target', data_files_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)

    log.info('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    log.info('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'], dicts['tgt'])
    sys.stdout.flush()

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    if opt.load_src_emb == 1 and opt.pre_word_vecs_enc:
        word_vecs = make_word_vecs(opt.pre_word_vecs_enc, dicts['src'])
        torch.save(word_vecs, opt.save_data + ".word_vecs.enc")
    if opt.load_tgt_emb == 1 and opt.pre_word_vecs_dec:
        word_vecs = make_word_vecs(opt.pre_word_vecs_dec, dicts['tgt'], True)
        torch.save(word_vecs, opt.save_data + ".word_vecs.dec")


    log.info('Saving data to \'' + opt.save_data + '-train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '-train.pt')


if __name__ == "__main__":
    main()
