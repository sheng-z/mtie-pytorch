import mtie
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=300,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=9,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-load_src_emb', default=1, type=int,
                    help="Load pretrained embedding on the source side.")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-load_tgt_emb', default=1, type=int,
                    help="Load pretrained embedding on the source side.")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-update_emb', default=1, type=int,
                    help="Update embeddings during training.")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

opt = parser.parse_args()
opt.cuda = len(opt.gpus)

mtie.utils.set_seed(opt.seed)
log = mtie.utils.get_logging()

log.info(opt)


def Criterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[mtie.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()

    batch_size = outputs.size(0)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets.contiguous(), opt.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        pred_t = generator(out_t)
        loss_t = crit(pred_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0

    model.eval()
    for i in range(len(data)):
        batch = [x.transpose(0, 1) for x in data[i]] # must be batch first for gather/scatter in DataParallel
        outputs = model(batch)  # FIXME volatile
        targets = batch[1][:, 1:]  # exclude <s> from targets
        loss, _ = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_words += targets.data.ne(mtie.Constants.PAD).sum()

    model.train()
    return total_loss / total_words


def trainModel(model, trainData, validData, dataset, optim):
    log.info(model)
    model.train()

    # define criterion of each GPU
    criterion = Criterion(dataset['dicts']['tgt'].size())

    start_time = time.time()
    def trainEpoch(epoch):

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]
            batch = [x.transpose(0, 1) for x in batch] # must be batch first for gather/scatter in DataParallel

            model.zero_grad()
            outputs = model(batch)
            targets = batch[1][:, 1:]  # exclude <s> from targets
            loss, gradOutput = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # update the parameters
            grad_norm = optim.step()

            report_loss += loss
            total_loss += loss
            num_words = targets.data.ne(mtie.Constants.PAD).sum()
            total_words += num_words
            report_words += num_words
            if i % opt.log_interval == 0 and i > 0:
                loss_per_word = report_loss / report_words
                log.info("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s; %6.0f s elapsed" %
                      (epoch, i, len(trainData),
                      math.exp(loss_per_word) if loss_per_word < 300 else float("inf"), # report_loss / report_words),
                      report_words/(time.time()-start),
                      time.time()-start_time))

                report_loss = report_words = 0
                start = time.time()

        return total_loss / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        log.info('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        log.info('Train perplexity: %g' % math.exp(min(train_loss, 100)))

        #  (2) evaluate on the validation set
        valid_loss = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        log.info('Validation perplexity: %g' % valid_ppl)

        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            optim.updateLearningRate(valid_loss, epoch)

        #  (4) drop a checkpoint
        checkpoint = {
            'model': model,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim,
        }
        if epoch == opt.epochs:
            torch.save(checkpoint,
                       '%s_e%d_%.2f.pt' % (opt.save_model, epoch, valid_ppl))


def load_pretrained_params(weight, word_vecs):
    log.info("emb_size:" + str(weight.size()))
    word_vecs = torch.load(word_vecs)
    log.info("loaded_size:" + str(word_vecs.size()))
    weight.copy_(word_vecs)


def main():

    log.info("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = mtie.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = mtie.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda)

    dicts = dataset['dicts']
    log.info(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    log.info(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    log.info(' * maximum batch size. %d' % opt.batch_size)

    log.info('Building model...')

    if opt.train_from is None:
        encoder = mtie.Models.Encoder(opt, dicts['src'])
        decoder = mtie.Models.Decoder(opt, dicts['tgt'])
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.cuda > 1:
            generator = nn.DataParallel(generator, device_ids=opt.gpus)
        model = mtie.Models.Model(encoder, decoder, generator)
        if opt.cuda > 1:
            model = nn.DataParallel(model, device_ids=opt.gpus)
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()

        model.generator = generator

        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        if opt.load_src_emb == 1 and opt.pre_word_vecs_enc:
            log.info("Loading pretrained word vecs from %s" %opt.pre_word_vecs_enc)
            weight = model.encoder.word_lut.weight.data
            load_pretrained_params(weight, opt.pre_word_vecs_enc)
            if opt.update_emb == 0:
                model.encoder.word_lut.weight.requires_grad = False
        if opt.load_tgt_emb == 1 and opt.pre_word_vecs_dec:
            log.info("Loading pretrained word vecs from %s" %opt.pre_word_vecs_dec)
            weight = model.decoder.word_lut.weight.data
            load_pretrained_params(weight, opt.pre_word_vecs_dec)
            if opt.update_emb == 0:
                model.decoder.word_lut.weight.requires_grad = False

        optim = mtie.Optim(
            filter(lambda p: p.requires_grad, model.parameters()),
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        log.info('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        model = checkpoint['model']
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()
        optim = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1

    nParams = sum([p.nelement() for p in model.parameters()])
    log.info('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
