#! /usr/bin/env python3
from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import mkl
import numpy as np
import random

import torch.multiprocessing as mp

# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from treenode import TreeNode
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SICKDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer
from utils import map_label_to_target

# MAIN BLOCK
def main():
    global args

    # use global variables to make them available for subprocesses after fork (Unix system assumed)
    global train_dataset
    global dev_dataset
    global test_dataset

    global model
    global grad_locks
    global grad_memory
    global optimizer
    global criterion

    args = parse_args()

    mkl.set_num_threads(1)

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd!=0:
        print('Sparsity and weight decay are incompatible, pick one!')
        exit()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files_a = [os.path.join(split,'a.toks') for split in [train_dir,dev_dir,test_dir]]
    token_files_b = [os.path.join(split,'b.toks') for split in [train_dir,dev_dir,test_dir]]
    token_files = token_files_a+token_files_b
    sick_vocab_file = os.path.join(args.data,'sick.vocab')
    build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    print('==> SICK vocabulary size : %d ' % vocab.size())

    # load SICK dataset splits
    train_file = os.path.join(args.data,'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    print('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data,'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    print('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data,'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, args.num_classes)
        torch.save(test_dataset, test_file)
    print('==> Size of test data    : %d ' % len(test_dataset))

    data = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
            args.encoder_type,
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.hidden_dim, args.num_classes,
                args.sparse,
                args
    )
    criterion = nn.KLDivLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    if args.optim=='adam':
        optimizer   = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim=='sgd':
        optimizer   = optim.SGD(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(),glove_emb.size(1)).normal_(-0.05,0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
            # TODO '<s>', '</s>' these tokens present in glove w2v but probably with different meaning.
            # though they are not currently used
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if word in glove_vocab.labelToIdx.keys():
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.encoder.emb.state_dict()['weight'].copy_(emb)

    model.share_memory() # gradients are allocated lazily, so they are not shared here

    # create shared space to store gradient updates
    grad_memory = dict(
        (name, torch.FloatTensor(*param.size())) for name, param in model.named_parameters() if param.requires_grad)
    for tensor in grad_memory.values():
        tensor.share_memory_()

    # locks will ensure only one process changes gradient values at a time
    grad_locks = dict((name, mp.Lock()) for name in grad_memory)

    # now all subprocesses will have their own copies of model and optimizer with their current states:
    # model has shared weights but do not have any gradient variables (have Nones)
    # optimizer have pointers to those variables(Parameters) of the model
    with mp.Pool() as pool:

        # create trainer object for training and testing
        trainer = Trainer(args, model, criterion, optimizer, pool, train_sample, test_sample, data)

        metric_functions = [metrics.pearson, metrics.mse]

        # init gradients of main process's model here to be able to rewrite underlying data in next section of code.
        # Parameters of subprocesses's models still do not have any gradients (since fork already have been made)
        train_sample(0)

        # replace parameters' gradients of the main process's model with new memory location which was shared earlier
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter.grad.data = grad_memory[name]

        for epoch_id in range(args.epochs):
            trainer.train(epoch_id)
            train_loss, train_pred = trainer.test("train", epoch_id)
            dev_loss, dev_pred     = trainer.test("dev", epoch_id)
            test_loss, test_pred   = trainer.test("test", epoch_id)

            pearson_stats, mse_stats = get_median_and_confidence_interval(
                train_pred, train_dataset.labels, metric_functions)
            print_results("Train", train_loss, pearson_stats, mse_stats)

            pearson_stats, mse_stats = get_median_and_confidence_interval(
                dev_pred, dev_dataset.labels, metric_functions)
            print_results("Dev", dev_loss, pearson_stats, mse_stats)

            pearson_stats, mse_stats = get_median_and_confidence_interval(
                test_pred, test_dataset.labels, metric_functions)
            print_results("Test", test_loss, pearson_stats, mse_stats)

def train_sample(sample_id):
    global args

    global train_dataset

    global model
    global grad_locks
    global grad_memory
    global optimizer
    global criterion

    # ensure that model in this subprocess is in train mode
    model.train()

    # zero gradiets of subprocess's model before computing new ones
    # this will not affect main process's gradients
    optimizer.zero_grad()

    # make a forward pass for the single sample
    ltree, lsent, ltokens, rtree, rsent, rtokens, label = train_dataset[sample_id]
    linput, rinput = Var(lsent), Var(rsent)
    target = Var(map_label_to_target(label, train_dataset.num_classes))
    if args.cuda:
        linput, rinput = linput.cuda(), rinput.cuda()
        target = target.cuda()
    output = model(ltree, linput, rtree, rinput)
    err = criterion(output, target)

    # compute gradients for subprocess's model
    err.backward()

    # update gradients of main process's model
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lock = grad_locks[name]
        lock.acquire()
        try:
            grad_memory[name] += parameter.grad.data
        finally:
            lock.release()


def test_sample():
    pass



def print_results(dataset_name, loss, pearson_stats, mse_stats):
    pearson_median = pearson_stats[1]
    pearson_iqr = pearson_stats[2] - pearson_stats[0]

    mse_median = mse_stats[1]
    mse_iqr = mse_stats[2] - mse_stats[0]

    print('==> {} loss   : {:.6} \t'.format(dataset_name, loss), end="")
    print('{} Pearson    : {:.6} ({:.6}) \t'.format(dataset_name, pearson_median, pearson_iqr), end="")
    print('{} MSE        : {:.6} ({:.6}) \t'.format(dataset_name, mse_median, mse_iqr), end="\n")


def get_median_and_confidence_interval(predictions, targets, metric_functions_list, bootstrap_size = 2000):
    metric_statistics = np.ndarray([len(metric_functions_list), bootstrap_size])

    num_of_samples = predictions.size()[0]
    for bs_i in range(bootstrap_size):
        bs_ids = torch.LongTensor(np.random.choice(range(num_of_samples), num_of_samples))
        bs_predictions = predictions[bs_ids]
        bs_targets = targets[bs_ids]
        for m_i, m_func in enumerate(metric_functions_list):
            metric_statistics[m_i, bs_i] = m_func(bs_predictions, bs_targets)

    metric_statistics = np.percentile(metric_statistics, [25, 50, 75], axis=1).transpose()

    return metric_statistics


if __name__ == "__main__":
    main()