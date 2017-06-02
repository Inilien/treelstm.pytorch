from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target
import numpy as np
from datetime import datetime

class Trainer(object):
    def __init__(
            self, args, model, criterion, optimizer,
            pool, pool_size,
            train_sample, test_sample,
            data, prediction_storage):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.pool = pool
        self.pool_size = pool_size
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.data = data
        self.prediction_storage = prediction_storage

    # helper function for training
    def train(self, epoch_id):
        dataset = self.data["train"]

        # set main process's model into the train mode.
        # This will not affect subprocesses since they already have been created
        self.model.train()

        indices = torch.randperm(len(dataset))
        for batch_shift in \
                tqdm(range(0, len(dataset), self.args.batchsize), desc='Training epoch {}'.format(epoch_id + 1)):

            # zero grad memory of main process (and also of shared gradient memory in grad_memory dictionary)
            self.optimizer.zero_grad()

            subprocess_tasks = []
            for in_batch_id in range(self.args.batchsize):
                sample_id = indices[in_batch_id + batch_shift]
                task = self.pool.apply_async(self.train_sample, (sample_id,))
                subprocess_tasks.append(task)

            # wait till subprocesses finish their jobs
            [t.get() for t in subprocess_tasks]

            self.optimizer.step()

        return

    # helper function for testing
    def test(self, dataset_name, epoch_id):
        start_time = datetime.now()
        print("Testing epoch {}. Start time: {}. ".format(epoch_id + 1, start_time), end="")

        dataset = self.data[dataset_name]
        predictions = self.prediction_storage[dataset_name]

        self.model.eval()

        predictions.zero_()

        # prepare to share testing load among all processes equally in average
        sample_indices = list(range(len(dataset))) + [None] * (-len(dataset) % self.pool_size)  # note minus

        sample_indices_sliced = np.random.permutation(sample_indices)
        sample_indices_sliced.resize(self.pool_size, len(sample_indices) // self.pool_size)

        subprocess_tasks = []
        for idx in range(sample_indices_sliced.shape[0]):
            subsample_of_indices = sample_indices_sliced[idx]
            task = self.pool.apply_async(self.test_sample, (dataset_name, subsample_of_indices))
            subprocess_tasks.append(task)

        # wait till subprocesses finish their jobs
        loss_list = [t.get() for t in subprocess_tasks]
        loss = sum(loss_list)

        end_time = datetime.now()
        print("End time: {}. Spent time: {}.".format(end_time, end_time - start_time))

        return loss/len(dataset), predictions
