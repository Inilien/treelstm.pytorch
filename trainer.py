from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, pool, train_sample, test_sample, data):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.pool = pool
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.data = data

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
        dataset = self.data[dataset_name]

        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1,dataset.num_classes+1)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  {}'.format(epoch_id + 1)):
            ltree, lsent, ltokens, rtree, rsent, rtokens, label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)
            loss += err.data[0]
            predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions
