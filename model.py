import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, sparsity, dropout_prob, rhn_depth):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.dropout_prob = dropout_prob

        self.emb = nn.Embedding(vocab_size,in_dim,
                                padding_idx=Constants.PAD,
                                sparse=sparsity)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fx = nn.Linear(self.in_dim,self.mem_dim)
        self.fh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        # Recurrent Dropout without Memory Loss
        # https://arxiv.org/pdf/1603.05118.pdf
        self.drop_forward_inputs = nn.Dropout(self.dropout_prob)
        self.drop_forward_child_h = nn.Dropout(self.dropout_prob)
        self.drop_recurrent = nn.Dropout(self.dropout_prob)

        # Recurrent Highway Networks
        # https://arxiv.org/pdf/1607.03474.pdf

        self.h_rhn_list = [nn.Linear(self.mem_dim,self.mem_dim) for i in range(rhn_depth)]

        if self.cudaFlag:
            self.ix = self.ix.cuda()
            self.ih = self.ih.cuda()

            self.fx = self.fx.cuda()
            self.fh = self.fh.cuda()

            self.ox = self.ox.cuda()
            self.oh = self.oh.cuda()

            self.ux = self.ux.cuda()
            self.uh = self.uh.cuda()

            self.drop_forward_inputs = self.drop_forward_inputs.cuda()
            self.drop_forward_child_h = self.drop_forward_child_h.cuda()
            self.drop_recurrent = self.drop_recurrent.cuda()

            self.h_rhn_list = [h_rhn.cuda() for h_rhn in self.h_rhn_list]

    def node_forward(self, inputs, child_c, child_h):
        inputs = self.drop_forward_inputs(inputs)
        child_h = self.drop_forward_child_h(child_h)

        child_h_sum = F.torch.sum(torch.squeeze(child_h,1),0)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        u = self.drop_recurrent(u)

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs),1)
        f = F.torch.cat([self.fh(child_hi)+fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f,1)
        fc = F.torch.squeeze(F.torch.mul(f,child_c),1)

        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o, F.tanh(c))
        # h = F.tanh(c)  # same logic as in the original paper's source code

        for h_rhn in self.h_rhn_list:
            h = h + F.tanh(h_rhn(h))

        return c,h

    def forward(self, tree, inputs):
        # add singleton dimension for future call to node_forward
        embs = F.torch.unsqueeze(self.emb(inputs),1)
        for idx in range(tree.num_children):
            _ = self.forward(tree.children[idx], inputs)
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1], child_c, child_h)
        return tree.state

    def get_child_states(self, tree):
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h

# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2*self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = F.torch.mul(lvec, rvec)
        abs_dist = F.torch.abs(F.torch.add(lvec,-rvec))
        vec_dist = F.torch.cat((mult_dist, abs_dist),1)
        out = F.sigmoid(self.wh(vec_dist))
        # out = F.sigmoid(out)
        out = F.log_softmax(self.wp(out))
        return out

# puttinh the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, dropout_prob, rhn_depth):
        super(SimilarityTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.childsumtreelstm = ChildSumTreeLSTM(cuda, vocab_size, in_dim, mem_dim, sparsity, dropout_prob, rhn_depth)
        self.similarity = Similarity(cuda, mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output
