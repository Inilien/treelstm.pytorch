import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, sparsity, args):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.dropout_prob = args.dropout_prob

        self.output_gate = args.output_gate

        self.emb = nn.Embedding(vocab_size,in_dim,
                                padding_idx=Constants.PAD,
                                sparse=sparsity)
        self.emb.weight.requires_grad = False

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
        self.drop_rhn = nn.Dropout(self.dropout_prob)

        # Recurrent Highway Networks
        # https://arxiv.org/pdf/1607.03474.pdf

        self.h_rhn_list = [nn.Linear(self.mem_dim,self.mem_dim) for i in range(args.rhn_depth)]

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
            self.drop_rhn = self.drop_rhn.cuda()

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

        if self.output_gate:
            h = F.torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)  # same logic as in the original paper's source code

        for h_rhn in self.h_rhn_list:
            h = h + self.drop_rhn(F.tanh(h_rhn(h)))

        return c,h

    def forward(self, tree_node, inputs):
        # add singleton dimension for future call to node_forward
        embs = F.torch.unsqueeze(self.emb(inputs),1)
        for idx in range(tree_node.num_children):
            _ = self.forward(tree_node.children[idx], inputs)
        child_c, child_h = self.get_child_states(tree_node)
        tree_node.state = self.node_forward(embs[tree_node.idx], child_c, child_h)
        return tree_node.state

    def get_child_states(self, tree_node):
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree_node.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree_node.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree_node.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(tree_node.num_children):
                child_c[idx], child_h[idx] = tree_node.children[idx].state
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
    def __init__(self, encoder_type, cuda, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, args):
        super(SimilarityTreeLSTM, self).__init__()
        self.cudaFlag = cuda

        self.encoder_type = encoder_type

        if self.encoder_type == "TreeLSTM":
            self.encoder = ChildSumTreeLSTM(cuda, vocab_size, in_dim, mem_dim, sparsity, args)
        elif self.encoder_type == "LSTM":
            self.encoder = LSTMEncoder(cuda, vocab_size, in_dim, mem_dim, sparsity, args)
        else:
            raise NotImplementedError()

        self.similarity = Similarity(cuda, mem_dim, hidden_dim, num_classes)

        self.args = args

    def forward(self, ltree, linputs, rtree, rinputs):
        lstate, lhidden = self.encoder(ltree, linputs)
        rstate, rhidden = self.encoder(rtree, rinputs)

        if self.args.cell_m:
            output = self.similarity(lstate, rstate)
        else:
            output = self.similarity(lhidden, rhidden)
        return output


class LSTMEncoder(nn.Module):
    # def __init__(self, ntoken, emb_size, lstm_hid, lstm_num_layers, sim_hid, dropout_prob=0, w2v_embeddings=None):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, sparsity, args):
        super().__init__()
        self.drop = nn.Dropout(args.dropout_prob)

        self.emb = nn.Embedding(vocab_size, in_dim,
                                padding_idx=Constants.PAD,
                                sparse=sparsity)
        self.emb.weight.requires_grad = False

        self.rnn = nn.LSTM(in_dim, mem_dim, 1, dropout=args.dropout_prob)

        self.mem_dim = mem_dim

    def forward(self, tree, inputs):
        embs = F.torch.unsqueeze(self.emb(inputs), 1)

        start_hidden = self.init_hidden(bsz=1)

        _, (end_cell_m, end_hidden) = self.rnn(embs, start_hidden)

        num_layers = end_cell_m.size()[0]

        out_cell_m = end_cell_m[num_layers - 1, :, :]
        out_hidden = end_hidden[num_layers - 1, :, :]

        return out_cell_m, out_hidden

    def init_hidden(self, bsz):
        # print("WARNING: what is this code string doing?")
        weight = next(self.parameters()).data
        return (
            Var(weight.new(1, 1, self.mem_dim).zero_()),
            Var(weight.new(1, 1, self.mem_dim).zero_()))


