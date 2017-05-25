import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from treenode import TreeNode
from vocab import Vocab
import Constants

# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        a_toks_file = os.path.join(path,'a.toks')
        b_toks_file = os.path.join(path, 'b.toks')
        self.lsentences_tokens = self.store_sentences(a_toks_file)
        self.rsentences_tokens = self.store_sentences(b_toks_file)

        self.lsentences = self.read_sentences(a_toks_file)
        self.rsentences = self.read_sentences(b_toks_file)

        self.ltrees = self.read_trees(os.path.join(path,'a.parents'), a_toks_file)
        self.rtrees = self.read_trees(os.path.join(path,'b.parents'), b_toks_file)

        self.labels = self.read_labels(os.path.join(path,'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        ltokens = deepcopy(self.lsentences_tokens[index])
        rtokens = deepcopy(self.rsentences_tokens[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (
            ltree, lsent, ltokens,
            rtree, rsent, rtokens,
            label)

    def store_sentences(self, filename):
        tokens = []
        with open(filename) as f:
            for line in f.readlines():
                tokens.append(line.strip().split())
        return tokens

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, parents_filename, tokens_filename):
        with open(parents_filename,'r') as p_f:
            with open(tokens_filename, 'r') as t_f:
                trees = [self.read_tree(p, t) for p, t in tqdm(zip(p_f.readlines(), t_f.readlines()))]
        return trees

    def read_tree(self, parent_line, token_line):
        parents = list(map(lambda x: int(x) - 1,parent_line.split()))
        tokens = token_line.strip().split()
        tree_nodes = dict()
        root = None
        for i in range(len(parents)):
            crnt_node_id = i
            if crnt_node_id not in tree_nodes.keys():
                prev_node = None
                while True:
                    if crnt_node_id == -1:
                        break
                    parent_node_id = parents[crnt_node_id]

                    crnt_node = TreeNode()
                    if prev_node is not None:
                        crnt_node.add_child(prev_node)
                    tree_nodes[crnt_node_id] = crnt_node
                    crnt_node.idx = crnt_node_id
                    crnt_node.token = tokens[crnt_node_id]
                    #if trees[parent-1] is not None:
                    if parent_node_id in tree_nodes.keys():
                        tree_nodes[parent_node_id].add_child(crnt_node)
                        break
                    elif parent_node_id == -1:
                        root = crnt_node
                        break
                    else:
                        prev_node = crnt_node
                        crnt_node_id = parent_node_id
        return root

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels