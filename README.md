# Tree-Structured Long Short-Term Memory Networks
A [PyTorch](http://pytorch.org/) based implementation of Tree-LSTM from Kai Sheng Tai's paper
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory
Networks](http://arxiv.org/abs/1503.00075).

## Comment from author of forked repository:
Fixed a lot of bugs and now results are similar to those of the original paper. 

### Requirements
- [PyTorch](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 3 (tested on 3.5)
- (Python 2.7 for data downloading)

### Usage
First run the script `./fetch_and_preprocess.sh`, which downloads:
  - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and 
  [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

The preprocessing script also generates dependency parses of the SICK dataset using the
[Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).

To try the Dependency Tree-LSTM from the paper to predict similarity for pairs of sentences on the SICK dataset, 
run `python main.py` to train and test the model, and have a look at `config.py` for command-line arguments.
**Note:** You may want to change default folder in which GLOVE vectors are expected by the model specifying the path
  `--glove "data/glove"` 

The first run takes a few minutes because the GLOVE embeddings for the words in the SICK vocabulary will need to be read 
and stored to a cache for future runs. In later runs, only the cache is read in during later runs.

This code with default configs gives a Pearson's coefficient of `0.859269` and a MSE of `0.273632`, 
as opposed to a Pearson's coefficient of `0.8676` and a MSE of `0.2532` in the original paper. 
There are several differences in default configs compared to the original paper:
- optimizer: adam vs adagrad
- wd: 0 vs 1e-4
- lr: 1e-3 vs 0.1

### Notes
PyTorch 0.1.12 has support for sparse tensors in both CPU and GPU modes. This means that `nn.Embedding` can now have sparse updates, potentially reducing memory usage. Enable this by the `--sparse` argument, but be warned of two things:

- Sparse training has not been tested by me. The code works, but performance has not been benchmarked for this code.
- Weight decay does not work with sparse gradients/parameters.

### Acknowledgements
Shout-out to [Kai Sheng Tai](https://github.com/kaishengtai/) for the [original LuaTorch implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.

### Original author
[Riddhiman Dasgupta](https://researchweb.iiit.ac.in/~riddhiman.dasgupta/)

*This is my first PyTorch based implementation, and might contain bugs. Please let me know if you find any!*

### License
MIT