import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument('--data', default='data/sick/',
                        help='path to dataset')
    parser.add_argument('--glove', default='/data/tmp/glove/',
                        help='directory with GLOVE embeddings')

    parser.add_argument('--encoder_type', default="TreeLSTM", choices=["TreeLSTM", "LSTM"],
        help='model type to use as sentence encoder')

    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--dropout_prob', default=0, type=float,
                        help='dropout probability')
    parser.add_argument('--rhn_depth', default=0, type=int,
        help='number of additional steps in recurrent highway network')

    # dimensions
    parser.add_argument('--input_dim', type=int, default=300, help="embedding's dimension")
    parser.add_argument('--mem_dim', type=int, default=150, help="LSTM's hidden state (and cell memory) dimension")
    parser.add_argument('--hidden_dim', type=int, default=50, help="'Similarity' FC network hidden state dimension")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of predicted classes")

    parser.add_argument('--cell_m', action='store_true', default=False,
        help='use cell memory as sentence embedding instead of hidden state')
    parser.add_argument('--output_gate', action='store_true', default=True,
        help="use LSTM's output gate if True")

    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=False)

    args = parser.parse_args()
    return args