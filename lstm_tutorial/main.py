import argparse
from util import Utils
from data_loader import DataLoader
from evaluator import Evaluator
from timeit import default_timer as timer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for Text Classification')

    # Data Related
    parser.add_argument("--train", dest="train", type=str, default='data/train/')
    parser.add_argument("--test", dest="test", type=str, default='data/test/')
    parser.add_argument("--pte", dest="pte", type=str, default='data/sample_embeddings.txt', help='Pre-trained embeds')

    # Hyper-parameters
    parser.add_argument("--freq_cutoff", dest="freq_cutoff", type=int, default=2)
    parser.add_argument("--emb_dim", dest="emb_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=100)
    parser.add_argument("--filters", dest="filters", type=int, default=100)
    parser.add_argument("--max_epochs", dest="max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=1e-5)

    return parser.parse_args()


def main():
    params = parse_arguments()
    s_t = timer()
    dl = DataLoader(params)
    u = Utils(params, dl)
    evaluator = Evaluator(params, u)
    evaluator.log_time['data_loading'] = timer() - s_t
    evaluator.evaluate_lstm()
    print(evaluator.log_time)
    print("Total time taken (in seconds): {}".format(timer() - s_t))


if __name__ == '__main__':
    main()
