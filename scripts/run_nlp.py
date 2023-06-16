import argparse
from utils.utils import seed_everything


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--loc_eps', type=int, default=1, help="rounds of training")
    parser.add_argument('--glob_eps', type=int, default=40, help="global training round")
    parser.add_argument('--client_num', type=int, default=30, help="number of client")
    parser.add_argument('--client_sample_rate', type=float, default=1, help="client_sample_rate")
    parser.add_argument('--decay_factor', type=float, default=1, help="decay factor of learning rate")
    parser.add_argument('--aggr_method', type=str, default='avg', help='aggregation method')
    parser.add_argument('--fed_dict', type=str, default='all', help='only keys in this will use fed-learning')
    parser.add_argument('--sample_method', type=str, default='sequential', help="method for sampling")
    parser.add_argument('--alpha', type=float, default=0.2, help="alpha for dirichlet distribution")

    # training arguments
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--resume', type=bool, default=False, help="whether to resume")
    parser.add_argument('--start_round', type=int, default=0, help='round to start with')
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--loss', type=str, default='gpt', help='loss type')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')

    # model
    parser.add_argument('--model', type=str, default='gpt', help='model name')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel')
    parser.add_argument('--class_number', type=int, default=200, help='class channel')
    parser.add_argument('--block_size', type=int, default=128, help='block_size for gpt')
    parser.add_argument('--n_layer', type=int, default=3, help='block_size for gpt')
    parser.add_argument('--n_head', type=int, default=6, help='block_size for gpt')
    parser.add_argument('--n_embd', type=int, default=384, help='block_size for gpt')

    # dataset
    parser.add_argument('--dataset', type=str, default='openwebtext', help="name of dataset")
    parser.add_argument('--dataloader_type', type=str, default='nlp', help="type for dataloader")
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--resize', type=int, default=-1, help='resize the input image, -1 means no resizing')

    # logging and evaluation
    parser.add_argument('--test_freq', type=int, default=3, help="rounds of testing")
    parser.add_argument('--logging_path', type=str, default='./logging/test_nlp', help='logging path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from utils.simulator import Simulator
    args = args_parser()
    seed_everything(args.seed)
    simulator = Simulator(args)
    simulator.run()
