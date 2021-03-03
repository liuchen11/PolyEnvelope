import os
import sys
sys.path.insert(0, './')
import argparse
import faulthandler
faulthandler.enable()
import numpy as np

import torch
import torch.nn as nn

from util.io import load_ckpt_model, load_mat_model, load_pth_model
from util.attack import PGM
from util.models import MLP
from util.train import train_test_ibp
from util.dataset import mnist, cifar10
from util.evaluation import AverageCalculator, accuracy
from util.optim_parser import parse_optim
from util.sequence_parser import discrete_seq, continuous_seq
from util.device_parser import parse_device_alloc, config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'Specify the dataset to use, default = "mnist"')
    parser.add_argument('--normalization', type = DictParser, default = None,
        help = 'The normalization applied to the input data, default = None, format: mean=XX,std=XX')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'Batch size, default = 128')
    parser.add_argument('--epochs', type = int, default = 100,
        help = 'Number of epochs, default = 100')

    parser.add_argument('--in_dim', type = int, default = 784,
        help = 'Number of input dimensions, default = 784')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'Number of neurons in the hidden layers, default = []')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'Number of output dimensions, default = 10')
    parser.add_argument('--nonlinearity', type = str, default = 'relu',
        help = 'The activation function, default = "relu"')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')

    parser.add_argument('--optim', action = DictParser,
        default = {'name': 'adam', 'lr': 1e-4},
        help = 'The optimizer, default is name=adam,lr=1e-4')

    parser.add_argument('--alpha', action = DictParser,
        default = {'name': 'constant', 'start_v': 1., 'pt_num': 1000},
        help = 'The weight of base loss v.s. ibp-crown loss, default is name=constant,start_v=1.,pt_num=1000')
    parser.add_argument('--beta', action = DictParser,
        default = {'name': 'constant', 'start_v': 1., 'pt_num': 1000},
        help = 'The weight of ibp bound v.s. ibp-crown bound, default is name=constant,start_v=1.,pt_num=1000')
    parser.add_argument('--eps', action = DictParser,
        default = {'name': 'constant', 'start_v': 1e-1, 'pt_num': 1000},
        help = 'The schedule of the original adversarial budget, default is name=constant,start_v=1e-1,pt_num=1000')
    parser.add_argument('--norm', type = float, default = -1,
        help = 'The norm used in adversarial budget, default = -1, meaning infinity')
    parser.add_argument('--bound_calc_per_batch', type = int, default = None,
        help = 'The number of instances calculated for bound per batch, default = None')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None')
    parser.add_argument('--pixel_range', action = FloatListParser, default = None,
        help = 'Allowable pixel range [min, max], default = None')

    parser.add_argument('--model2load', type = str, default = None,
        help = 'The pretrained model to be loaded, default = None')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU id to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Data set
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader = mnist(batch_size = args.batch_size, batch_size_test = args.batch_size, normalization = args.normalization)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader = cifar10(batch_size = args.batch_size, batch_size_test = args.batch_size, normalization = args.normalization)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model
    model = MLP(in_dim = args.in_dim, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda(device) if use_gpu else model
    criterion = criterion.cuda(device) if use_gpu else criterion

    # load pretrained model
    if args.model2load != None:
        if args.model2load.endswith('.ckpt'):
            model = load_ckpt_model(model = model, name = 'fc1', ckpt_file = args.model2load, device = device, param2load = None)
        elif args.model2load.endswith('.mat'):
            model = load_mat_model(model = model, name = 'fc1', mat_file = args.model2load, device = device, param2load = None)
        elif args.model2load.endswith('.pth'):
            model = load_pth_model(model = model, name = 'fc1', pth_file = args.model2load, device = device, param2load = None)
        else:
            raise ValueError('The format of %s is not supported' % args.model2load)

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    # Parse norms
    norm = args.norm if args.norm > 0 else np.inf

    # Parse parameter
    alpha_list = discrete_seq(**args.alpha)
    beta_list = discrete_seq(**args.beta)
    eps_list = discrete_seq(**args.eps)
    if args.normalization != None:
        for idx in range(len(eps_list)):
            eps_list[idx] = eps_list[idx] / args.normalization['std']

    # Parse pixel_range
    if args.pixel_range != None and args.normalization != None:
        pixel_range = [(args.pixel_range[0] - args.normalization['mean']) / args.normalization['std'], (args.pixel_range[1] - args.normalization['mean']) / args.normalization['std']]
    else:
        pixel_range = args.pixel_range

    # Parser attacker
    attacker = None if args.attack == None else PGM(**args.attack)

    # Prepare the item to save
    configs = {kwarg: value for kwarg, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs,
        'train_loss': {}, 'train_acc': {}, 'test_loss': {}, 'test_acc': {},}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s'%(param, tosave['setup_config'][param]))

    # Train
    train_test_ibp(model = model, train_loader = train_loader, test_loader = test_loader, attacker = attacker, epoch_num = args.epochs,
        optimizer = optimizer, out_folder = args.out_folder, model_name = args.model_name, eps_list = eps_list,
        alpha_list = alpha_list, beta_list = beta_list, norm = norm, device = device,
        criterion = criterion, tosave = tosave, pixel_range = args.pixel_range, bound_calc_per_batch = args.bound_calc_per_batch)
