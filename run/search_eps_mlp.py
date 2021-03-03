import os
import sys
sys.path.insert(0, './')
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn

from util.models import MLP
from util.io import load_mat_model, load_pth_model
from util.dataset import mnist, cifar10
from util.search_eps import search_eps
from util.device_parser import parse_device_alloc, config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'Specify the dataset to use, default = "mnist"')
    parser.add_argument('--subset', type = str, default = 'test',
        help = 'Specify the subset of the dataset, default = "test"')
    parser.add_argument('--instances', action = DictParser, default = None,
        help = 'Specify the instance range to analyze: min=XX,max=XX, default = None, meaning all data points')

    parser.add_argument('--in_dim', type = int, default = 784,
        help = 'Number of input dimensions, default = 784')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'Number of neurons in the hidden layers, default = []')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'Number of output dimensions, default = 10')
    parser.add_argument('--nonlinearity', type = str, default = 'relu',
        help = 'The activation function, default = "relu"')

    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded')
    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file')

    parser.add_argument('--eps_range', action = DictParser, default = {'min': 0., 'max': 0.4},
        help = 'The minimum and maximum value of epsilon, default = min=0.,max=0.4')
    parser.add_argument('--precision', type = float, default = 0.0001,
        help = 'The precision required for optimal epsilon, default = 0.0001')
    parser.add_argument('--norm', type = float, default = -1,
        help = 'The norm used in adversarial budget, default = -1, meaning infinity')
    parser.add_argument('--bound_est', type = str, default = 'bound_quad',
        help = 'The bounding estimation algorithm, default = "bound_quad"')
    parser.add_argument('--pixel_range', action = FloatListParser, default = None,
        help = 'Allowable pixel range [min, max], default = None')

    parser.add_argument('--certify_mode', type = str, default = 'per',
        help = 'The certification mode, default = "per"')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU id to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Data set
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader = mnist(batch_size = 1, batch_size_test = 1)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader = cifar10(batch_size = 1, batch_size_test = 1)
    else:
        raise ValueError('Unrecognized dataset: %s' % args.dataset)
    assert args.subset in ['train', 'test'], 'Subset tag can be only "train" or "test", but %s found' % args.subset
    data_loader = train_loader if args.subset in ['train',] else test_loader

    # Parse IO
    assert os.path.exists(args.model2load), 'model2load file %s does not exists' % args.model2load
    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Parse model
    model = MLP(in_dim = args.in_dim, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    model = model.cuda(device) if use_gpu else model
    if args.model2load.endswith('.ckpt'):
        ckpt = torch.load(args.model2load)
        model.load_state_dict(ckpt)
    elif args.model2load.endswith('.mat'):
        model = load_mat_model(model = model, name = 'fc1', mat_file = args.model2load, device = device)
    elif args.model2load.endswith('.pth'):
        model = load_pth_model(model = model, name = 'fc1', pth_file = args.model2load, device = device)
    else:
        raise ValueError('The format of %s is not supported' % args.model2load)

    # Parse norm
    norm = args.norm if args.norm > 0 else np.inf

    # Prepare the item to save
    configs = {kwarg: value for kwarg, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'guaranteed_distances': {}}

    # Parse hyper-params
    min_idx = args.instances['min'] if args.instances != None and 'min' in args.instances else None
    max_idx = args.instances['max'] if args.instances != None and 'max' in args.instances else None
    min_eps = args.eps_range['min']
    max_eps = args.eps_range['max']

    # Search for optimal eps
    results = search_eps(model = model, data_loader = data_loader, min_idx = min_idx, max_idx = max_idx,
        min_eps = min_eps, max_eps = max_eps, precision_eps = args.precision, certify_mode = args.certify_mode,
        bound_est = args.bound_est, norm = norm, pixel_range = args.pixel_range, device = device)

    tosave['guaranteed_distances'] = results

    pickle.dump(tosave, open(args.out_file, 'wb'))


