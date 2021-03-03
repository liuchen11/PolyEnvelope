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
from util.models import ConvNet
from util.train import train_test
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
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'Batch size, default = 128')
    parser.add_argument('--update_freq', type = int, default = 1,
        help = 'Update frequency, applicable to large batch, default = 1')
    parser.add_argument('--epochs', type = int, default = 100,
        help = 'Number of epochs, default = 100')

    parser.add_argument('--in_size', type = int, default = 32,
        help = 'The size of the input images, default = 32')
    parser.add_argument('--in_channel', type = int, default = 3,
        help = 'The number of channels of the input images, default = 3')
    parser.add_argument('--conv_kernels', action = IntListParser, default = [3, 3],
        help = 'The kernel size in each convolutional layer, default = [3, 3]')
    parser.add_argument('--conv_strides', action = IntListParser, default = [2, 2],
        help = 'The convolutional strides in each layer, default = [2, 2]')
    parser.add_argument('--conv_channels', action = IntListParser, default = [16, 32],
        help = 'The number of convolutional channels, default = [16, 32]')
    parser.add_argument('--conv_pads', action = IntListParser, default = [1, 1],
        help = 'Padding in each convolutional layer, default = [1, 1]')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [100,],
        help = 'Number of neurons in the hidden layers, default = [100,]')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'The output dimension, default = 10')
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
        default = {'name': 'constant', 'start_v': 1e-2, 'pt_num': 1000},
        help = 'The schedule of regularization normalizer, default is name=constant,start_v=1e-2,pt_num=1000')
    parser.add_argument('--eps', action = DictParser,
        default = {'name': 'constant', 'start_v': 1e-2, 'pt_num': 1000},
        help = 'The schedule of the original adversarial budget, default is name=constant,start_v=1e-2,pt_num=1000')
    parser.add_argument('--gamma', action = DictParser,
        default = {'name': 'constant', 'start_v': 1e-2, 'pt_num': 1000},
        help = 'The schedule of regularization coefficient, default is name=constant,start_v=1e-2,pt_num=1000')
    parser.add_argument('--norm', type = float, default = -1,
        help = 'The norm used in adversarial budget, default = -1, meaning infinity')
    parser.add_argument('--T', type = int, default = 1,
        help = 'To regulaize smallest T distances to linear envelope')
    parser.add_argument('--bound_calc_per_batch', type = int, default = None,
        help = 'The number of instances calculated for bound per batch, default = None')
    parser.add_argument('--at_per', action = BooleanParser, default = False,
        help = 'Whether to apply adversarial training during per regularization, default = False')
    parser.add_argument('--regularize_mode', type = str, default = 'per',
        help = 'The regularization mode, default = "per"')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None')
    parser.add_argument('--bound_est', type = str, default = 'bound_quad',
        help = 'The source of bound estimation, default = "bound_quad"')
    parser.add_argument('--pixel_range', action = FloatListParser, default = None,
        help = 'Allowable pixel range [min, max], default = None')

    parser.add_argument('--model2load', type = str, default = None,
        help = 'The pretrained model to be loaded, default = None')
    parser.add_argument('--load_mode', type = str, default = None,
        help = 'Which part of parameters to be loaded, support value = ["fc", "conv", None], default = None')
    parser.add_argument('--frozen_mode', type = str, default = None,
        help = 'Which part of parameters is frozen, support value = ["fc", "conv", None], default = None')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU id to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Data set
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader = mnist(batch_size = args.batch_size, batch_size_test = args.batch_size)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader = cifar10(batch_size = args.batch_size, batch_size_test = args.batch_size)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model
    model = ConvNet(in_size = args.in_size, in_channel = args.in_channel, conv_kernels = args.conv_kernels, conv_strides = args.conv_strides,
        conv_channels = args.conv_channels, conv_pads = args.conv_pads, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda(device) if use_gpu else model
    criterion = criterion.cuda(device) if use_gpu else criterion

    # load pretrained model
    if args.model2load != None:
        if args.model2load.endswith('.ckpt'):
            model = load_ckpt_model(model = model, name = 'lenet', ckpt_file = args.model2load, device = device, param2load = args.load_mode)
        elif args.model2load.endswith('.mat'):
            model = load_mat_model(model = model, name = 'lenet', mat_file = args.model2load, device = device, param2load = args.load_mode)
        elif args.model2load.endswith('.pth'):
            model = load_pth_model(model = model, name = 'lenet', pth_file = args.model2load, device = device, param2load = args.load_mode)
        else:
            raise ValueError('The format of %s is not supported' % args.model2load)

    # Parse the optimizer
    if args.frozen_mode == None:
        optimizer = parse_optim(policy = args.optim, params = model.parameters())
    elif args.frozen_mode.lower() in ['fc',]:
        for param in model.fc_params():
            param.requires_grad = False
        optimizer = parse_optim(policy = args.optim, params = model.conv_params())
    elif args.frozen_mode.lower() in ['conv',]:
        for param in model.conv_params():
            param.requires_grad = False
        optimizer = parse_optim(policy = args.optim, params = model.fc_params())
    else:
        raise ValueError('Unrecognized frozen mode: %s' % args.frozen_mode)

    # Parse norms
    norm = args.norm if args.norm > 0 else np.inf

    # Parse parameter
    alpha_list = discrete_seq(**args.alpha)
    eps_list = discrete_seq(**args.eps)
    gamma_list = discrete_seq(**args.gamma)

    # Parser attacker
    attacker = None if args.attack == None else PGM(**args.attack)

    # Prepare the item to save
    configs = {kwarg: value for kwarg, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs,
        'train_loss': {}, 'train_acc': {}, 'test_loss': {}, 'test_acc': {},
        'train_safe_distance': {}, 'test_safe_distance': {}, 'guaranteed_distances': []}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s'%(param, tosave['setup_config'][param]))

    # Train
    train_test(model = model, train_loader = train_loader, test_loader = test_loader, attacker = attacker, epoch_num = args.epochs,
        optimizer = optimizer, out_folder = args.out_folder, model_name = args.model_name, bound_est = args.bound_est,
        alpha_list = alpha_list, eps_list = eps_list, gamma_list = gamma_list, T = args.T, norm = norm,
        device = device, criterion = criterion, tosave = tosave, at_per = args.at_per, pixel_range = args.pixel_range,
        update_freq = args.update_freq, bound_calc_per_batch = args.bound_calc_per_batch, regularize_mode = args.regularize_mode)
