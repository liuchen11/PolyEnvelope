import os
import sys
sys.path.insert(0, './')
import time
import argparse
import faulthandler
faulthandler.enable()
import numpy as np

import torch
import torch.nn as nn

from util.models import ConvNet
from util.io import load_mat_model, load_pth_model
from util.dataset import mnist, cifar10
from util.certify import certify_per, certify_crown, certify_kw, certify_pgd, certify_ibp
from util.device_parser import parse_device_alloc, config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'Specify the dataset to use, default = "mnist"')
    parser.add_argument('--subset', type = str, default = 'test',
        help = 'Specify the subset of the dataset, default = "test"')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'Batch size, default = 128')

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

    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded')
    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file')

    parser.add_argument('--eps', type = float, default = 0.1,
        help = 'The pre-defined adversarial budget, default = 0.1')
    parser.add_argument('--norm', type = float, default = -1,
        help = 'The norm used in adversarial budget, default = -1, meaning infinity')
    parser.add_argument('--bound_est', type = str, default = 'bound_quad',
        help = 'The bound estimation algorithm, default = "bound_quad"')
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
        train_loader, test_loader = mnist(batch_size = args.batch_size, batch_size_test = args.batch_size)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader = cifar10(batch_size = args.batch_size, batch_size_test = args.batch_size)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)
    assert args.subset in ['train', 'test'], 'Subset tag can be only "train" or "test", but %s found' % args.subset
    data_loader = train_loader if args.subset in ['train',] else test_loader

    # Parse IO
    assert os.path.exists(args.model2load), 'model2load file %s does not exists' % args.model2load
    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Parse model
    model = ConvNet(in_size = args.in_size, in_channel = args.in_channel, conv_kernels = args.conv_kernels, conv_strides = args.conv_strides,
        conv_channels = args.conv_channels, conv_pads = args.conv_pads, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    model = model.cuda(device) if use_gpu else model
    if args.model2load.endswith('.ckpt'):
        ckpt = torch.load(args.model2load)
        model.load_state_dict(ckpt)
    elif args.model2load.endswith('.mat'):
        model = load_mat_model(model = model, name = 'lenet', mat_file = args.model2load, device = device)
    elif args.model2load.endswith('.pth'):
        model = load_pth_model(model = model, name = 'lenet', pth_file = args.model2load, device = device)
    else:
        raise ValueError('The format of %s is not supported' % args.model2load)

    # Parse norm
    norm = args.norm if args.norm > 0 else np.inf

    # Prepare the item to save
    configs = {kwarg: value for kwarg, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'guaranteed_distances': []}

    start_time = time.time()
    if args.certify_mode.lower() in ['per',]:
        certify_per(model = model, data_loader = data_loader, out_file = args.out_file, eps = args.eps,
            norm = norm, bound_est = args.bound_est, device = device, tosave = tosave, pixel_range = args.pixel_range)
    elif args.certify_mode.lower() in ['crown', 'lin']:
        certify_crown(model = model, data_loader = data_loader, out_file = args.out_file, eps = args.eps,
            norm = norm, bound_est = args.bound_est, device = device, tosave = tosave, pixel_range = args.pixel_range)
    elif args.certify_mode.lower() in ['kw',]:
        certify_kw(model = model, data_loader = data_loader, out_file = args.out_file, eps = args.eps,
            norm = norm, bound_est = args.bound_est, device = device, tosave = tosave, pixel_range = args.pixel_range)
    elif args.certify_mode.lower() in ['pgd',]:
        certify_pgd(model = model, data_loader = data_loader, out_file = args.out_file, eps = args.eps,
            norm = norm, bound_est = args.bound_est, device = device, tosave = tosave, pixel_range = args.pixel_range)
    elif args.certify_mode.lower() in ['ibp',]:
        certify_ibp(model = model, data_loader = data_loader, out_file = args.out_file, eps = args.eps,
            norm = norm, bound_est = args.bound_est, device = device, tosave = tosave, pixel_range = args.pixel_range)
    else:
        raise ValueError('Unrecognized Cerification Mode: %s' % args.ceritfy_mode)
    end_time = time.time()

    print('Time Cost: %.4f seconds' % (end_time - start_time))

