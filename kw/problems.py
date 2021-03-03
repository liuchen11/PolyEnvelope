# a hack to ensure scripts search cwd
import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
from kw.convex_adversarial import epsilon_from_model, DualNetBounds
from kw.convex_adversarial import Dense, DenseSequential
import math
import os

def replace_10_with_0(y): 
    return y % 10

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, shuffle_test=False): 
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def fashion_mnist_loaders(batch_size): 
    mnist_train = datasets.MNIST("./fashion_mnist", train=True,
       download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./fashion_mnist", train=False,
       download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def svhn_loaders(batch_size): 
    train = datasets.SVHN("./data", split='train', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    test = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def cifar_loaders(batch_size, shuffle_test=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.225, 0.225, 0.225])
    normalize = transforms.Normalize(mean = [0., 0., 0.], std = [1., 1., 1.])
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def mnist_fc1():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 1024),
        nn.ReLU(inplace = True),
        nn.Linear(1024, 10)
        )
    return model

def mnist_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(inplace = True),
        nn.Linear(100, 10)
        )
    return model

def cifar_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(inplace = True),
        nn.Linear(100, 10)
        )
    return model

def cifar_model_resnet(N = 1, factor=1): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, False)
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*8*8,1000), 
        nn.ReLU(), 
        nn.Linear(1000, 10)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model

def svhn_cnn(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(inplace = True),
        Flatten(),
        nn.Linear(32 * 8 * 8,100),
        nn.ReLU(inplace = True),
        nn.Linear(100, 10)
    ).cuda()
    return model

def argparser(batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3, 
              epsilon=0.1, starting_epsilon=None, 
              proj=None, 
              norm_train='l1', norm_test='l1', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size,
        help = 'The batch size')
    parser.add_argument('--test_batch_size', type=int, default=batch_size,
        help = 'Test batch size')
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon,
        help = 'The value of epsilon')
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon,
        help = 'The number of starting epsilon')
    parser.add_argument('--schedule_length', type=int, default=10)

    # projection settings
    parser.add_argument('--proj', type=int, default=proj)
    parser.add_argument('--norm_train', default=norm_train,
        help = 'The norm used during training, default = "l1"')
    parser.add_argument('--norm_test', default=norm_test,
        help = 'The norm used during test, default = "l1"')

    # model arguments
    parser.add_argument('--model', default=None,
        help = 'The type of the model, like "FC1" or "CNN"')
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)


    # other arguments
    parser.add_argument('--prefix', help = 'The prefix of the model saved')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None, help = 'The id of cuda devices used')

    args = parser.parse_args()

    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon 
    if args.prefix: 
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load', 'real_time', 
                  'test_batch_size']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # Ignore these parameters for filename since we never change them
        banned += ['momentum', 'weight_decay']

        if args.cascade == 1: 
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        # if args.model != 'resnet': 
        banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args

def args2kwargs(args, X=None): 

    if args.proj is not None: 
        kwargs = {
            'proj' : args.proj, 
        }
    else:
        kwargs = {
        }
    kwargs['parallel'] = (args.cuda_ids is not None)
    return kwargs



def argparser_evaluate(epsilon=0.1, norm='l1'): 

    parser = argparse.ArgumentParser()

    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument('--proj', type=int, default=None)
    parser.add_argument('--norm', default=norm)
    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset', default='mnist')

    parser.add_argument('--load')
    parser.add_argument('--output')

    parser.add_argument('--real_time', action='store_true')
    # parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=True)
    parser.add_argument('--cuda_ids', default=None)

    
    args = parser.parse_args()

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args
