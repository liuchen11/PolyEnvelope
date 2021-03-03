import sys
sys.path.insert(0, './')
import torch
import torch.nn as nn

import math
import numpy as np

from abc import ABCMeta, abstractmethod
from kw.problems import cifar_model_resnet
from .modules import FCLayer, Conv2dLayer, ReLULayer, SigdLayer, TanhLayer, ArctanLayer
from .utility import reduced_bm_m, reduced_bv_bm, reduced_bm_bv, norm_based_bounds, gen_diff_matrix, logit_diff_quad

str2func = {'relu': ReLULayer, 'sigd': SigdLayer, 'tanh': TanhLayer, 'arctan': ArctanLayer}
approx_layer = (ReLULayer, SigdLayer, TanhLayer, ArctanLayer)

class Flatten(nn.Module):

    def __init__(self,):

        super(Flatten, self).__init__()

    def forward(self, x):

        return x.reshape(x.size(0), -1)

class Model(nn.Module, metaclass = ABCMeta):

    def __init__(self,):

        super(Model, self).__init__()

    @abstractmethod
    def forward(self, x):
        '''
        >>> do forward pass with a given input
        '''

        raise NotImplementedError

    @abstractmethod
    def bound_quad(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> do bound propagation based on quadratic algorithm

        >>> x: the input data
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> pixel_range: [min, max], allowable pixel range, default is None

        >>> return: l, u, W_list, m1_list, m2_list
        '''

        raise NotImplementedError

    @abstractmethod
    def bound_simp(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> do bound propagation based on simple algorithm

        >>> x: the input data
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> pixel_range: [min, max], allowable pixel range, default is None

        >>> return: l, u, W1, W2, m1, m2
        '''

        raise NotImplementedError

    @abstractmethod
    def crown_ibp(self, x, y, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, only_ibp = False):
        '''
        >>> do bound propagation in CROWN-IBP

        >>> x: the input data
        >>> y: the true label of the input data
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> pixel_range: [min, max], allowable pixel range, default is None

        >>> return: l, u, W1, W2, m1, m2
        '''

        raise NotImplementedError

    @abstractmethod
    def model2sequence(self,):
        '''
        >>> convert the model to nn.Sequence
        '''

        raise NotImplementedError

class MLP(Model):
    '''
    >>> General class for multilayer perceptron
    '''

    def __init__(self, in_dim = 784, hidden_dims = [1024,], out_dim = 10, nonlinearity = 'relu'):

        super(MLP, self).__init__()

        self.neurons = [in_dim,] + hidden_dims + [out_dim,]
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.nonlinearity = nonlinearity

        self.main_block = nn.Sequential()

        for idx, (in_neuron, out_neuron) in enumerate(zip(self.neurons[:-2], self.neurons[1:-1])):
            linear_layer = FCLayer(in_features = in_neuron, out_features = out_neuron)
            nonlinear_layer = str2func[self.nonlinearity]()
            self.main_block.add_module('layer_%d'%idx, linear_layer)
            self.main_block.add_module('nonlinear_%d'%idx, nonlinear_layer)

        self.output = FCLayer(in_features = self.neurons[-2], out_features = self.neurons[-1])

    def forward(self, x):

        out = x.view(x.size(0), -1)
        out = self.main_block(out)
        out = self.output(out)

        return out

    def bound_quad(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W_list = [torch.ones_like(x.view(batch_size, in_dim))]
        m1_list = [x.view(batch_size, in_dim),]
        m2_list = [x.view(batch_size, in_dim),]

        # Main Block
        for idx, layer in enumerate(self.main_block):

            l = torch.zeros_like(m1_list[-1])
            u = torch.zeros_like(m2_list[-1])
            if isinstance(layer, approx_layer): # Layers that need lower and upper bound

                # approximation from the first layer
                l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
                    perturb_eps = ori_perturb_eps, pixel_range = pixel_range)

                # approximation from the subsequent layer
                for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
                    W_pos = torch.clamp(W, min = 0.)
                    W_neg = torch.clamp(W, max = 0.)

                    l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
                    u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)

            first_layer = idx == 0

            W_list, m1_list, m2_list = layer.bound_quad(x, l, u, W_list, m1_list, m2_list,
                 first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range, is_certify)
            in_shape = layer.shape_transfer(in_shape)

        first_layer = len(self.hidden_dims) == 0
        W_list, m1_list, m2_list = self.output.bound_quad(x, l, u, W_list, m1_list, m2_list,
            first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range, is_certify)

        # calculate the final bound
        l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
            perturb_eps = ori_perturb_eps, pixel_range = pixel_range)

        for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
            W_pos = torch.clamp(W, min = 0.)
            W_neg = torch.clamp(W, max = 0.)

            l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
            u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)

        return l, u, W_list, m1_list, m2_list

    def bound_simp(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> For l infty norm based perturbation budget, use the ori_perturb_norm/ori_perturb_eps instead of m1/m2
        '''

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W = torch.ones_like(x.view(batch_size, in_dim))
        m1 = torch.zeros_like(x)
        m2 = torch.zeros_like(x)

        l = x.view(batch_size, in_dim)
        u = x.view(batch_size, in_dim)

        # Main Block
        for idx, layer in enumerate(self.main_block):

            l = torch.zeros_like(m1)
            u = torch.zeros_like(m2)
            if isinstance(layer, approx_layer): # Layers that need lower and upper bound

                # approximation from the input perturbation
                l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W, perturb_norm = ori_perturb_norm,
                    perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
                l = l + m1
                u = u + m2

            ori_perturb_norm_this_layer = ori_perturb_norm if idx == 0 else None
            ori_perturb_eps_this_layer = ori_perturb_eps if idx == 0 else None
            pixel_range_this_layer = pixel_range if idx == 0 else None
            first_layer = True if idx == 0 else False

            W, m1, m2 = layer.bound_simp(x = x, l = l, u = u, W = W, m1 = m1, m2 = m2,
                ori_perturb_norm = ori_perturb_norm_this_layer, ori_perturb_eps = ori_perturb_eps_this_layer,
                in_shape = in_shape, pixel_range = pixel_range_this_layer, first_layer = first_layer, is_certify = is_certify)
            in_shape = layer.shape_transfer(in_shape)

        W, m1, m2 = self.output.bound_simp(x = x, l = l, u = u, W = W, m1 = m1, m2 = m2, in_shape = in_shape, is_certify = is_certify)

        # calculate the final bound
        l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W, perturb_norm = ori_perturb_norm,
            perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
        l = l + m1
        u = u + m2

        return l, u, W, m1, m2

    def crown_ibp(self, x, y, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, only_ibp = False):

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W_list = [torch.ones_like(x.view(batch_size, in_dim)),]
        m1_list = [x.view(batch_size, in_dim),]
        m2_list = [x.view(batch_size, in_dim),]

        l = u = None

        # Main Block
        for idx, layer in enumerate(self.main_block):

            first_layer = idx == 0
            if only_ibp == False:
                W_list, m1_list, m2_list = layer.bound_quad(x, l, u, W_list, m1_list, m2_list,
                    first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            l, u = layer.bound_ibp(x, l, u, first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            in_shape = layer.shape_transfer(in_shape)

        # Last layer, merge it with the comparision matrix
        first_layer = len(self.hidden_dims) == 0
        if only_ibp == False:
            W_list, m1_list, m2_list = self.output.bound_quad(x, l, u, W_list, m1_list, m2_list,
                first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            W_list, m1_list, m2_list = logit_diff_quad(W_list, m1_list, m2_list, c = y)
            # Calculate the CROWN-IBP bound
            crown_l, crown_u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
                perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
            for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
                W_pos = torch.clamp(W, min = 0.)
                W_neg = torch.clamp(W, max = 0.)
                crown_l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
                crown_u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)
        else:
            crown_l = None
            crown_u = None

        # Calculate IBP bound
        diff_matrix = gen_diff_matrix(c = y, out_dim = self.out_dim)
        weight = reduced_bm_m(diff_matrix, self.output.layer.weight)
        bias = reduced_bm_m(diff_matrix, self.output.layer.bias.unsqueeze(1)).squeeze(2)

        weight_pos = torch.clamp(weight, min = 0.)
        weight_neg = torch.clamp(weight, max = 0.)

        ibp_l = reduced_bm_bv(weight_pos, l) + reduced_bm_bv(weight_neg, u) + bias
        ibp_u = reduced_bm_bv(weight_neg, l) + reduced_bm_bv(weight_pos, u) + bias

        return ibp_l, ibp_u, crown_l, crown_u

    def model2sequence(self,):

        input_base = [Flatten()]
        main_block_base = [layer.base() for layer in self.main_block]
        layer_list = input_base + main_block_base + [self.output.base(),]

        return nn.Sequential(*layer_list)

class ConvNet(Model):
    '''
    >>> Class for ConvNet containing convolutional layers followed by fully connected layers
    '''

    def __init__(self, in_size = 28, in_channel = 1, conv_kernels = [4, 4], conv_strides = [2, 2],
        conv_channels = [16, 32], conv_pads = [1, 1], hidden_dims = [100,], out_dim = 10, nonlinearity = 'relu'):

        super(ConvNet, self).__init__()

        self.in_size = in_size
        self.in_channel = in_channel
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_channels = conv_channels
        self.conv_pads = conv_pads
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.nonlinearity = nonlinearity

        assert len(self.conv_kernels) == len(self.conv_strides) == len(self.conv_channels) == len(self.conv_pads), \
            'conv_kernels, conv_strides, conv_channels, conv_pads should have the same length'

        self.conv_block = nn.Sequential()
        self.fc_block = nn.Sequential()

        current_size = self.in_size
        current_channel = self.in_channel

        for idx, (conv_kernel, conv_stride, conv_channel, conv_pad) in enumerate(zip(self.conv_kernels, self.conv_strides, self.conv_channels, self.conv_pads)):

            linear_layer = Conv2dLayer(in_channel = current_channel, out_channel = conv_channel, kernel_size = conv_kernel, stride = conv_stride, padding = conv_pad)
            nonlinear_layer = str2func[self.nonlinearity]()
            self.conv_block.add_module('conv_layer_%d'%idx, linear_layer)
            self.conv_block.add_module('conv_nonlinear_%d'%idx, nonlinear_layer)

            current_size = (current_size + conv_pad * 2 - conv_kernel) // conv_stride + 1
            current_channel = conv_channel

        current_neuron = current_size ** 2 * current_channel

        for idx, (in_neuron, out_neuron) in enumerate(zip([current_neuron,] + self.hidden_dims[:-1], self.hidden_dims)):

            linear_layer = FCLayer(in_features = in_neuron, out_features = out_neuron)
            nonlinear_layer = str2func[self.nonlinearity]()
            self.fc_block.add_module('fc_layer_%d'%idx, linear_layer)
            self.fc_block.add_module('fc_nonlinear_%d'%idx, nonlinear_layer)

        self.output = FCLayer(in_features = self.hidden_dims[-1], out_features = self.out_dim)

    def forward(self, x):

        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        out = self.fc_block(out)
        out = self.output(out)

        return out

    def bound_quad(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W_list = [torch.ones_like(x.view(batch_size, in_dim)),]
        m1_list = [x.view(batch_size, in_dim),]
        m2_list = [x.view(batch_size, in_dim),]

        conv_layer_list = [layer for layer in self.conv_block]
        fc_layer_list = [layer for layer in self.fc_block]

        for idx, layer in enumerate(conv_layer_list + fc_layer_list):

            l = torch.zeros_like(m1_list[-1])
            u = torch.zeros_like(m2_list[-1])
            if isinstance(layer, approx_layer):         # Layers that need lower and upper bound

                # approximation from the first layer
                l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
                    perturb_eps = ori_perturb_eps, pixel_range = pixel_range)

                # approximation from the subsequent layer
                for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
                    W_pos = torch.clamp(W, min = 0.)
                    W_neg = torch.clamp(W, max = 0.)

                    l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
                    u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)

            first_layer = idx == 0

            W_list, m1_list, m2_list = layer.bound_quad(x, l, u, W_list, m1_list, m2_list,
                first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range, is_certify)
            in_shape = layer.shape_transfer(in_shape)

        first_layer = len(conv_layer_list + fc_layer_list) == 0
        W_list, m1_list, m2_list = self.output.bound_quad(x, l, u, W_list, m1_list, m2_list,
            first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range, is_certify)

        # calculate the final bound
        l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
            perturb_eps = ori_perturb_eps, pixel_range = pixel_range)

        for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
            W_pos = torch.clamp(W, min = 0.)
            W_neg = torch.clamp(W, max = 0.)

            l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
            u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)

        return l, u, W_list, m1_list, m2_list

    def bound_simp(self, x, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> For l infty norm based perturbation budget, use the ori_perturb_norm/ori_perturb_eps instead of m1/m2
        '''

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W = torch.ones_like(x.view(batch_size, in_dim))
        m1 = torch.zeros_like(x)
        m2 = torch.zeros_like(x)

        l = x.view(batch_size, in_dim)
        u = x.view(batch_size, in_dim)

        conv_layer_list = [layer for layer in self.conv_block]
        fc_layer_list = [layer for layer in self.fc_block]

        for idx, layer in enumerate(conv_layer_list + fc_layer_list):

            l = torch.zeros_like(m1)
            u = torch.zeros_like(m2)
            if isinstance(layer, approx_layer):     # Layers that need lower and upper bound

                # approximation from the input perturbation
                l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W, perturb_norm = ori_perturb_norm,
                    perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
                l = l + m1
                u = u + m2

            ori_perturb_norm_this_layer = ori_perturb_norm if idx == 0 else None
            ori_perturb_eps_this_layer = ori_perturb_eps if idx == 0 else None
            pixel_range_this_layer = pixel_range if idx == 0 else None
            first_layer = True if idx == 0 else False

            W, m1, m2 = layer.bound_simp(x = x, l = l, u = u, W = W, m1 = m1, m2 = m2,
                ori_perturb_norm = ori_perturb_norm_this_layer, ori_perturb_eps = ori_perturb_eps_this_layer,
                in_shape = in_shape, pixel_range = pixel_range_this_layer, first_layer = first_layer, is_certify = is_certify)
            in_shape = layer.shape_transfer(in_shape)

        W, m1, m2 = self.output.bound_simp(x = x, l = l, u = u, W = W, m1 = m1, m2 = m2, in_shape = in_shape, is_certify = is_certify)

        # calculate the final bound
        l, u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W, perturb_norm = ori_perturb_norm,
            perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
        l = l + m1
        u = u + m2

        return l, u, W, m1, m2

    def crown_ibp(self, x, y, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, only_ibp = False):

        in_dim = int(np.prod(x.shape[1:]))
        batch_size = x.shape[0]

        W_list = [torch.ones_like(x.view(batch_size, in_dim)),]
        m1_list = [x.view(batch_size, in_dim),]
        m2_list = [x.view(batch_size, in_dim),]

        conv_layer_list = [layer for layer in self.conv_block]
        fc_layer_list = [layer for layer in self.fc_block]

        l = u = None

        for idx, layer in enumerate(conv_layer_list + fc_layer_list):

            first_layer = idx == 0
            if only_ibp == False:
                W_list, m1_list, m2_list = layer.bound_quad(x, l, u, W_list, m1_list, m2_list,
                    first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            l, u = layer.bound_ibp(x, l, u, first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            in_shape = layer.shape_transfer(in_shape)

        # Last layer, merge it with the comparison metric
        first_layer = len(conv_layer_list + fc_layer_list) == 0
        if only_ibp == False:
            W_list, m1_list, m2_list = self.output.bound_quad(x, l, u, W_list, m1_list, m2_list,
                first_layer, ori_perturb_norm, ori_perturb_eps, in_shape, pixel_range)
            W_list, m1_list, m2_list = logit_diff_quad(W_list, m1_list, m2_list, c = y)
            # Calculate the CROWN-IBP bound
            crown_l, crown_u = norm_based_bounds(x = x.view(batch_size, in_dim), W = W_list[0], perturb_norm = ori_perturb_norm,
                perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
            for W, m1, m2 in zip(W_list[1:], m1_list[1:], m2_list[1:]):
                W_pos = torch.clamp(W, min = 0.)
                W_neg = torch.clamp(W, max = 0.)
                crown_l += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
                crown_u += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)
        else:
            crown_l = None
            crown_u = None

        # Calculate IBP bound
        diff_matrix = gen_diff_matrix(c = y, out_dim = self.out_dim)
        weight = reduced_bm_m(diff_matrix, self.output.layer.weight)
        bias = reduced_bm_m(diff_matrix, self.output.layer.bias.unsqueeze(1)).squeeze(2)

        weight_pos = torch.clamp(weight, min = 0.)
        weight_neg = torch.clamp(weight, max = 0.)

        ibp_l = reduced_bm_bv(weight_pos, l) + reduced_bm_bv(weight_neg, u) + bias
        ibp_u = reduced_bm_bv(weight_neg, l) + reduced_bm_bv(weight_pos, u) + bias

        return ibp_l, ibp_u, crown_l, crown_u

    def model2sequence(self,):

        conv_layer_base_list = [layer.base() for layer in self.conv_block]
        fc_layer_base_list = [layer.base() for layer in self.fc_block]
        layer_list = conv_layer_base_list + [Flatten(),] + fc_layer_base_list + [self.output.base()]

        return nn.Sequential(*layer_list)

    def conv_params(self,):
        '''
        >>> return convolutional parameters
        '''
        param_list = []
        for layer in self.conv_block:
            if 'layer' in dir(layer) and 'weight' in dir(layer.layer):
                param_list.append(layer.layer.weight)
            if 'layer' in dir(layer) and 'bias' in dir(layer.layer):
                param_list.append(layer.layer.bias)

        return param_list

    def fc_params(self,):
        '''
        >>> return fully-connected parameters
        '''
        param_list = []
        for layer in self.fc_block:
            if 'layer' in dir(layer) and 'weight' in dir(layer.layer):
                param_list.append(layer.layer.weight)
            if 'layer' in dir(layer) and 'bias' in dir(layer.layer):
                param_list.append(layer.layer.bias)

        param_list = param_list + [self.output.layer.weight, self.output.layer.bias]

        return param_list
