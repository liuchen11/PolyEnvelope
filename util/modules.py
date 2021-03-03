'''
>>> This file creates modules that can do forward, backward pass as well as bound propagation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from abc import ABCMeta, abstractmethod

import math
import numpy as np

from .linearize import linearize_relu, linearize_sigd, linearize_tanh, linearize_arctan
from .utility import reduced_m_bm, reduced_bm_bm, reduced_bv_bm, reduced_bm_bv, norm_based_bounds, merge_W_m_list

class Layer(nn.Module, metaclass = ABCMeta):

    def __init__(self,):

        super(Layer, self).__init__()

    @abstractmethod
    def forward(self, x):
        '''
        >>> do forward pass with a given input
        '''

        raise NotImplementedError

    @abstractmethod
    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> do bound propagation based on quadratic algorithm

        >>> x: the input data point
        >>> l: the lower bound of the input, not flattened
        >>> u: the upper bound of the input, not flattened
        >>> W_list: the transformation matrix introduced by the previous layers, of shape [batch_size, current_dim, input_dim]
        >>> m1_list, m2_list: the bias introduced by the previous layers, of shape [batch_size, input_dim]
        >>> first_layer: boolean, whether or not this is the first layer of the model
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> in_shape: the original shape of the input, used for convolutional kernel flatten
        >>> pixel_range: [min, max], the allowable pixel range, default is None
        >>> is_certify: whether or not in a certify mode

        >>> return the updated W_list, m1_list, m2_list
        '''

        raise NotImplementedError

    @abstractmethod
    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):
        '''
        >>> do bound propagation based on simple algorithm

        >>> x: the input data point
        >>> l: the lower bound of the input
        >>> u: the upper bound of the input
        >>> W, m1, m2: the linearized bounds of the input, Wx + m1 < input < Wx + m2
            W have the shape [batch_size, out_dim, in_dim]
            m1, m2 have the shape [batch_size, out_dim]
        >>> first_layer: boolean, whether or not this is the first layer of the model
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> in_shape: the original shape of the input, used for convolutional kernel flatten
        >>> pixel_range: [min, max], the allowable pixel range, default is None
        >>> is_certify: whether or not in a certify mode

        >>> return the lower and upper bound of the current layer and updated W1, W2, m1, m2
        '''

        raise NotImplementedError

    @abstractmethod
    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):
        '''
        >>> do IBP bound propagation

        >>> x: the input data point
        >>> l: the lower bound of the input
        >>> u: the upper bound of the input
        >>> first_layer: boolean, whether or not this is the first layer of the model
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> in_shape: the original shape of the input, used for convolutional kernel flatten
        >>> pixel_range: [min, max], the allowable pixel range, default is None

        >>> return the lower and upper bound of the current layer
        '''

        raise NotImplementedError

    @abstractmethod
    def shape_transfer(self, in_shape):
        '''
        >>> given the input shape, return the output shape
        '''

        raise NotImplementedError

    @abstractmethod
    def base(self,):
        '''
        >>> return the base primitive pytorch model
        '''

        raise NotImplementedError

class FCLayer(Layer):

    def __init__(self, in_features, out_features):

        super(FCLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):

        return F.linear(x, self.layer.weight, self.layer.bias)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        batch_size = in_shape[0]

        # if the bias term in the last iteration is the same, we can merge it with the current one
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update the transition weight matrix
        update_list = W_list if max_var > 1e-4 or first_layer == True else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_m_bm(self.layer.weight, W)

        # Add the contribution of this layer
        if max_var > 1e-4 or first_layer == True:
            W_list.append(torch.ones([batch_size, self.out_features], device = self.layer.weight.device))
            m1_list.append(self.layer.bias.unsqueeze(0).repeat(batch_size, 1))
            m2_list.append(self.layer.bias.unsqueeze(0).repeat(batch_size, 1))
        else:
            W_list[-1] = torch.ones([batch_size, self.out_features], device = self.layer.weight.device)
            m1_list[-1] = torch.matmul(m1_list[-1], self.layer.weight.transpose(0, 1)) + self.layer.bias
            m2_list[-1] = torch.matmul(m2_list[-1], self.layer.weight.transpose(0, 1)) + self.layer.bias

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        W_pos = torch.clamp(self.layer.weight, min = 0.)
        W_neg = torch.clamp(self.layer.weight, max = 0.)

        new_W = reduced_m_bm(self.layer.weight, W)
        new_m1 = torch.matmul(m1, W_pos.transpose(0, 1)) + torch.matmul(m2, W_neg.transpose(0, 1)) + self.layer.bias
        new_m2 = torch.matmul(m2, W_pos.transpose(0, 1)) + torch.matmul(m1, W_neg.transpose(0, 1)) + self.layer.bias

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        if first_layer == True:
            transit_matrix = self.layer.weight.unsqueeze(0).repeat(x.size(0), 1, 1)
            l, u = norm_based_bounds(x.view(x.size(0), -1), W = transit_matrix, perturb_norm = ori_perturb_norm, perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
            new_l = l + self.layer.bias
            new_u = u + self.layer.bias
        else:
            W_pos = torch.clamp(self.layer.weight, min = 0.)
            W_neg = torch.clamp(self.layer.weight, max = 0.)
            new_l = torch.matmul(l, W_pos.transpose(0, 1)) + torch.matmul(u, W_neg.transpose(0, 1)) + self.layer.bias
            new_u = torch.matmul(u, W_pos.transpose(0, 1)) + torch.matmul(l, W_neg.transpose(0, 1)) + self.layer.bias

        return new_l, new_u

    def shape_transfer(self, in_shape):

        assert np.prod(in_shape[1:]) == self.in_features, 'dimension mismatch, %d required but %d given' % (self.in_features, np.prod(in_shape[1:]))

        return [in_shape[0], self.out_features]

    def base(self,):

        layer = nn.Linear(self.in_features, self.out_features)
        layer.weight.data = self.layer.weight.data
        layer.bias.data = self.layer.bias.data

        return layer

class Conv2dLayer(Layer):

    def __init__(self, in_channel, out_channel, kernel_size, stride = 1, padding = 0):

        super(Conv2dLayer, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layer = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)

        self.flattened_weight = None            # The matrix form of convolutional kernel
        self.flattened_bias = None              # The vector form of convolutional bias
        self.flattened_shape = None             # The input shape of the convolutional layer

    def flatten(self, in_shape):
        '''
        >>> to obtain the flatten version of convolutional kernel and bias

        >>> in_shape: the shape of the input

        >>> Input/Output of ConvLayer: [batch_size, channel_idx, height, width]
        >>> Weight of ConvLayer: [channel_out, channel_in, kernel_height, kernel_width]
        >>> Size of output matrix: [out_dim, in_dim]
        '''

        assert len(in_shape[1:]) == 3, 'the input shape should be [channel, height, width]'
        assert in_shape[1] == self.in_channel, 'the number of channels of the input should be aligned with the kernel, but they are %d and %d respectively' % (in_shape[1], self.in_channel)

        self.flattened_shape = in_shape[1:]

        out_height = (in_shape[2] + 2 * self.padding - self.kernel_size + 0) // self.stride + 1
        out_width = (in_shape[3] + 2 * self.padding - self.kernel_size + 0) // self.stride + 1

        in_dim = int(np.prod(in_shape[1:]))
        out_dim = int(np.prod([out_width, out_height, self.out_channel]))

        inp_I = torch.eye(in_dim).view(-1, *in_shape[1:]).to(self.layer.weight.device)
        outp_I = F.conv2d(inp_I, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)

        outp_I = outp_I.view(in_dim, out_dim)
        self.flattened_weight = outp_I.t()
        self.flattened_bias = self.layer.bias.reshape(-1, 1).repeat(1, out_width * out_height).reshape(-1)

        return self.flattened_weight, self.flattened_bias

    def forward(self, x):

        return F.conv2d(x, self.layer.weight, self.layer.bias, self.layer.stride,
            self.layer.padding, self.layer.dilation, self.layer.groups)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        batch_size = in_shape[0]

        # Obtain the flattened weights and bias
        weight_matrix, bias_vector = self.flatten(in_shape = in_shape)
        out_dim = weight_matrix.shape[0]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update the transition weight matrix
        update_list = W_list if max_var > 1e-4 or first_layer == True else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_m_bm(weight_matrix, W)

        # Add the contribution of this layer
        if max_var > 1e-4 or first_layer == True:
            W_list.append(torch.ones([batch_size, out_dim], device = weight_matrix.device))
            m1_list.append(bias_vector.unsqueeze(0).repeat(batch_size, 1))
            m2_list.append(bias_vector.unsqueeze(0).repeat(batch_size, 1))
        else:
            W_list[-1] = torch.ones([batch_size, out_dim], device = weight_matrix.device)
            m1_list[-1] = torch.matmul(m1_list[-1], weight_matrix.transpose(0, 1)) + bias_vector.unsqueeze(0).repeat(batch_size, 1)
            m2_list[-1] = torch.matmul(m2_list[-1], weight_matrix.transpose(0, 1)) + bias_vector.unsqueeze(0).repeat(batch_size, 1)

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        weight_matrix, bias_vector = self.flatten(in_shape = in_shape)

        W_pos = torch.clamp(weight_matrix, min = 0.)
        W_neg = torch.clamp(weight_matrix, max = 0.)

        new_W = reduced_m_bm(weight_matrix, W)
        new_m1 = torch.matmul(m1, W_pos.transpose(0, 1)) + torch.matmul(m2, W_neg.transpose(0, 1)) + bias_vector
        new_m2 = torch.matmul(m2, W_pos.transpose(0, 1)) + torch.matmul(m1, W_neg.transpose(0, 1)) + bias_vector

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        weight_matrix, bias_vector = self.flatten(in_shape = in_shape)

        if first_layer == True:
            transit_matrix = weight_matrix.unsqueeze(0).repeat(x.size(0), 1, 1)
            l, u = norm_based_bounds(x = x.view(x.size(0), -1), W = transit_matrix, perturb_norm = ori_perturb_norm, perturb_eps = ori_perturb_eps, pixel_range = pixel_range)
            new_l = l + bias_vector
            new_u = u + bias_vector
        else:
            W_pos = torch.clamp(weight_matrix, min = 0.)
            W_neg = torch.clamp(weight_matrix, max = 0.)
            new_l = torch.matmul(l, W_pos.transpose(0, 1)) + torch.matmul(u, W_neg.transpose(0, 1)) + bias_vector
            new_u = torch.matmul(u, W_pos.transpose(0, 1)) + torch.matmul(l, W_neg.transpose(0, 1)) + bias_vector

        return new_l, new_u

    def shape_transfer(self, in_shape):

        assert len(in_shape) == 4, 'the in_shape of convolutional layer should be [batch_size, channel, height, width]'
        assert in_shape[1] == self.in_channel, 'in_channel dimension mismatch, %d required but %d given' % (self.in_channel, in_shape[1])
        out_height = (in_shape[2] + 2 * self.padding - self.kernel_size + 0) // self.stride + 1
        out_width = (in_shape[3] + 2 * self.padding - self.kernel_size + 0) // self.stride + 1
        return [in_shape[0], self.out_channel, out_height, out_width]

    def base(self,):

        layer = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, self.stride, self.padding)
        layer.weight.data = self.layer.weight.data
        layer.bias.data = self.layer.bias.data

        return layer

class ReLULayer(Layer):

    def __init__(self,):

        super(ReLULayer, self).__init__()

    def forward(self, x):

        return F.relu(x, inplace = True)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        batch_size = in_shape[0]

        # Obtain D, m1, m2
        D, m1, m2 = linearize_relu(l, u, is_certify)
        D = D.reshape(batch_size, -1)               # of shape [batch_size, dim]
        m1 = m1.reshape(batch_size, -1)             # of shape [batch_size, dim]
        m2 = m2.reshape(batch_size, -1)             # of shape [batch_size, dim]
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        D_this_layer, m1_this_layer, m2_this_layer = linearize_relu(l, u, is_certify)       # of shape [batch_size, out_dim]

        new_W = reduced_bm_bm(D_this_layer, W)
        new_m1 = m1 * D_this_layer + m1_this_layer
        new_m2 = m2 * D_this_layer + m2_this_layer

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        assert first_layer == False, 'activation layer cannot be the first layer'

        l = self.forward(l)
        u = self.forward(u)

        return l, u

    def shape_transfer(self, in_shape):

        return in_shape

    def base(self,):

        return torch.nn.ReLU()

class SigdLayer(Layer):

    def __init__(self,):

        super(SigdLayer, self).__init__()

    def forward(self, x):

        return F.sigmoid(x)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        batch_size = in_shape[0]

        # Obtain D, m1, m2
        D, m1, m2 = linearize_sigd(l, u, is_certify)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        D_this_layer, m1_this_layer, m2_this_layer = linearize_sigd(l, u, is_certify)       # of shape [batch_size, out_dim]

        new_W = reduced_bm_bm(D_this_layer, W)
        new_m1 = m1 * D_this_layer + m1_this_layer
        new_m2 = m2 * D_this_layer + m2_this_layer

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        assert first_layer == False, 'activation layer cannot be the first layer'

        l = self.forward(l)
        u = self.forward(u)

        return l, u

    def shape_transfer(self, in_shape):

        return in_shape

    def base(self,):

        return torch.nn.Sigmoid()

class TanhLayer(Layer):

    def __init__(self,):

        super(TanhLayer, self).__init__()

    def forward(self, x):

        return torch.tanh(x)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        batch_size = in_shape[0]

        # Obtain D, m1, m2
        D, m1, m2 = linearize_tanh(l, u, is_certify)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        D_this_layer, m1_this_layer, m2_this_layer = linearize_tanh(l, u, is_certify)       # of shape [batch_size, out_dim]

        new_W = reduced_bm_bm(D_this_layer, W)
        new_m1 = m1 * D_this_layer + m1_this_layer
        new_m2 = m2 * D_this_layer + m2_this_layer

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        assert first_layer == False, 'activation layer cannot be the first layer'

        l = self.forward(l)
        u = self.forward(u)

        return l, u

    def shape_transfer(self, in_shape):

        return in_shape

    def base(self,):

        return torch.nn.Tanh()

class ArctanLayer(Layer):

    def __init__(self,):

        super(ArctanLayer, self).__init__()

    def forward(self, x):

        return torch.atan(x)

    def bound_quad(self, x, l, u, W_list, m1_list, m2_list, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        batch_size = in_shape[0]

        # Obtain D, m1, m2
        D, m1, m2 = linearize_arctan(l, u, is_certify)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        return W_list, m1_list, m2_list

    def bound_simp(self, x, l, u, W, m1, m2, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None, is_certify = 0):

        assert first_layer == False, 'activation layer cannot be the first layer'

        D_this_layer, m1_this_layer, m2_this_layer = linearize_arctan(l, u, is_certify)     # of shape [batch_size, out_dim]

        new_W = reduced_bm_bm(D_this_layer, W)
        new_m1 = m1 * D_this_layer + m1_this_layer
        new_m2 = m2 * D_this_layer + m2_this_layer

        return new_W, new_m1, new_m2

    def bound_ibp(self, x, l, u, first_layer = False, ori_perturb_norm = None, ori_perturb_eps = None, in_shape = None, pixel_range = None):

        assert first_layer == False, 'activation layer cannot be the first layer'

        l = self.forward(l)
        u = self.forward(u)

        return l, u

    def shape_transfer(self, in_shape):

        return in_shape

    def base(self,):

        class ArcTan(nn.modules):

            def __init__(self, ):

                super(ArcTan, self).__init__()

            def forward(self, x):

                return torch.atan(x)

        return ArcTan()

