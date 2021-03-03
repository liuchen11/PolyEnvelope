import torch
import torch.nn as nn

import numpy as np

def reduced_m_bm(m1, m2):
    '''
    >>> merge a new constant transformation with a batch transformation

    >>> m1: tensor of shape [dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2] or [batch_size, dim2, dim3]
    '''
    assert len(m1.shape) == 2, 'The dim of m1 should be 2.'
    dim2 = len(m2.shape)

    if dim2 == 2:
        mbm = m1.unsqueeze(0) * m2.unsqueeze(1)
    elif dim2 == 3:
        mbm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m2 should be either 2 or 3.')

    return mbm

def reduced_bm_m(m1, m2):
    '''
    >>> merge a batch transformation with a new constant transformation

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [dim2, dim3]
    '''
    assert len(m2.shape) == 2, 'The dim of m2 should be 2.'
    dim1 = len(m1.shape)

    if dim1 == 2:
        bmm = m1.unsqueeze(2) * m2.unsqueeze(0)
    elif dim1 == 3:
        bmm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m1 should be either 2 or 3.')

    return bmm

def reduced_bm_bm(m1, m2):
    '''
    >>> merge a batch trasformation with a new batch of transformation

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2] or [batch_size, dim2, dim3]
    '''
    dim1 = len(m1.shape)
    dim2 = len(m2.shape)

    if (dim1, dim2) == (2, 2):
        bmbm = m1 * m2
    elif (dim1, dim2) == (2, 3):
        bmbm = m1.unsqueeze(2) * m2
    elif (dim1, dim2) == (3, 2):
        bmbm = m1 * m2.unsqueeze(1)
    elif (dim1, dim2) == (3, 3):
        bmbm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m1 and m2 should be either 2 or 3.')

    return bmbm

def reduced_bv_bm(m1, m2):
    '''
    >>> merge a batch of values with a batch of transformation

    >>> m1: tensor of shape [batch_size, dim1]
    >>> m2: tensor of shape [batch_size, dim1] or [batch_size, dim1, dim2]
    '''
    assert len(m1.shape) == 2, 'The dim of m1 should be 2.'
    dim2 = len(m2.shape)

    if dim2 == 2:
        bvbm = m1 * m2
    elif dim2 == 3:
        bvbm = torch.matmul(m1.unsqueeze(1), m1).squeeze(1)
    else:
        raise ValueError('The dim of m2 should be either 2 or 3.')

    return bvbm

def reduced_bm_bv(m1, m2):
    '''
    >>> merge a batch of transformation with a batch of values

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2]
    '''
    assert len(m2.shape) == 2, 'The dim of m2 should be 2.'
    dim1 = len(m1.shape)

    if dim1 == 2:
        bmbv = m1 * m2
    elif dim1 == 3:
        bmbv = torch.matmul(m1, m2.unsqueeze(2)).squeeze(2)
    else:
        raise ValueError('The dim of m1 should be either 2 or 3.')

    return bmbv

def logit_diff_quad(W_list, m1_list, m2_list, c):
    '''
    >>> See 'elision trick' in CROWN/Fast-Lin/KW
    >>> W_list, m1_list, m2_list: same as module.bound_quad
    >>> c: tensor of shape [batch_size], the label 

    >>> return post-transformation matrices and biases
    '''

    for idx, W in enumerate(W_list):

        if len(W.shape) == 2:
            batch_size, out_dim = W.shape
            base_W = torch.eye(out_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(c.device)   # shape = [batch_size, out_dim, out_dim]
            subs_W = torch.zeros_like(base_W).scatter_(2, c.view(-1, 1, 1).repeat(1, out_dim, 1), torch.ones_like(base_W[:, :, 0:1]))
            transform_W = -(base_W - subs_W)                        # shape = [batch_size, out_dim, out_dim]
            W_list[idx] = transform_W * W.unsqueeze(1)              # shape = [batch_size, out_dim, out_dim]
        else:
            Wc = W.gather(dim = 1, index = c.view(-1, 1, 1).repeat(1, 1, W.shape[2]))
            W_list[idx] = Wc - W

    return W_list, m1_list, m2_list

def logit_diff_simp(W, p1, p2, c):
    '''
    >>> See 'elision trick' in CROWN/Fast-Lin/KW
    >>> W, p1, p2: same as module.bound_simp
    >>> c: tensor of shape [batch_size], the label

    >>> return post-transformation matrices, biases and bounds
    '''
    out_dim = W.shape[1]
    batch_size = c.shape[0]

    Wc = W.gather(dim = 1, index = c.view(-1, 1, 1).repeat(1, 1, W.shape[2]))
    W_new = Wc - W
    
    p1c = p1.gather(dim = 1, index = c.view(-1, 1))
    p2c = p2.gather(dim = 1, index = c.view(-1, 1))
    p1_new = p1c - p2
    p2_new = p2c - p1

    p1_new = p1_new.scatter_(1, c.view(-1, 1), torch.zeros_like(c).view(-1, 1).float())
    p2_new = p2_new.scatter_(1, c.view(-1, 1), torch.zeros_like(c).view(-1, 1).float())

    return W_new, p1_new, p2_new

def gen_diff_matrix(c, out_dim):
    '''
    >>> Generate the difference matrix 

    >>> c: label_batch of size [batch_size,]

    >>> return diff matrix [batch_size, out_dim, out_dim]
    '''

    batch_size = c.size(0)

    base = torch.eye(out_dim, device = c.device).unsqueeze(0).repeat(batch_size, 1, 1)
    one_hot = torch.zeros([batch_size, out_dim], device = c.device).scatter_(1, c.view(-1, 1), torch.ones_like(c.view(-1, 1)).float())

    return one_hot.unsqueeze(1) - base

def norm_based_bounds(x, W, perturb_norm, perturb_eps, pixel_range = None):
    '''
    >>> x: data batch, of shape [batch_size, in_dim]
    >>> W: transformation batch, can be either [batch_size, in_dim] or [batch_size, out_dim, in_dim]
    >>> perturb_norm: the norm used for perturbation
    >>> perturb_eps: the magenitute of the perturbation
    >>> pixel_range: the feasible pixel range, default is None
    '''

    if pixel_range == None:             # No pixel range constraint
        base = reduced_bm_bv(W, x)
        if len(W.shape) == 2:
            adjust = W * perturb_eps            # of shape [batch_size, out_dim]
        else:
            q = 1. / (1. - 1. / perturb_norm)
            adjust = torch.norm(W, dim = 2, p = q) * perturb_eps        # of shape [batch_size, out_dim]
        return base - adjust, base + adjust
    else:
        pixel_min, pixel_max = pixel_range
        W_neg = torch.clamp(W, max = 0.)
        W_pos = torch.clamp(W, min = 0.)

        if perturb_norm == np.inf:
            x_min = torch.clamp(x - perturb_eps, min = pixel_min)
            x_max = torch.clamp(x + perturb_eps, max = pixel_max)
            out_min = reduced_bm_bv(W_pos, x_min) + reduced_bm_bv(W_neg, x_max)
            out_max = reduced_bm_bv(W_pos, x_max) + reduced_bm_bv(W_neg, x_min)
            return out_min, out_max
        else:
            x_min = torch.ones_like(x) * pixel_min
            x_max = torch.ones_like(x) * pixel_max
            out_min1 = reduced_bm_bv(W_pos, x_min) + reduced_bm_bv(W_neg, x_max)
            out_max1 = reduced_bm_bv(W_pos, x_max) + reduced_bm_bv(W_neg, x_min)

            base = reduced_bm_bv(W, x)
            if len(W.shape) == 2:
                adjust = W * perturb_eps
            else:
                q = 1. / (1. - 1. / perturb_norm)
                adjust = torch.norm(W, dim = 2, p = q) * perturb_eps
            out_min2 = base - adjust
            out_max2 = base + adjust

            out_min = (out_min1 + out_min2 + torch.abs(out_min1 - out_min2)) / 2.
            out_max = (out_max1 + out_max2 - torch.abs(out_max1 - out_max2)) / 2.
            return out_min, out_max

def merge_W_m_list(W_list_1, m1_list_1, m2_list_1, W_list_2, m1_list_2, m2_list_2):
    '''
    >>> merge two W, m1, m2 lists

    >>> list1 should be no shorter than list2
    '''
    assert len(W_list_1) == len(m1_list_1) == len(m2_list_1)
    assert len(W_list_2) == len(m1_list_2) == len(m2_list_2)
    assert len(W_list_1) >= len(W_list_2)

    common_length = len(W_list_2)

    W_list = []
    m1_list = []
    m2_list = []

    for W_1, W_2, m1_1, m1_2, m2_1, m2_2 in zip(
        W_list_1[:common_length], W_list_2, m1_list_1[:common_length], m1_list_2, m2_list_1[:common_length], m2_list_2):

        assert float(torch.max(torch.abs(m1_1 - m1_2))) < 1e-8, 'the bias of two item merged must be the same'
        assert float(torch.max(torch.abs(m2_1 - m2_2))) < 1e-8, 'the bias of two item merged must be the same'

        m1_list.append(m1_1)
        m2_list.append(m2_1)

        if len(W_1.shape) == 2 and len(W_2.shape) == 2:
            W_list.append(W_1 + W_2)
        elif len(W_1.shape) == 3 and len(W_2.shape) == 3:
            W_list.append(W_1 + W_2)
        elif len(W_1.shape) == 2 and len(W_2.shape) == 3:
            batch_size, out_dim = W_1.shape
            extend_base = torch.eye(out_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(W_1.device)
            extend_W_1 = W_1.unsqueeze(2) * extend_base
            W_list.append(extend_W_1 + W_2)
        elif len(W_1.shape) == 3 and len(W_2.shape) == 2:
            batch_size, out_dim = W_2.shape
            extend_base = torch.eye(out_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(W_2.device)
            extend_W_2 = W_2.unsqueeze(2) * extend_base
            W_list.append(W_1 + extend_W_2)
        else:
            raise ValueError('Invalid tensor shape of W: %s and %s' % (W_1.shape, W_2.shape))

    for W, m1, m2 in zip(W_list_1[common_length:], m1_list_1[common_length:], m2_list_1[common_length:]):

        W_list.append(W)
        m1_list.append(m1)
        m2_list.append(m2)

    return W_list, m1_list, m2_list

