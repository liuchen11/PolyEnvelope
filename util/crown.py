import sys
import torch
import torch.nn as nn

import numpy as np
from .utility import reduced_bm_bv, norm_based_bounds, logit_diff_quad, logit_diff_simp
from .attack import PGM

'''
>>> calculate the upper and lower and upper bound of logit differences
'''
def calc_bounds(W1, W2, p1, p2, x, c, norm, eps, pixel_range = None):
    '''
    >>> W1, W2, p1, p2: W1x + p1 <= z_c - z <= W2 + p2
        W1, W2: of shape [batch_size, in_dim] or [batch_size, out_dim, in_dim]
        p1, p2: of shape [batch_size, in_dim]
    >>> c: of shape [batch_size,]
    >>> norm: float, the norm of distance metric
    >>> eps: float, the adversarial budget
    >>> pixel_range: [min, max], allowable pixel range, default = None

    Return
    >>> min_diff, max_diff: tensor of shape [batch_size, out_dim]
    '''

    primal_norm = norm
    dual_norm = 1. / (1. - 1. / primal_norm)

    min_diff, max_diff = norm_based_bounds(x = x, W = W1, perturb_norm = norm,
        perturb_eps = eps, pixel_range = pixel_range)
    min_diff += p1
    max_diff += p1

    return min_diff, max_diff

'''
>>> This function calculate the CROWN upper and lower bound of logit differences
'''
def crown(model, bound_est, x, c, norm, eps, pixel_range = None):
    '''
    >>> model: nn.Module, the network model
    >>> bound_est: tensor of shape [batch_size, out_dim], the bounding algorithm
    >>> x: tensor of shape [batch_size, in_dim], the input batch
    >>> norm: float, the norm of distance metric
    >>> eps: float, the input perturbation
    >>> pixel_range: [min, max], allowable pixel range, default = None

    Return
    >>> min_diff, max_diff: tensor of shape [batch_size, out_dim]
    '''

    in_shape = list(x.shape)
    x = x.view(x.size(0), -1).detach()

    # Construct W1x + p1 < output < W2x + p2
    if bound_est.lower() in ['bound_quad',]:
        _, _, W_list, m1_list, m2_list = model.bound_quad(x, ori_perturb_norm = norm, ori_perturb_eps = eps,
            in_shape = in_shape, pixel_range = pixel_range)
        W_list, m1_list, m2_list = logit_diff_quad(W_list = W_list, m1_list = m1_list, m2_list = m2_list, c = c)
        W1 = W2 = W_list[0]
        p1 = 0.
        p2 = 0.
        for idx, (W, m1, m2) in enumerate(zip(W_list, m1_list, m2_list)):
            if idx == 0:
                continue
            W_pos = torch.clamp(W, min = 0.)
            W_neg = torch.clamp(W, max = 0.)
            p1 += reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)
            p2 += reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)
    elif bound_est.lower() in ['bound_simp',]:
        _, _, W, p1, p2 = model.bound_simp(x, ori_perturb_norm = norm, ori_perturb_eps = eps,
            in_shape = in_shape, pixel_range = pixel_range)
        W, p1, p2 = logit_diff_simp(W = W, p1 = p1, p2 = p2, c = c)
        W1 = W2 = W
    else:
        raise ValueError('Unrecognized bounding algorithm.')

    min_diff, max_diff = calc_bounds(W1 = W1, W2 = W2, p1 = p1, p2 = p2, x = x, c = c, norm = norm, eps = eps, pixel_range = pixel_range)

    return min_diff, max_diff

'''
>>> calculate the crown based loss
'''
def crown_loss(model, bound_est, x, c, norm, gamma, eps, at = False, pixel_range = None, criterion = nn.CrossEntropyLoss(), bound_calc_per_batch = None):

    bound_calc_num = bound_calc_per_batch if bound_calc_per_batch != None else x.size(0)

    # Calculate the KW loss
    min_diff, max_diff = crown(model, bound_est, x[:bound_calc_num], c[:bound_calc_num], norm, eps, pixel_range)
    kw_loss = criterion(-min_diff, c[:bound_calc_num])

    if at == False:
        general_loss = criterion(model(x), c)
    else:
        attacker = PGM(step_size = eps / 5., threshold = eps, iter_num = 10, order = norm, pixel_range = pixel_range)
        optimizer = torch.optim.SGD(model.parameters(), lr = 1.)

        at_num = x.size(0) // 2
        at_x = attacker.attack(model, optimizer, data_batch = x[:at_num], label_batch = c[:at_num], criterion = criterion)
        at_x = torch.cat([at_x, x[at_num:]], dim = 0)
        general_loss = criterion(model(at_x), c)

    loss = general_loss + gamma * kw_loss

    return loss


