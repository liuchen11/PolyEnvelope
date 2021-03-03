import sys
import torch
import torch.nn as nn

import numpy as np

from .utility import reduced_bm_bv, logit_diff_quad, logit_diff_simp
from .attack import PGM

'''
>>> Calculate the distance between a data point and a linear decision boundary
'''
def calc_distance(W1, W2, p1, p2, x, c, norm, pixel_range = None):
    '''
    >>> W1, W2, p1, p2: W1x + p1 <= z_c - z_i <= W2x + p2
        W1, W2: of shape [batch_size, out_dim] or [batch_size, out_dim, in_dim]
        p1, p2: of shape [batch_size, out_dim]
        x: of shape [batch_size, in_dim]
    >>> c: of shape [batch_size]
    >>> norm: float, the norm of distance metric
    >>> pixel_range [min, max], allowable pixel range, default = None

    Return
    >>> distances: tensor of shape [batch_size, out_dim]
    '''

    assert norm > 0, 'invalid norm: %s' % norm
    primal_norm = norm
    dual_norm = 1. / (1. - 1. / primal_norm)

    x_ext = x.unsqueeze(2)              # of shape [batch_size, in_dim, 1]

    if pixel_range == None:

        if len(W1.shape) == 2:
            W1x = W1 * x                # of shape [batch_size, out_dim]
            devisor = W1x + p1          # of shape [batch_size, out_dim]
            devidend = W1               # of shape [batch_size, out_dim]
        else:
            devisor = torch.matmul(W1, x_ext).squeeze(2) + p1 # of shape [batch_size, out_dim]
            devidend = W1.norm(p = dual_norm, dim = 2)        # of shape [batch_size, out_dim]

        distance = devisor / (devidend + 1e-8)

    else:

        p_min, p_max = pixel_range
        perb_min = (p_min - x).unsqueeze(1)             # of shape [batch_size, 1, in_dim]
        perb_max = (p_max - x).unsqueeze(1)             # of shape [batch_size, 1, in_dim]

        assert len(W1.shape) == 3 and len(W2.shape) == 3, 'W1, W2 should have the shape [batch_size, out_dim, in_dim]'

        a = W1                                          # of shape [batch_size, out_dim, in_dim]
        b = p1 + torch.matmul(a, x_ext).squeeze(2)      # of shape [batch_size, out_dim]

        fix_min = torch.zeros_like(a)
        fix_max = torch.zeros_like(a)
        if primal_norm == np.inf:
            delta = - b.unsqueeze(2) * torch.sign(a) / (a.norm(p = 1, dim = 2, keepdim = True) + 1e-8) 
        elif primal_norm > 2. - 1e-6 and primal_norm < 2. + 1e-6:
            delta = - b.unsqueeze(2) * a / ((a * a).sum(dim = 2, keepdim = True) + 1e-8)
        else:
            delta = - b.unsqueeze(2) * torch.sign(a) * torch.abs(a) ** (dual_norm / primal_norm) / ((a.norm(p = dual_norm, dim = 2, keepdim = True) ** dual_norm) + 1e-8)

        iter_max = 20
        iter_idx = 0

        while ((perb_min - 1e-8) < delta).all().item() != 1 or (delta < (perb_max + 1e-8)).all().item() != 1:
            iter_idx += 1
            if iter_idx >= iter_max:
                print('WARNING: Failed to obtain optimal in %d iterations' % iter_max)
                break
            fix_min = fix_min + (delta < (perb_min - 1e-8)).float()
            fix_max = fix_max + ((perb_max + 1e-8) < delta).float()
            fix_delta = fix_min * (torch.max(delta, perb_min) + 1e-6) + fix_max * (torch.min(delta, perb_max) - 1e-6)   # of shape [batch_size, out_dim, in_dim]
            flex_mask = 1. - fix_min - fix_max
            ai = a * flex_mask
            bi = torch.sum(a * fix_delta, dim = 2) + b
            if primal_norm == np.inf:
                flex_delta = - bi.unsqueeze(2) * torch.sign(ai) * flex_mask / (ai.norm(p = 1, dim = 2, keepdim = True) + 1e-8)
            elif primal_norm > 2. - 1e-6 and primal_norm < 2. + 1e-6:
                flex_delta = - bi.unsqueeze(2) * ai * flex_mask / ((ai * ai).sum(dim = 2, keepdim = True) + 1e-8)
            else:
                flex_delta = - bi.unsqueeze(2) * (torch.sign(ai) * torch.abs(ai) ** (dual_norm / primal_norm)) * (1. - fix_min - fix_max) / (ai.norm(p = dual_norm, dim = 2, keepdim = True) ** dual_norm + 1e-8)
            delta = flex_delta + fix_delta

        distance = delta.norm(p = primal_norm, dim = 2) * torch.sign(b)

    return distance

'''
>>> This function calculate the Polyhedral Envelope regularization term
'''
def per(model, bound_est, x, T, c, norm, alpha, gamma, eps, at = False, pixel_range = None, criterion = nn.CrossEntropyLoss(), bound_calc_per_batch = None, is_certify = 0):
    '''
    >>> model: nn.Module, the network model
    >>> bound_est: tensor of shape [batch_size, out_dim], the bounding algorithm
    >>> x: tensor of shape [batch_size, in_dim], the input batch
    >>> T: int, the top N distances to take into account
    >>> c: tensor of shape [batch_size,], the true label
    >>> norm: float, the norm of distance metric
    >>> alpha: float, the normalizer
    >>> gamma: float, the coefficient to balance basic loss and regularization
    >>> eps: float, the input perturbation
    >>> at: whether or not to use adversarial attack
    >>> pixel_range: [min, max], allowable pixel range, default = None

    Return
    >>> loss: scalar, the total loss
    >>> distance: tensor of shape [batch_size], guaranteed distances
    '''
    # Flatten the input
    bound_calc_num = bound_calc_per_batch if bound_calc_per_batch != None else x.size(0)
    in_shape = [bound_calc_num,] + list(x.shape[1:])
    x_flat = x.view(x.size(0), -1).detach()
    x_ext = x_flat.unsqueeze(2)

    # Construct W1x + p1 < z_c - z_i < W2x + p2
    if bound_est.lower() in ['bound_quad',]:
        _, _, W_list, m1_list, m2_list = model.bound_quad(x_flat[:bound_calc_num], ori_perturb_norm = norm, ori_perturb_eps = eps,
            in_shape = in_shape, pixel_range = pixel_range, is_certify = is_certify)
        W_list, m1_list, m2_list = logit_diff_quad(W_list, m1_list, m2_list, c[:bound_calc_num])
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
        _, _, W, p1, p2 = model.bound_simp(x_flat[:bound_calc_num], ori_perturb_norm = norm, ori_perturb_eps = eps,
            in_shape = in_shape, pixel_range = pixel_range, is_certify = is_certify)
        W, p1, p2 = logit_diff_simp(W, p1, p2, c[:bound_calc_num])
        W1 = W2 = W
    else:
        raise ValueError('Unrecognized bounding algorithm.')

    if at == False:
        distance = calc_distance(W1 = W1, W2 = W2, p1 = p1, p2 = p2, x = x_flat[:bound_calc_num], c = c[:bound_calc_num], norm = norm, pixel_range = pixel_range)
        distance.scatter_(dim = 1, index = c[:bound_calc_num].view(-1, 1), src = torch.ones_like(c[:bound_calc_num]).float().view(-1, 1) * (-100.)) # Filter out the trivial hyperplane
        general_loss = criterion(model(x), c)
    else:
        attacker = PGM(step_size = eps / 5., threshold = eps, iter_num = 10, order = norm, pixel_range = pixel_range)
        optimizer = torch.optim.SGD(model.parameters(), lr = 1.)

        at_num = x.size(0) // 2
        at_bound_calc_num = bound_calc_num // 2

        at_x = attacker.attack(model, optimizer, data_batch = x[at_bound_calc_num:at_num + at_bound_calc_num], label_batch = c[at_bound_calc_num:at_num + at_bound_calc_num], criterion = criterion)
        at_x = torch.cat([x[:at_bound_calc_num], at_x, x[at_num + at_bound_calc_num:]], dim = 0)
        at_x_flat = at_x.view(at_x.size(0), -1)
        distance = calc_distance(W1 = W1, W2 = W2, p1 = p1, p2 = p2, x = at_x_flat[:bound_calc_num], c = c[:bound_calc_num], norm = norm, pixel_range = pixel_range)
        distance.scatter_(dim = 1, index = c[:bound_calc_num].view(-1, 1), src = torch.ones_like(c[:bound_calc_num]).float().view(-1, 1) * (-100.)) # Filter out the trivial hyperplane
        general_loss = criterion(model(at_x), c)

    # Obtain the shortest T distances
    topT, _ = distance.topk(dim = 1, k = T + 1, largest = False)        # of shape [batch_size, T + 1]
    topT = topT[:, 1:]

    lossT = torch.clamp(1. - topT / alpha, min = 0.)
    per_loss = lossT.sum(dim = 1).mean()

    # Obtain the guaranteed distances
    guaranteed_distances = torch.clamp(topT[:, 0], min = 0., max = eps)

    loss = general_loss + gamma * per_loss

    return loss, guaranteed_distances

