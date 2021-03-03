import os
import sys
sys.path.insert(0, './')

import pickle

import torch
import torch.nn as nn

from util.per import per
from util.crown import crown
from util.evaluation import accuracy

def certify_this_batch(model, data_batch, label_batch, certify_mode, eps, bound_est, norm, pixel_range):
    '''
    >>> give the certified bound of a mini-batch
    '''

    # trivial cases
    if eps <= 1e-8:
        return [0. for _ in label_batch]

    if certify_mode.lower() in ['lin', 'crown']:
        l, u = crown(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
            norm = norm, eps = eps, pixel_range = pixel_range)
        margin, _ = torch.min(l, dim = 1)
        success_bits_this_batch = (margin > -1e-8).data.cpu().numpy()
        eps_this_batch = [eps * flag for flag in success_bits_this_batch]
    elif certify_mode.lower() in ['per',]:
        per_loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
            T = 1, norm = norm, alpha = 1., gamma = 0., eps = eps, pixel_range = pixel_range)
        eps_this_batch = list(safe_distances.data.cpu().numpy())
    else:
        raise ValueError('Certify mode %s is not supported' % certify_mode)

    return eps_this_batch

def search_eps(model, data_loader, min_idx, max_idx, min_eps, max_eps, precision_eps,
    certify_mode, bound_est, norm, pixel_range, device, **tricks):
    '''
    >>> Find the optimal eps
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    tosave = {}
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Processing batch: %d\r' % idx)
        sys.stdout.flush()

        if (min_idx != None and idx < min_idx) or (max_idx != None and idx >= max_idx):
            continue

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)
        acc = accuracy(logits.data, label_batch)

        batch_size = data_batch.shape[0]
        assert batch_size == 1, 'The batch size should be 1 in this case, but %d found' % batch_size

        call_time = 0           # number of attempt to calculate certified bounds

        # eps_this_batch = certify_this_batch(model = model, data_batch = data_batch, label_batch = label_batch,
        #     certify_mode = certify_mode, eps = max_eps, bound_est = bound_est, norm = norm, pixel_range = pixel_range)
        # call_time += 1
        # if eps_this_batch[0] > max_eps - 1e-8:
        #     tosave[idx] = {'eps': max_eps, 'call_time': 1, 'acc': 1 if acc.item() > 0.5 else 0}
        #     continue
        # up_eps = max_eps
        # low_eps = eps_this_batch[0]

        # eps_this_batch = certify_this_batch(model = model, data_batch = data_batch, label_batch = label_batch,
        #     certify_mode = certify_mode, eps = min_eps, bound_est = bound_est, norm = norm, pixel_range = pixel_range)
        # call_time += 1
        # assert eps_this_batch[0] > (min_eps - 1e-8), 'data cannot be certified when eps = %1.2e' % min_eps
        # low_eps = max(low_eps, eps_this_batch[0])

        up_eps = max_eps
        low_eps = min_eps

        while up_eps - low_eps > precision_eps:

            attempt_eps = (up_eps + low_eps) / 2.
            eps_this_batch = certify_this_batch(model = model, data_batch = data_batch, label_batch = label_batch,
                certify_mode = certify_mode, eps = attempt_eps, bound_est = bound_est, norm = norm, pixel_range = pixel_range)
            call_time += 1

            low_eps = max(low_eps, eps_this_batch[0])
            if eps_this_batch[0] < attempt_eps - 1e-6:
                up_eps = attempt_eps

        tosave[idx] = {'eps': (low_eps + up_eps) / 2., 'call_time': call_time, 'acc': 1 if acc.item() > 0.5 else 0}

    return tosave

