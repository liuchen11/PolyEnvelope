import os
import sys
sys.path.insert(0, './')

import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from util.ibp import calc_ibp_certify
from util.attack import PGM
from util.crown import crown
from util.per import per
from util.evaluation import AverageCalculator, accuracy

from kw.convex_adversarial import robust_loss

def certify_per(model, data_loader, out_file, eps, norm, bound_est, device, tosave, pixel_range = None, **tricks):
    '''
    >>> Certification function using per
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    model.eval()
    guaranteed_distance_list = []
    acc_calculator = AverageCalculator()
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Batch Index = %d\r' % idx)

        if 'batch_num' in tricks and idx >= tricks['batch_num'] and tricks['batch_num'] > 0:
            print('The certification process stops after %d batches' % tricks['batch_num'])
            break

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)
        criterion = nn.CrossEntropyLoss() if use_gpu == False else nn.CrossEntropyLoss().cuda()
        total_loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch,
            T = 1, c = label_batch, norm = norm, alpha = 1., gamma = 0., eps = eps, at = False,
            pixel_range = pixel_range, criterion = criterion, is_certify = 0)

        for is_certify in [1, 2, 3]:
            _, safe_distances_this_mode = per(model = model, bound_est = bound_est, x = data_batch,
                T = 1, c = label_batch, norm = norm, alpha = 1., gamma = 0., eps = eps, at = False,
                pixel_range = pixel_range, criterion = criterion, is_certify = is_certify)
            safe_distances = torch.max(safe_distances, safe_distances_this_mode)

        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))
        guaranteed_distance_list += list(safe_distances.data.cpu().numpy())

    print('')

    acc_this_epoch = acc_calculator.average
    tosave['guaranteed_distances'] = guaranteed_distance_list

    success_bits = [1. if d > eps - 1e-6 else 0. for d in guaranteed_distance_list]

    print('>>>>> The results of PEC <<<<<')
    print('Average Accuracy: %.2f%%' % (acc_this_epoch * 100.))
    print('Average Certified Distances: %.4f' % np.mean(guaranteed_distance_list))
    print('Certified Bounds over %.4f: %.2f%%' % (eps, np.mean(success_bits) * 100.))

    if out_file != None:
        pickle.dump(tosave, open(out_file, 'wb'))    

    return tosave

def certify_crown(model, data_loader, out_file, eps, norm, bound_est, device, tosave, pixel_range = None, **tricks):
    '''
    >>> Certification function using Fast-Lin / CROWN
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    model.eval()
    success_bits = []
    acc_calculator = AverageCalculator()
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Batch Index = %d\r' % idx)

        if 'batch_num' in tricks and idx >= tricks['batch_num'] and tricks['batch_num'] > 0:
            print('The certification process stops after %d batches' % tricks['batch_num'])
            break

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)

        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))

        l, u = crown(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
            norm = norm, eps = eps, pixel_range = pixel_range)

        margin, _ = torch.min(l, dim = 1)
        success_bits_this_batch = (margin > -1e-8).data.cpu().numpy()
        success_bits += list(success_bits_this_batch)

    print('')

    acc_this_epoch = acc_calculator.average
    tosave['guaranteed_distances'] = [eps * flag for flag in success_bits]

    print('>>> The results of Fast-Lin / CROWN <<<<')
    print('Average Accuracy: %.2f%%' % (acc_this_epoch * 100.))
    print('Average Certified Distances: %.4f' % (eps * np.mean(success_bits)))
    print('Certified Bounds over %.4f: %.2f%%' % (eps, np.mean(success_bits) * 100.))

    if out_file != None:
        pickle.dump(tosave, open(out_file, 'wb'))

    return tosave

def certify_ibp(model, data_loader, out_file, eps, norm, bound_est, device, tosave, pixel_range = None, **tricks):
    '''
    >>> Certification function using IBP/CROWN-IBP
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    model.eval()
    success_bits_ibp = []
    success_bits_crown = []
    acc_calculator = AverageCalculator()
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Batch Index = %d\r' % idx)

        if 'batch_num' in tricks and idx >= tricks['batch_num'] and tricks['batch_num'] > 0:
            print('The certification process stops after %d batches' % tricks['batch_num'])
            break

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)

        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))

        results_ibp_this_batch, results_crown_this_batch = calc_ibp_certify(model = model, data_batch = data_batch, label_batch = label_batch,
            perturb_norm = norm, perturb_eps = eps, pixel_range = pixel_range)
        results_ibp_this_batch = results_ibp_this_batch.data.cpu().numpy()
        results_crown_this_batch = results_crown_this_batch.data.cpu().numpy()

        success_bits_ibp += list(results_ibp_this_batch)
        success_bits_crown += list(results_crown_this_batch)

    print('')

    acc_this_epoch = acc_calculator.average
    tosave['guaranteed_distances_ibp'] = [eps * flag for flag in success_bits_ibp]
    tosave['guaranteed_distances_crownibp'] = [eps * flag for flag in success_bits_crown]

    print('>>> The results of IBP/CROWN-IBP <<<')
    print('Average Accuracy: %.2f%%' % (acc_this_epoch * 100.))
    print('Average Certified Distances by IBP: %.4f' % (eps * np.mean(success_bits_ibp)))
    print('Certified Bounds over %.4f: %.2f%%' % (eps, np.mean(success_bits_ibp) * 100.))
    print('Average Certified Distances by CROWN-IBP: %.4f' % (eps * np.mean(success_bits_crown)))
    print('Certified Bounds over %.4f: %.2f%%' % (eps, np.mean(success_bits_crown) * 100.))

    if out_file != None:
        pickle.dump(tosave, open(out_file, 'wb'))

    return tosave

def certify_kw(model, data_loader, out_file, eps, norm, bound_est, device, tosave, pixel_range = None, **tricks):
    '''
    >>> Certification function using Kolter-Wong's framework
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()
    if pixel_range == None:
        bounded_input = False
    else:
        assert pixel_range[0] == 0. and pixel_range[1] == 1., 'pixel_range %s is not supported' % pixel_range
        bounded_input = True

    model.eval()
    seq_model = model.model2sequence()

    acc_calculator = AverageCalculator()
    success_bits = []
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Batch Index = %d\r' % idx)

        if 'batch_num' in tricks and idx >= tricks['batch_num'] and tricks['batch_num'] > 0:
            print('The certification process stops after %d batches' % tricks['batch_num'])
            break

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)

        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))

        norm_type = {2.: 'l2', np.inf: 'l1'}[norm]
        _, robust_err = robust_loss(seq_model, eps, data_batch, label_batch,
            norm_type = norm_type, bounded_input = bounded_input, size_average = False) # of shape [batch_size]
        success_bits_this_batch = (robust_err.data.float() < 1e-8).data.cpu().numpy()
        success_bits += list(success_bits_this_batch)

    print('')

    acc_this_epoch = acc_calculator.average
    tosave['guaranteed_distances'] = [eps * flag for flag in success_bits]

    print('>>>>> The results of KW <<<<<')
    print('Average Accuracy: %.2f%%' % (acc_this_epoch * 100.))
    print('Average Certified Distances: %.4f' % (eps * np.mean(success_bits)))
    print('Certified Bounds over %.4f: %.2f%%' % (eps, np.mean(success_bits) * 100.))

    if out_file != None:
        pickle.dump(tosave, open(out_file, 'wb'))

    return tosave

def certify_pgd(model, data_loader, out_file, eps, norm, bound_est, device, tosave, pixel_range = None, **tricks):
    '''
    >>> Accuracy under adversarial attack, providing the upper bound of certified accuracy
    '''

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    attacker = PGM(step_size = eps / 10., threshold = eps, iter_num = 20, order = norm, pixel_range = pixel_range)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1.)

    model.eval()
    acc_calculator = AverageCalculator()
    robust_acc_calculator = AverageCalculator()
    normal_success_bits = []
    robust_success_bits = []
    for idx, (data_batch, label_batch) in enumerate(data_loader, 0):

        sys.stdout.write('Batch Index = %d\r' % idx)

        if 'batch_num' in tricks and idx >= tricks['batch_num'] and tricks['batch_num'] > 0:
            print('The test process stops after %d batches' % tricks['batch_num'])
            break

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)

        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))
        normal_success_bits_this_batch = (torch.argmax(logits, dim = 1) == label_batch).float().data.cpu().numpy()
        normal_success_bits += list(normal_success_bits_this_batch)

        data_batch = attacker.attack(model, optimizer, data_batch, label_batch)

        logits = model(data_batch)

        acc = accuracy(logits.data, label_batch)
        robust_acc_calculator.update(acc.item(), data_batch.size(0))
        robust_success_bits_this_batch = (torch.argmax(logits, dim = 1) == label_batch).float().data.cpu().numpy()
        robust_success_bits += list(robust_success_bits_this_batch)

    print('')

    acc_this_epoch = acc_calculator.average
    robust_acc_this_epoch = robust_acc_calculator.average
    tosave['normal_success_bits'] = normal_success_bits
    tosave['robust_success_bits'] = robust_success_bits

    print('>>>>> The results of PGD <<<<<')
    print('Average Accuracy: %.2f%%' % (acc_this_epoch * 100.))
    print('Robust Accuracy: %.2f%%' % (robust_acc_this_epoch * 100.))

    if out_file != None:
        pickle.dump(tosave, open(out_file, 'wb'))

    return tosave
