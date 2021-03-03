import os
import sys
sys.path.insert(0, './')

import pickle

import torch
import torch.nn as nn

from util.evaluation import AverageCalculator, accuracy
from util.per import per
from util.crown import crown_loss
from util.ibp import calc_ibp_loss

def train_test(model, train_loader, test_loader, attacker, epoch_num, optimizer,
    out_folder, model_name, alpha_list, eps_list, gamma_list, bound_est,
    T, norm, device, criterion, tosave, at_per = False, pixel_range = None, update_freq = 1, **tricks):
    '''
    >>> General training function without validation set
    '''

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()
    safe_distance_calculator = AverageCalculator()
    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    global_batch_idx = 0        # use to update the parameter
    for epoch_idx in range(epoch_num):

        alpha = alpha_list[epoch_idx]
        eps = eps_list[epoch_idx]
        gamma = gamma_list[epoch_idx]

        acc_calculator.reset()
        loss_calculator.reset()
        safe_distance_calculator.reset()

        model.train()
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            sys.stdout.write('Batch_idx = %d\r' % idx)

            if 'lr_func' in tricks and tricks['lr_func'] != None:
                lr_func = tricks['lr_func']
                local_idx = epoch_idx if 'train_batch_per_epoch' not in tricks else epoch_idx + float(idx) / tricks['train_batch_per_epoch']
                local_lr = lr_func(local_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = local_lr

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            # Play adversarial attack
            if attacker != None:
                attack_num = data_batch.shape[0] // 2
                data_batch_attack = attacker.attack(model, optimizer, data_batch[:attack_num], label_batch[:attack_num], criterion)
                data_batch = torch.cat([data_batch_attack, data_batch[attack_num:]], dim = 0)

            logits = model(data_batch)
            if gamma > 1e-6:        # Turn on per
                if 'bound_calc_per_batch' in tricks and tricks['bound_calc_per_batch'] != None:
                    ins_num = tricks['bound_calc_per_batch']
                    if 'regularize_mode' not in tricks or tricks['regularize_mode'] == 'per':
                        loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch,
                            T = T, c = label_batch, norm = norm, alpha = alpha, gamma = gamma, eps = eps, at = at_per,
                            pixel_range = pixel_range, criterion = criterion, bound_calc_per_batch = ins_num)
                    elif tricks['regularize_mode'] == 'kw':
                        loss = crown_loss(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
                            norm = norm, gamma = gamma, eps = eps, at = at_per, pixel_range = pixel_range,
                            criterion = criterion, bound_calc_per_batch = ins_num)
                        safe_distances = torch.zeros_like(label_batch).float()
                    else:
                        raise ValueError('Invalid regularizer_mode')
                else:
                    if 'regularize_mode' not in tricks or tricks['regularize_mode'] == 'per':
                        loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch,
                            T = T, c = label_batch, norm = norm, alpha = alpha, gamma = gamma,
                            eps = eps, at = at_per, pixel_range = pixel_range, criterion = criterion)
                    elif tricks['regularize_mode'] == 'kw':
                        loss = crown_loss(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
                            norm = norm, gamma = gamma, eps = eps, at = at_per, pixel_range = pixel_range, criterion = criterion)
                        safe_distances = torch.zeros_like(label_batch).float()
                    else:
                        raise ValueError('Invalid regularizer_mode')
            else:
                loss = criterion(logits, label_batch)
                safe_distances = torch.zeros_like(label_batch).float()

            global_batch_idx += 1
            loss.backward(retain_graph = True)
            if global_batch_idx % update_freq == 0:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.grad.data = p.grad.data / update_freq
                        p.grad.data = torch.clamp(p.grad.data, min = -0.5, max = 0.5)         # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))
            safe_distance_calculator.update(safe_distances.mean().item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        safe_distance_this_epoch = safe_distance_calculator.average
        print('Train loss/acc after epoch %d: %.4f/%.2f%%'%(epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        print('Train safe distance: %.4f'%safe_distance_this_epoch)
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch
        tosave['train_safe_distance'][epoch_idx] = safe_distance_this_epoch

        acc_calculator.reset()
        loss_calculator.reset()
        safe_distance_calculator.reset()

        model.eval()
        guaranteed_distance_list = []
        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            logits = model(data_batch)
            if gamma > 1e-6:            # Turn on per
                if 'bound_calc_per_batch' in tricks and tricks['bound_calc_per_batch'] != None:
                    ins_num = tricks['bound_calc_per_batch']
                    if 'regularize_mode' not in tricks or tricks['regularize_mode'] == 'per':
                        loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch,
                            T = T, c = label_batch, norm = norm, alpha = alpha, gamma = gamma, eps = eps, at = at_per,
                            pixel_range = pixel_range, criterion = criterion, bound_calc_per_batch = ins_num)
                    elif tricks['regularize_mode'] == 'kw':
                        loss = crown_loss(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
                            norm = norm, gamma = gamma, eps = eps, at = at_per, pixel_range = pixel_range,
                            criterion = criterion, bound_calc_per_batch = ins_num)
                        safe_distances = torch.zeros_like(label_batch).float()
                    else:
                        raise ValueError('Invalid regularizer_mode')
                else:
                    if 'regularize_mode' not in tricks or tricks['regularize_mode'] == 'per':
                        loss, safe_distances = per(model = model, bound_est = bound_est, x = data_batch,
                            T = T, c = label_batch, norm = norm, alpha = alpha, gamma = gamma,
                            eps = eps, at = at_per, pixel_range = pixel_range, criterion = criterion)
                    elif tricks['regularize_mode'] == 'kw':
                        loss = crown_loss(model = model, bound_est = bound_est, x = data_batch, c = label_batch,
                            norm = norm, gamma = gamma, eps = eps, at = at_per, pixel_range = pixel_range, criterion = criterion)
                        safe_distances = torch.zeros_like(label_batch).float()
                    else:
                        raise ValueError('Invalid regularizer_mode')
            else:
                loss = criterion(logits, label_batch)
                safe_distances = torch.zeros_like(label_batch).float()

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))
            safe_distance_calculator.update(safe_distances.mean().item(), data_batch.size(0))

            guaranteed_distance_list += list(safe_distances.data.cpu().numpy())

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        safe_distance_this_epoch = safe_distance_calculator.average
        print('Test loss/acc after epoch %d: %.4f/%.2f%%'%(epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        print('Test safe distance: %.4f'%safe_distance_this_epoch)
        tosave['test_loss'][epoch_idx] = loss_this_epoch
        tosave['test_acc'][epoch_idx] = acc_this_epoch
        tosave['test_safe_distance'][epoch_idx] = safe_distance_this_epoch
        tosave['guaranteed_distances'] = guaranteed_distance_list

        pickle.dump(tosave, open(os.path.join(out_folder, '%s.pkl'%model_name), 'wb'))
        torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt'%model_name))

    pickle.dump(tosave, open(os.path.join(out_folder, '%s.pkl'%model_name), 'wb'))
    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt'%model_name))

def train_test_ibp(model, train_loader, test_loader, attacker, epoch_num, optimizer, out_folder,
    model_name, alpha_list, beta_list, eps_list, norm, device, criterion, tosave, pixel_range = None, **tricks):
    '''
    >>> General training function using IBP
    '''

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()
    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    for epoch_idx in range(epoch_num):

        alpha = alpha_list[epoch_idx]
        beta = beta_list[epoch_idx]
        eps = eps_list[epoch_idx]

        print('alpha = %1.2e, beta = %1.2e, eps = %1.2e' % (alpha, beta, eps))

        acc_calculator.reset()
        loss_calculator.reset()

        model.train()
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            sys.stdout.write('Batch_idx = %d\r' % idx)

            if 'lr_func' in tricks and tricks['lr_func'] != None:
                lr_func = tricks['lr_func']
                local_idx = epoch_idx if 'train_batch_per_epoch' not in tricks else epoch_idx + float(idx) / tricks['train_batch_per_epoch']
                local_lr = lr_func(local_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = local_lr

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            if attacker != None:
                attack_num = data_batch.shape[0] // 2
                data_batch_attack = attacker.attack(model, optimizer, data_batch[:attack_num], label_batch[:attack_num], criterion)
                data_batch = torch.cat([data_batch_attack, data_batch[attack_num:]], dim = 0)

            logits = model(data_batch)
            if 'bound_calc_per_batch' in tricks and tricks['bound_calc_per_batch'] != None:
                loss = calc_ibp_loss(model, data_batch, label_batch, norm, eps, pixel_range, alpha, beta, criterion, tricks['bound_calc_per_batch'])
            else:
                loss = calc_ibp_loss(model, data_batch, label_batch, norm, eps, pixel_range, alpha, beta, criterion)
            loss.backward()
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data = torch.clamp(p.grad.data, min = -0.5, max = 0.5)
            optimizer.step()
            optimizer.zero_grad()

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Train loss/acc after epoch %d: %.4f/%.2f%%'%(epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch

        acc_calculator.reset()
        loss_calculator.reset()

        model.eval()
        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            logits = model(data_batch)
            if 'bound_calc_per_batch' in tricks and tricks['bound_calc_per_batch'] != None:
                loss = calc_ibp_loss(model, data_batch, label_batch, norm, eps, pixel_range, alpha, beta, criterion, tricks['bound_calc_per_batch'])
            else:
                loss = calc_ibp_loss(model, data_batch, label_batch, norm, eps, pixel_range, alpha, beta, criterion)

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Test loss/acc after epoch %d: %.4f/%.2f%%'%(epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['test_loss'][epoch_idx] = loss_this_epoch
        tosave['test_acc'][epoch_idx] = acc_this_epoch

        pickle.dump(tosave, open(os.path.join(out_folder, '%s.pkl'%model_name), 'wb'))
        torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt'%model_name))

    pickle.dump(tosave, open(os.path.join(out_folder, '%s.pkl'%model_name), 'wb'))
    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt'%model_name))

