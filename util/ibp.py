import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_ibp_loss(model, data_batch, label_batch, perturb_norm, perturb_eps, pixel_range,
    alpha, beta, criterion = nn.CrossEntropyLoss(), bound_calc_per_batch = None):

    # Normal Loss
    logits = model(data_batch)
    base_loss = criterion(logits, label_batch)

    if perturb_eps > 1e-6:
        # Bound estimation
        only_ibp = True if beta > 1. - 1e-6 else False
        calc_num = bound_calc_per_batch if bound_calc_per_batch != None else data_batch.size(0)
        ibp_l, ibp_u, crown_l, crown_u = model.crown_ibp(data_batch[:calc_num], label_batch[:calc_num], perturb_norm, perturb_eps,
            in_shape = data_batch[:calc_num].shape, pixel_range = pixel_range, only_ibp = only_ibp)

        logit_bound = beta * (-ibp_l) + (1. - beta) * (-crown_l) if only_ibp == False else -ibp_l
        ibp_loss = criterion(logit_bound, label_batch[:calc_num])

        loss = alpha * base_loss + (1. - alpha) * ibp_loss
    else:
        loss = base_loss

    return loss


def calc_ibp_certify(model, data_batch, label_batch, perturb_norm, perturb_eps, pixel_range):
    '''
    >>> return the success rate given by bounds from IBP and CROWN-IBP
    '''

    ibp_l, ibp_u, crown_l, crown_u = model.crown_ibp(data_batch, label_batch, perturb_norm, perturb_eps,
        in_shape = data_batch.shape, pixel_range = pixel_range, only_ibp = False)

    gap_ibp, _ = ibp_l.min(dim = 1)
    gap_crown, _ = crown_l.min(dim = 1)

    results_ibp = (gap_ibp > -1e-6).float()
    results_crown = (gap_crown > -1e-6).float()

    return results_ibp, results_crown
