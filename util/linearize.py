import torch
import torch.nn
import torch.nn.functional as F

# The minimum non-zero slope, to avoid numeric unstability
D_threshold = 1e-2

'''
>>> linearization function
>>> low_bound: torch.Tensor, of shape [batch_size, dim]
>>> up_bound: torch.Tensor, of shape [batch_size, dim]

return
>>> D: slope, of shape [batch_size, dim]
>>> m1: lower bound bias term, of shape [batch_size, dim]
>>> m2: upper bound bias term, of shape [batch_size, dim]
'''

def linearize_relu(low_bound, up_bound, is_certify = 0):
    '''
    slope = (relu(u) - relu(l)) / (u - l)
    m1 = 0
    m2 = relu(l) - (relu(u) - relu(l)) * l / (u - l)
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    D = (F.relu(up_bound) - F.relu(low_bound)) / (up_bound - low_bound)
    m1 = D * 0.
    m2 = F.relu(low_bound) - low_bound * D

    return D, m1, m2

def linearize_sigd(low_bound, up_bound, is_certify = 0):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = F.sigmoid(low_bound)
    y_up = F.sigmoid(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = -torch.log((- (2 * D - 1) + torch.sqrt(torch.clamp(1 - 4 * D, min = 1e-6)) + 1e-6)/ (2 * D + 1e-6))
    t2 = -t1

    y1 = F.sigmoid(t1) - t1 * D - 1e-6
    y2 = 1. - y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= t2).float()
    pos_mask = (low_bound >= t1).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    if is_certify == 1:

        t1_low = t1
        t1_up = 0.
        for _ in range(20):

            t1_try = (t1_low + t1_up) / 2.
            D_try = torch.exp(t1_try) / (1. + torch.exp(t1_try)) ** 2
            y1_try = F.sigmoid(t1_try) - D_try * t1_try
            y2_try = 1. - y1_try

            y_low_low = low_bound * D_try + y1_try
            y_low_up = low_bound * D_try + y2_try
            y_up_low = up_bound * D_try + y1_try
            y_up_up = up_bound * D_try + y2_try

            success_bit = neu_mask * ((y_low_low <= y_low).float() * (y_low <= y_low_up).float() * (y_up_low <= y_up).float() * (y_up <= y_up_up).float())

            # Update
            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + y1_try * success_bit
            m2 = m2 * (1. - success_bit) + y2_try * success_bit

            t1_low = t1_low * (1. - success_bit) + t1_try * success_bit
            t1_up = t1_up * success_bit + t1_try * (1. - success_bit)

    if is_certify == 2:

        t1_low = t1
        t1_up = 0.
        for _ in range(20):

            t1_try = (t1_low + t1_up) / 2.
            D_try = torch.exp(t1_try) / (1. + torch.exp(t1_try)) ** 2
            y1_try = F.sigmoid(t1_try) - D_try * t1_try

            y_low_low = low_bound * D_try + y1_try
            y_up_low = up_bound * D_try + y1_try

            success_bit = neu_mask * ((y_low_low <= y_low).float() * (y_up_low <= y_up).float())

            y2_try = 1. - y1_try
            y2_interec = y_low - D_try * low_bound

            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + y1_try * success_bit
            m2 = m2 * (1. - success_bit) + torch.max(y2_try, y2_interec) * success_bit

            t1_low = t1_low * (1. - success_bit) + t1_try * success_bit
            t1_up = t1_up * success_bit + t1_try * (1. - success_bit)

    if is_certify == 3:

        t2_low = 0
        t2_up = t2
        for _ in range(20):

            t2_try = (t2_low + t2_up) / 2.
            D_try = torch.exp(t2_try) / (1. + torch.exp(t2_try)) ** 2
            y2_try = F.sigmoid(t2_try) - D_try * t2_try

            y_low_up = low_bound * D_try + y2_try
            y_up_up = up_bound * D_try + y2_try

            success_bit = neu_mask * ((y_low_up >= y_low).float() * (y_up_up >= y_up).float())

            y1_try = 1. - y2_try
            y1_intersec = y_up - D_try * up_bound

            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + torch.min(y1_try, y1_intersec) * success_bit
            m2 = m2 * (1. - success_bit) + y2_try * success_bit

            t2_low = t2_low * success_bit + t2_try * (1. - success_bit)
            t2_up = t2_up * (1. - success_bit) + t2_try * success_bit


    return D, m1, m2

def linearize_tanh(low_bound, up_bound, is_certify = 0):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check the linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = F.tanh(low_bound)
    y_up = F.tanh(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = torch.log((- (D - 2) - 2 * torch.sqrt(torch.clamp(1. - D, min = 1e-6)) + 1e-6) / (D + 1e-6)) / 2.
    t2 = -t1

    y1 = F.tanh(t1) - t1 * D - 1e-6
    y2 = -y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= t2).float()
    pos_mask = (low_bound >= t1).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    if is_certify == 1:

        t1_low = t1
        t1_up = 0.
        for _ in range(20):

            t1_try = (t1_low + t1_up) / 2.
            D_try = 4. / (torch.exp(t1_try) + torch.exp(-t1_try)) ** 2
            y1_try = F.tanh(t1_try) - D_try * t1_try
            y2_try = - y1_try

            y_low_low = low_bound * D_try + y1_try
            y_low_up = low_bound * D_try + y2_try
            y_up_low = up_bound * D_try + y1_try
            y_up_up = up_bound * D_try + y2_try

            success_bit = neu_mask * ((y_low_low <= y_low).float() * (y_low <= y_low_up).float() * (y_up_low <= y_up).float() * (y_up <= y_up_up).float())

            # Update
            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + y1_try * success_bit
            m2 = m2 * (1. - success_bit) + y2_try * success_bit

            t1_low = t1_low * (1. - success_bit) + t1_try * success_bit
            t1_up = t1_up * success_bit + t1_try * (1. - success_bit)

    if is_certify == 2:

        t1_low = t1
        t1_up = 0.
        for _ in range(20):

            t1_try = (t1_low + t1_up) / 2.
            D_try = 4. / (torch.exp(t1_try) + torch.exp(-t1_try)) ** 2
            y1_try = F.tanh(t1_try) - D_try * t1_try

            y_low_low = low_bound * D_try + y1_try
            y_up_low = up_bound * D_try + y1_try

            success_bit = neu_mask * ((y_low_low <= y_low).float() * (y_up_low <= y_up).float())

            y2_try = - y1_try
            y2_interec = y_low - D_try * low_bound

            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + y1_try * success_bit
            m2 = m2 * (1. - success_bit) + torch.max(y2_try, y2_interec) * success_bit

            t1_low = t1_low * (1. - success_bit) + t1_try * success_bit
            t1_up = t1_up * success_bit + t1_try * (1. - success_bit)

    if is_certify == 3:

        t2_low = 0
        t2_up = t2
        for _ in range(20):

            t2_try = (t2_low + t2_up) / 2.
            D_try = 4. / (torch.exp(t2_try) + torch.exp(-t2_try)) ** 2
            y2_try = F.tanh(t2_try) - D_try * t2_try

            y_low_up = low_bound * D_try + y2_try
            y_up_up = up_bound * D_try + y2_try

            success_bit = neu_mask * ((y_low_up >= y_low).float() * (y_up_up >= y_up).float())

            y1_try = - y2_try
            y1_intersec = y_up - D_try * up_bound

            D = D * (1. - success_bit) + D_try * success_bit
            m1 = m1 * (1. - success_bit) + torch.min(y1_try, y1_intersec) * success_bit
            m2 = m2 * (1. - success_bit) + y2_try * success_bit

            t2_low = t2_low * success_bit + t2_try * (1. - success_bit)
            t2_up = t2_up * (1. - success_bit) + t2_try * success_bit


    return D, m1, m2

def linearize_arctan(low_bound, up_bound, is_certify = 0):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check the linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = torch.atan(low_bound)
    y_up = torch.atan(up_bound)
    low_sign = torch.sign(low_bound)
    up_sign = torch.sign(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = - torch.sqrt(1. / D - 1.)
    t2 = -t1

    y1 = torch.atan(t1) - t1 * D - 1e-6
    y2 = -y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= 0.).float()
    pos_mask = (low_bound >= 0.).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    return D, m1, m2

