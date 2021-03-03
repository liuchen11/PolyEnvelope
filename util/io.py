import os
import sys
sys.path.insert(0, './')
import argparse

import scipy
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn

def load_ckpt_model(model, name, ckpt_file, device, param2load = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> ckpt_file: the ckpt file
    >>> device: the device of the model
    '''

    if param2load == None:
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt)
    else:
        assert name.lower() in ['fc1',], 'The name of the model must be FC1'
        if param2load.lower() in ['conv',]:
            model.conv_block.conv_layer_0.layer.weight.data = ckpt['conv_block.conv_layer_0.layer.weight']
            model.conv_block.conv_layer_0.layer.bias.data = ckpt['conv_block.conv_layer_0.layer.bias']
            model.conv_block.conv_layer_1.layer.weight.data = ckpt['conv_block.conv_layer_1.layer.weight']
            model.conv_block.conv_layer_1.layer.bias.data = ckpt['conv_block.conv_layer_1.layer.bias']
        elif param2load.lower() in ['fc',]:
            model.fc_block.fc_layer_0.layer.weight.data = ckpt['fc_block.fc_layer_0.layer.weight']
            model.fc_block.fc_layer_0.layer.bias.data = ckpt['fc_block.fc_layer_0.layer.bias']
            model.output.layer.weight.data = ckpt['output.layer.weight']
            model.output.layer.bias.data = ckpt['output.layer.bias']
        else:
            raise ValueError('Unsupported param2load value: %s' % param2load)

    return model

def dump_ckpt_model(model, name, ckpt_file, device, param2dump = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> ckpt_file: the ckpt file
    >>> device: the device of the model
    '''

    raise NotImplementedError

def load_mat_model(model, name, mat_file, device, param2load = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> mat_file: the mat file
    >>> device: the device of the model
    '''

    assert os.path.exists(mat_file), 'The file %s does not exist.' % mat_file

    param_list = sio.loadmat(mat_file)

    if name.lower() in ['fc1',]:
        assert param2load == None, 'In FC1 mode, param2load should be None'
        model.main_block.layer_0.layer.weight.data = torch.from_numpy(param_list['U'].transpose()).float().to(device)
        model.main_block.layer_0.layer.bias.data = torch.from_numpy(param_list['bU'].reshape(-1)).float().to(device)
        model.output.layer.weight.data = torch.from_numpy(param_list['W'].transpose()).float().to(device)
        model.output.layer.bias.data = torch.from_numpy(param_list['bW'].reshape(-1)).float().to(device)
    elif name.lower() in ['lenet',]:
        if param2load == None or param2load.lower() in ['conv',]:
            model.conv_block.conv_layer_0.layer.weight.data = torch.from_numpy(param_list['weights_conv1'].transpose(3, 2, 0, 1)).float().to(device)            # (h, w, c_in, c_out) -> (c_out, c_in, h, w)
            model.conv_block.conv_layer_0.layer.bias.data = torch.from_numpy(param_list['biases_conv1'].reshape(-1)).float().to(device)
            model.conv_block.conv_layer_1.layer.weight.data = torch.from_numpy(param_list['weights_conv2'].transpose(3, 2, 0, 1)).float().to(device)            # (h, w, c_in, c_out) -> (c_out, c_in, h, w)
            model.conv_block.conv_layer_1.layer.bias.data = torch.from_numpy(param_list['biases_conv2'].reshape(-1)).float().to(device)

        if param2load == None or param2load.lower() in ['fc',]:
            first_weight_matrix = param_list['weights_fc1']
            in_dim, out_dim = first_weight_matrix.shape
            channel_size = int(np.sqrt(in_dim // model.conv_block.conv_layer_1.layer.weight.data.shape[0]))
            first_weight_matrix = first_weight_matrix.reshape(channel_size, channel_size, -1, out_dim).transpose(2, 0, 1, 3).reshape(-1, out_dim)
            model.fc_block.fc_layer_0.layer.weight.data = torch.from_numpy(first_weight_matrix.transpose()).float().to(device)

            model.fc_block.fc_layer_0.layer.bias.data = torch.from_numpy(param_list['biases_fc1'].reshape(-1)).float().to(device)
            model.output.layer.weight.data = torch.from_numpy(param_list['weights_fc2'].transpose()).float().to(device)
            model.output.layer.bias.data = torch.from_numpy(param_list['biases_fc2'].reshape(-1)).float().to(device)

        if param2load != None and param2load.lower() not in ['conv', 'fc']:
            raise ValueError('Unsupported param2load value: %s' % param2load)
    else:
        raise ValueError('Unsupported model type: %s' % name)

    return model

def dump_mat_model(model, name, mat_file, device, param2dump = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> mat_file: the mat file
    >>> device: the device of the model
    '''

    raise NotImplementedError

def load_pth_model(model, name, pth_file, device, param2load = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> pth_file: the pth file
    >>> device: the device of the model
    '''

    assert os.path.exists(pth_file), 'The file %s does not exist.' % pth_file

    full_dict = torch.load(pth_file)

    if name.lower() in ['fc1',]:
        assert param2load == None, 'In FC1 mode, param2load should be None'
        model.main_block.layer_0.layer.weight.data = full_dict['state_dict'][0]['1.weight'].to(device)
        model.main_block.layer_0.layer.bias.data = full_dict['state_dict'][0]['1.bias'].to(device)
        model.output.layer.weight.data = full_dict['state_dict'][0]['3.weight'].to(device)
        model.output.layer.bias.data = full_dict['state_dict'][0]['3.bias'].to(device)
    elif name.lower() in ['lenet',]:
        if param2load == None or param2load.lower() in ['conv',]:
            model.conv_block.conv_layer_0.layer.weight.data = full_dict['state_dict'][0]['0.weight'].to(device)
            model.conv_block.conv_layer_0.layer.bias.data = full_dict['state_dict'][0]['0.bias'].to(device)
            model.conv_block.conv_layer_1.layer.weight.data = full_dict['state_dict'][0]['2.weight'].to(device)
            model.conv_block.conv_layer_1.layer.bias.data = full_dict['state_dict'][0]['2.bias'].to(device)
        if param2load == None or param2load.lower() in ['fc',]:
            model.fc_block.fc_layer_0.layer.weight.data = full_dict['state_dict'][0]['5.weight'].to(device)
            model.fc_block.fc_layer_0.layer.bias.data = full_dict['state_dict'][0]['5.bias'].to(device)
            model.output.layer.weight.data = full_dict['state_dict'][0]['7.weight'].to(device)
            model.output.layer.bias.data = full_dict['state_dict'][0]['7.bias'].to(device)
        if param2load != None and param2load.lower() not in ['conv', 'fc']:
            raise ValueError('Unsupported param2load value: %s' % param2load)
    elif name.lower() in ['resnet',]:
        # Load in_block
        model.in_block.conv.layer.weight.data = full_dict['state_dict'][0]['0.weight']
        model.in_block.conv.layer.bias.data = full_dict['state_dict'][0]['0.bias']
        # Load block1
        model.block1.conv1.layer.weight.data = full_dict['state_dict'][0]['2.Ws.0.weight']
        model.block1.conv1.layer.bias.data = full_dict['state_dict'][0]['2.Ws.0.bias']
        model.block1.skip_conv.layer.weight.data = full_dict['state_dict'][0]['4.Ws.0.weight']
        model.block1.skip_conv.layer.bias.data = full_dict['state_dict'][0]['4.Ws.0.bias']
        model.block1.conv2.layer.weight.data = full_dict['state_dict'][0]['4.Ws.2.weight']
        model.block1.conv2.layer.bias.data = full_dict['state_dict'][0]['4.Ws.2.bias']
        # Load block2
        model.block2.conv1.layer.weight.data = full_dict['state_dict'][0]['6.Ws.0.weight']
        model.block2.conv1.layer.bias.data = full_dict['state_dict'][0]['6.Ws.0.bias']
        model.block2.skip_conv.layer.weight.data = full_dict['state_dict'][0]['8.Ws.0.weight']
        model.block2.skip_conv.layer.bias.data = full_dict['state_dict'][0]['8.Ws.0.bias']
        model.block2.conv2.layer.weight.data = full_dict['state_dict'][0]['8.Ws.2.weight']
        model.block2.conv2.layer.bias.data = full_dict['state_dict'][0]['8.Ws.2.bias']
        # Load block3
        model.block3.conv1.layer.weight.data = full_dict['state_dict'][0]['10.Ws.0.weight']
        model.block3.conv1.layer.bias.data = full_dict['state_dict'][0]['10.Ws.0.bias']
        model.block3.skip_conv.layer.weight.data = full_dict['state_dict'][0]['12.Ws.0.weight']
        model.block3.skip_conv.layer.bias.data = full_dict['state_dict'][0]['12.Ws.0.bias']
        model.block3.conv2.layer.weight.data = full_dict['state_dict'][0]['12.Ws.2.weight']
        model.block3.conv2.layer.bias.data = full_dict['state_dict'][0]['12.Ws.2.bias']
        # Load block4
        model.block4.conv1.layer.weight.data = full_dict['state_dict'][0]['14.Ws.0.weight']
        model.block4.conv1.layer.bias.data = full_dict['state_dict'][0]['14.Ws.0.bias']
        model.block4.skip_conv.layer.weight.data = full_dict['state_dict'][0]['16.Ws.0.weight']
        model.block4.skip_conv.layer.bias.data = full_dict['state_dict'][0]['16.Ws.0.bias']
        model.block4.conv2.layer.weight.data = full_dict['state_dict'][0]['16.Ws.2.weight']
        model.block4.conv2.layer.bias.data = full_dict['state_dict'][0]['16.Ws.2.bias']
        # Load out_block
        model.out_block.fc1.layer.weight.data = full_dict['state_dict'][0]['19.weight']
        model.out_block.fc1.layer.bias.data = full_dict['state_dict'][0]['19.bias']
        model.out_block.fc2.layer.weight.data = full_dict['state_dict'][0]['21.weight']
        model.out_block.fc2.layer.bias.data = full_dict['state_dict'][0]['21.bias']
    else:
        raise ValueError('Unsupported model type: %s' % name)

    return model

def dump_pth_model(model, name, pth_file, device, param2dump = None):
    '''
    >>> model: the pytorch model
    >>> name: the type of the model
    >>> pth_file: the pth file
    >>> device: the device of the model
    '''

    raise NotImplementedError
