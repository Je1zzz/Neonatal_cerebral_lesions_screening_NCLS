import importlib
import torch 
import numpy as np
from collections import Counter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update(config, args):
    args_dict = vars(args)  
    for key, value in args_dict.items():
        if key == 'cfg':
            continue
        else:
            if value is not None:
                config[key] = value
        print(key,value)
    return config

def update_training_opt_from_wandb(config, wandb_config):
    updated_config = dict(config)
    for key in wandb_config.keys():
        if key in updated_config['training_opt']:
            updated_config['training_opt'][key] = wandb_config[key]
        else:
            raise KeyError(f"The key '{key}' from wandb.config does not exist in the 'training_opt' section of the original config.")
    return updated_config


def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def check_tensor_device(tensor_list):
    for i,t in enumerate(tensor_list):
        if isinstance(t,torch.Tensor):
            if t.is_cuda:
                tensor_list[i] = t.cpu()
    return tensor_list

def freeze_except_bag_classifier(model):
    print(model.named_parameters())
    for name, param in model.named_parameters():
        if 'disease_classifier' not in name and 'self_attention' not in name and 'cls_token' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def get_grad_norm(model):

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_pos_weights(data_loader, num_classes):
    label_counts = np.zeros(num_classes)
    
    for step, data in enumerate(data_loader):
        imgs, patient_labels, img_labels, plane_labels, bag_lengths, _, _, idx = data
        label_counts += img_labels.sum(axis=0).numpy()
    
    total_samples = label_counts.sum()
    pos_weights = []

    for pos_count in label_counts:
        neg_count = total_samples - pos_count
        pos_weight = neg_count / (pos_count + 1e-5)
        pos_weights.append(pos_weight)
    
    return pos_weights