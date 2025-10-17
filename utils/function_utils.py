# 本文件包含模型初始化相关的工具函数，用于推理
from utils.tool_utils import source_import
import torch
import os
from collections import OrderedDict


def init_model(config, pretrain=True):
    """
    初始化模型并加载预训练权重（如果提供）
    """
    weights_path = config['training_opt']['pretrain_path']
    use_my_pth = config['training_opt']['use_my_pth']
    device = torch.device(config['device'])
    networks_defs = config['network']
    
    def_file = networks_defs['def_file'] 
    model_args = networks_defs['params']  
    networks = source_import(def_file).create_model(**model_args)

    if weights_path is not None and pretrain:
        if os.path.exists(weights_path) and not use_my_pth:
            if 'convnext' in weights_path or 'swin' in weights_path:
                weights_dict = torch.load(weights_path)['model']
            else:
                weights_dict = torch.load(weights_path)

            model_dict = networks.state_dict()
            matched_weights = {}
            for k, v in weights_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    matched_weights[k] = v

            missing, unexpected = networks.load_state_dict(matched_weights, strict=False)
            print(f"missing:{missing}\nunexpected:{unexpected}")
        
        elif os.path.exists(weights_path) and use_my_pth:
            checkpoint = torch.load(weights_path, map_location='cpu')
            weight_dict = checkpoint['model']
            new_state_dict = OrderedDict()
            for k, v in weight_dict.items():
                name = k[7:] if k.startswith('module.') else k
                if not (name.startswith('disease_classifier') or 
                        name.startswith('self_attention') or
                        name.startswith('cls_token')):
                    new_state_dict[name] = v
            print(networks.load_state_dict(new_state_dict, strict=False))
        else:
            raise FileNotFoundError('not found weights file: {}'.format(weights_path))

    return networks
