import torch 
import numpy as np
from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh,angle_01_to_pi,angle_pi_to_01

def get_contrastive_denoising_training_group(targets,   
                                             num_classes,
                                             num_queries,   
                                             class_embed,   
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]   
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts)  
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num 
    num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 5], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i] 
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']  
            input_query_bbox[i, :num_gt] = targets[i]['boxes']      
            pad_gt_mask[i, :num_gt] = 1                             
    input_query_class = input_query_class.tile([1, 2 * num_group]) 
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)  
    negative_gt_mask[:, max_gt_num:] = 1   
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1]) 
    positive_gt_mask = 1 - negative_gt_mask
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group) 

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:4] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox[...,:4], 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox[...,:4])
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox[...,:4] += (rand_sign * rand_part * diff)
        known_bbox[...,:4] = torch.clip(known_bbox[...,:4], min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)

        known_angle = angle_01_to_pi(input_query_bbox)
        angle_noise_scale_positive = np.radians(20)
        angle_noise_scale_negative = np.radians(40)

        rand_sign = torch.randint_like(input_query_bbox[..., -1:], 0, 2) * 2.0 - 1.0

        rand_part_positive = torch.rand_like(input_query_bbox[..., -1:]) * angle_noise_scale_positive
        rand_part_negative = (torch.rand_like(input_query_bbox[..., -1:]) * (angle_noise_scale_negative - angle_noise_scale_positive)) + angle_noise_scale_positive

        angle_noise = (rand_sign * (rand_part_positive * (1 - negative_gt_mask) + rand_part_negative * negative_gt_mask))
        known_angle[..., -1] += angle_noise[...,-1]
        known_angle[..., -1] = torch.clip(known_angle[..., -1], min=-torch.pi, max=torch.pi)
        input_query_bbox = angle_pi_to_01(known_angle)

    input_query_bbox_unact = torch.cat([inverse_sigmoid(input_query_bbox)], dim=-1)
    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }
    
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta