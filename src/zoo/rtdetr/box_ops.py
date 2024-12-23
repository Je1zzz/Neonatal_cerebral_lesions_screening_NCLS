"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
from typing import Tuple

def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h, theta = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h),theta]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1, theta = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0),theta]
    return torch.stack(b, dim=-1)

def angle_01_to_pi(x: Tensor) -> Tensor:
    cx,cy,w,h,theta = x.unbind(-1)
    theta = (theta-0.5)*2*torch.pi
    b = [cx,cy,w,h,theta]
    return torch.stack(b,dim=-1)

def angle_pi_to_01(x: Tensor) -> Tensor:
    cx,cy,w,h,theta = x.unbind(-1)
    theta = theta/(2*torch.pi) + 0.5
    b = [cx,cy,w,h,theta]
    return torch.stack(b,dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):

    boxes1_r = RotatedBoxes(boxes1)
    boxes2_r = RotatedBoxes(boxes2)
    iou = pairwise_iou_rotated(boxes1_r,boxes2_r)
    area1 = boxes1_r.area()
    area2 = boxes2_r.area()
    inter = (iou * (area1.unsqueeze(1) + area2.unsqueeze(0))) / (1 + iou)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return iou, union



def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:-1] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:-1] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)    #[N,M]

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:-1], boxes2[:, 2:-1])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

