# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import numpy as np
from pycocotools.coco import COCO
from .RotatedCOCO import RotatedCOCO
from .coco_eval import CocoEvaluator, evaluate, convert_radians_to_degrees
import torch

from ...core import register

@register()
class RotatedCocoEvaluator(CocoEvaluator):
    """
    For rotated object detection evaluation
    """
    def update(self, predictions, device):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)   # (xmin,ymin,w,h,theta)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = RotatedCOCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval,device = device)

            self.eval_imgs[iou_type].append(eval_imgs)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_cxcywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            boxes = convert_angle_01_pi(boxes)
            boxes = convert_radians_to_degrees(boxes)

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results
    

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax,t = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin,t), dim=1) # 左上角xy,宽高wh

def convert_to_cxcywh(boxes):
    xmin, ymin, xmax, ymax, theta = boxes.unbind(1)
    return torch.stack(((xmax+xmin)/2, (ymax+ymin)/2, xmax - xmin, ymax - ymin, theta), dim=1) 

def convert_angle_01_pi(box):
    if all(isinstance(i, list) for i in box):
        for i,b in enumerate(box):
            x, y, w, h, angle_rad = b
            angle_rad = (angle_rad - 0.5)*  np.pi
            box[i] = [x,y,w,h,angle_rad]
        return box
    else:
        x, y, w, h, angle_rad = box
        angle_rad = (angle_rad - 0.5)*  np.pi
        return [x,y,w,h,angle_rad]

def convert_radians_to_degrees(boxlist,gt_tag=0):
    updated_boxlist = []
    if all(isinstance(i, list) for i in boxlist):   # 多维列表
        for box in boxlist:
            x, y, w, h, angle_rad = box
            angle_deg = np.degrees(angle_rad)
            updated_boxlist.append([x, y, w, h, angle_deg])
    elif gt_tag:
        x, y, w, h, angle_rad = boxlist
        angle_deg = np.degrees(angle_rad)
        updated_boxlist= [x, y, w, h, angle_deg]
    return updated_boxlist