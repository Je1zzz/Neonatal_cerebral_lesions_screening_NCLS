"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
import sys

from pycocotools.cocoeval import COCOeval, maskUtils
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.file_io import PathManager

from .oriented_iou_loss import cal_giou,cal_iou
from .rotate_boxes_iou import GIoU_Rotated_Rectangle 



from ...misc import dist_utils
from ...core import register

__all__ = ['CocoEvaluator',]


def convert_angle_01_pi(box):
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


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        for ann in coco_gt.dataset['annotations']:
            if 'bbox' in ann and 'angle' in ann:
                ann['bbox'] = ann['bbox'] + [ann['angle']]
                ann['bbox'] = convert_radians_to_degrees(ann['bbox'],gt_tag=1)
                #ann['bbox'] = convert_angle_01_pi(ann['bbox'])
                ann['bbox'] = convert_to_cxcywh(ann['bbox'])
                
        coco_gt.createIndex()
        
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = RotatedCOCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
    
    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = RotatedCOCOeval(self.coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
    
    
    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

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

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax, theta = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin, theta), dim=1) # 左上角xy,宽高wh


def convert_to_cxcywh(boxes):
    if isinstance(boxes, list):
        xmin, ymin, w, h, theta = boxes
        return [xmin+w/2, ymin+h/2, w, h, theta]
    else:
        xmin, ymin, xmax, ymax, theta = boxes.unbind(1)
        return torch.stack(((xmax-xmin)/2, (ymax-ymin)/2, xmax - xmin, ymax - ymin, theta), dim=1) 




def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# import io
# from contextlib import redirect_stdout
# def evaluate(imgs):
#     with redirect_stdout(io.StringIO()):
#         imgs.evaluate()
#     return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def evaluate(self,device=None):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId, device)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [ 
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################


@register()
class RotatedCOCOeval(COCOeval):
    @staticmethod
    def is_rotated(box_list):
        if type(box_list) == np.ndarray:
            return box_list.shape[1] == 5
        elif type(box_list) == list:
            if box_list == []:  # cannot decide the box_dim
                return False
            return np.all(
                np.array(
                    [
                        (len(obj) == 5) and ((type(obj) == list) or (type(obj) == np.ndarray))
                        for obj in box_list
                    ]
                )
            )
        return False

    @staticmethod
    def boxlist_to_tensor(boxlist, output_box_dim):
        if type(boxlist) == np.ndarray:
            box_tensor = torch.from_numpy(boxlist)
        elif type(boxlist) == list:
            if boxlist == []:
                return torch.zeros((0, output_box_dim), dtype=torch.float32)
            else:
                box_tensor = torch.FloatTensor(boxlist)
        else:
            raise Exception("Unrecognized boxlist type")

        input_box_dim = box_tensor.shape[-1]
        if input_box_dim != output_box_dim:
            if input_box_dim == 4 and output_box_dim == 5:
                box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)
            else:
                raise Exception(
                    "Unable to convert from {}-dim box to {}-dim box".format(
                        input_box_dim, output_box_dim
                    )
                )
        return box_tensor

    def compute_iou_dt_gt(self, dt, gt, is_crowd, device):
        if self.is_rotated(dt) or self.is_rotated(gt):
            # TODO: take is_crowd into consideration
            assert all(c == 0 for c in is_crowd)
            dt1 = RotatedBoxes(self.boxlist_to_tensor(dt, output_box_dim=5))
            gt1 = RotatedBoxes(self.boxlist_to_tensor(gt, output_box_dim=5))
            iou1 =  pairwise_iou_rotated(dt1,gt1)
        
            # dt = self.boxlist_to_tensor(dt, output_box_dim=5)
            # gt = self.boxlist_to_tensor(gt, output_box_dim=5)
            # return cal_iou(dt.unsqueeze(0).to(device),gt.unsqueeze(0).to(device))

            dt = self.boxlist_to_tensor(dt, output_box_dim=5)
            gt = self.boxlist_to_tensor(gt, output_box_dim=5)
            _, iou2 = GIoU_Rotated_Rectangle(dt,gt)
            return iou1

        else:
            # This is the same as the classical COCO evaluation
            return maskUtils.iou(dt, gt, is_crowd)

    def computeIoU(self, imgId, catId, device):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Note: this function is copied from cocoeval.py in cocoapi
        # and the major difference is here.
        #ious,_,_,_ = self.compute_iou_dt_gt(d, g, iscrowd, device)
        ious = self.compute_iou_dt_gt(d, g, iscrowd, device)    # 如果g为[]，会返回ious为[]; 如果len(d)为N，len(g)为M，会返回(N,M)的ndarray
        if torch.is_tensor(ious):
            if 0 in ious.shape:
                return []
            #ious = ious[0].cpu().numpy()
            ious = ious.cpu().numpy()
        return ious