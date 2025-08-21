import json
import time
from shapely.geometry import Polygon
import numpy as np
import copy
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import torch
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class RotatedCOCO(COCO):
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile # list
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']    # xywhd
                x1,y1,w,h,degree = bb
                cx,cy = x1+w/2, y1+h/2   
                
                theta = np.radians(degree)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]
                ])
                points = np.array([
                    [-w / 2, -h / 2],
                    [w / 2, -h / 2],
                    [w / 2, h / 2],
                    [-w / 2, h / 2]
                ])
                rotated_points = np.dot(points, rotation_matrix.T) + np.array([cx, cy])
                polygon = Polygon(rotated_points)
                ann['area'] = polygon.area

                #ann['bbox'][-1] = 0. 
                # corners = box2corners_th(torch.tensor(bb).unsqueeze(0).unsqueeze(1))   
                # corners = corners.squeeze(0).squeeze(0) 
                # x1, y1 = corners[0]
                # x2, y2 = corners[1]
                # x3, y3 = corners[2]
                # x4, y4 = corners[3]
                # ann['area'] = Polygon([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]).convex_hull.area
                      
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
    

def box2corners_th(box:torch.Tensor) -> torch.Tensor:
    """
    Convert box coordinates to corners
    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha
    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B, N = box.size()[:2]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5]  # (B, N, 1)

    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device)  # (1, 1, 4)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h  # (B, N, 4)

    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view(-1, 4, 2), rot_T.view(-1, 2, 2))
    rotated = rotated.view(B, N, 4, 2)  # (B*N, 4, 2) -> (B, N, 4, 2)

    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

