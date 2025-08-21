"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .cifar_dataset import CIFAR10
from .coco_dataset import CocoDetection
from .coco_dataset import (
    CocoDetection, 
    neonatal_cranium_category2name, 
    neonatal_cranium_category2label,
    neonatal_cranium_label2category,
)
#from .coco_eval import CocoEvaluator
#from .coco_utils import get_coco_api_from_dataset
#from .voc_detection import VOCDetection
#from .voc_eval import VOCEvaluator
#from .rotated_coco_eval import RotatedCocoEvaluator
