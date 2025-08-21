""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

import PIL 

__all__ = ['show_sample']

def show_sample(sample, idx, thrh=0.6):
    """for coco dataset/dataloader
    """
    output_path = './output/transform'
    os.makedirs(output_path, exist_ok=True)

    image, target = sample[0],sample[1]
    if isinstance(image, torch.Tensor):
        image = (image * 255).to(torch.uint8)  # Ensure it's in uint8 format
        image_pil = F.to_pil_image(image)  # Convert to PIL image for drawing
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    elif isinstance(image, PIL.Image.Image):
        image_pil = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    draw = ImageDraw.Draw(image_pil)
    boxes = target["boxes"]
    labels = target["labels"]
    angles = target.get("angles", torch.zeros(boxes.shape[0]))  # Default angles to 0 if not provided
    scores = target.get("scores", torch.ones(boxes.shape[0]))  # Default scores to 1 if not provided

    for i, (box, angle, label, score) in enumerate(zip(boxes, angles, labels, scores)):
        if score < thrh:
            continue

        x1, y1, x2, y2 = box.detach().cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        center = (x1 + width / 2, y1 + height / 2)

        # Rotate bounding box points
        rot_mat = cv2.getRotationMatrix2D(center, np.degrees(angle.item()), 1.0)
        points = np.array([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1]
        ], dtype=np.float32)
        points = np.hstack((points, np.ones((4, 1), dtype=np.float32)))
        points = np.dot(rot_mat, points.T).T
        points = points.astype(np.int32)

        draw.polygon([tuple(point) for point in points], outline='yellow')
        draw.text((x1, y1), f"Label: {label.item()} ; Score: {round(score.item(), 2)}", fill='white')

    # Save the result
    image_pil.save(os.path.join(output_path, f'{idx}.png'))
    print(f'Saved image {idx}.png')

