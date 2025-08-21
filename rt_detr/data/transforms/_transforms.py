""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import  Any, Callable, Dict, List, Optional, Union, Tuple

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
#RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
#SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes) 
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs
    

@register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
        torch.Tensor,  # For angles
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def _transform(self, image: Any, boxes: BoundingBoxes, angles: torch.Tensor) -> Any:
        flipped_image = F.hflip(image)
        flipped_box = F.hflip(boxes)
        flipped_angle =  angles * (-1)
        return flipped_image,flipped_box,flipped_angle

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) < self.p:
            image = inputs[0][0]
            data_dict = inputs[0][1]
            boxes = data_dict.get('boxes')
            angles = data_dict.get('angles')
            if boxes is not None and angles is not None:
                # Check if the input is angles
                f_img,f_box,f_angle = self._transform(image, boxes, angles)
                new_data_dict = data_dict.copy()
                new_data_dict['boxes'] = f_box
                new_data_dict['angles'] = f_angle
                return (f_img,new_data_dict,inputs[0][2:])
        return inputs[0]




@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
        torch.tensor,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, angle: Optional[torch.Tensor]=None) -> Any:  
        if isinstance(inpt, BoundingBoxes):
            spatial_size = getattr(inpt, _boxes_keys[1])
            if self.fmt:
                in_fmt = inpt.format.value.lower()
                inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
                inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

            if self.normalize:
                inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]
        if isinstance(angle, torch.Tensor):
            if self.normalize:
                #angle = angle / (2*torch.pi) + 0.5
                angle = angle / torch.pi + 0.5

        return (inpt, angle) if angle is not None else inpt

    def __call__(self, *inputs: Any) -> Any:
        if isinstance(inputs, tuple) and inputs[0] is not None:
            inputs[0][1]['boxes'],inputs[0][1]['angles'] = self._transform(inputs[0][1]['boxes'],inputs[0][1]['angles'])
        return inputs[0]
        



@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
    


class CustomSanitizeBoundingBoxes(T.Transform):
    """Remove degenerate/invalid bounding boxes and their corresponding labels, angles, and areas."""

    def __init__(
        self,
        min_size: float = 1.0,
        labels_getter: Union[Callable[[Any], Optional[torch.Tensor]], str, None] = "default",
        angles_getter: Union[Callable[[Any], Optional[torch.Tensor]], str, None] = "default",
        areas_getter: Union[Callable[[Any], Optional[torch.Tensor]], str, None] = "default"
    ) -> None:
        super().__init__()

        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}.")
        self.min_size = min_size

        self.labels_getter = labels_getter
        self.angles_getter = angles_getter
        self.areas_getter = areas_getter

    def _parse_getter(self, getter):
        if getter == "default":
            return lambda inputs: inputs.get(getter) if isinstance(inputs, dict) else None
        elif callable(getter):
            return getter
        else:
            raise ValueError(f"Invalid getter: {getter}")

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]

        labels = self._parse_getter(self.labels_getter)(inputs)
        angles = self._parse_getter(self.angles_getter)(inputs)
        areas = self._parse_getter(self.areas_getter)(inputs)

        if labels is not None and not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"The labels in the input to forward() must be a tensor or None, got {type(labels)} instead."
            )

        flat_inputs, spec = self.tree_flatten(inputs)
        boxes = self.get_bounding_boxes(flat_inputs)

        if boxes is None or boxes.numel() == 0:
            return inputs  # No boxes to sanitize, return inputs as is

        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match."
            )
        if angles is not None and boxes.shape[0] != angles.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of angles (shape={angles.shape}) do not match."
            )
        if areas is not None and boxes.shape[0] != areas.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of areas (shape={areas.shape}) do not match."
            )

        boxes = self.convert_bounding_box_format(boxes, new_format="XYXY")
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)

        image_h, image_w = boxes.canvas_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)

        params = dict(valid=valid.as_subclass(torch.Tensor), labels=labels, angles=angles, areas=areas)
        flat_outputs = [
            self._transform(inpt, params)
            for inpt in flat_inputs
        ]

        return self.tree_unflatten(flat_outputs, spec)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = inpt is not None and inpt is params["labels"]
        is_angle = inpt is not None and inpt is params["angles"]
        is_area = inpt is not None and inpt is params["areas"]
        is_bounding_boxes_or_mask = isinstance(inpt, (BoundingBoxes, Mask))

        if not (is_label or is_bounding_boxes_or_mask or is_angle or is_area):
            return inpt

        output = inpt[params["valid"]]

        if is_label or is_angle or is_area:
            return output

        return self.wrap(output, like=inpt)

    def tree_flatten(self, inputs: Any) -> Tuple[List[Any], Any]:
        # Simplified implementation of tree flattening
        return [inputs], None

    def tree_unflatten(self, flat_inputs: List[Any], spec: Any) -> Any:
        # Simplified implementation of tree unflattening
        return flat_inputs[0] if len(flat_inputs) == 1 else flat_inputs

    def get_bounding_boxes(self, inputs: List[Any]) -> Optional[torch.Tensor]:
        # Simplified implementation of getting bounding boxes
        return inputs[0]["boxes"] if isinstance(inputs[0], dict) and "boxes" in inputs[0] else None

    def convert_bounding_box_format(self, boxes: torch.Tensor, new_format: str) -> torch.Tensor:
        # Simplified implementation of bounding box format conversion
        return boxes

    def wrap(self, output: Any, like: Any) -> Any:
        # Simplified implementation of wrapping output
        return output
     
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(CustomSanitizeBoundingBoxes)