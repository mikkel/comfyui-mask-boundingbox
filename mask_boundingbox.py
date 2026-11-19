import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class MaskBoundingBox:
    def __init__(self, device="cpu"):
        self.device = device
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_bounding_box": ("MASK",),
                "min_width": ("INT", {"default": 512}),
                "min_height": ("INT", {"default": 512}),
                "image_mapped": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",  "MASK", "IMAGE")
    RETURN_NAMES = ("X1","X2", "Y1","Y2", "width", "height", "bounded mask", "bounded image")
    FUNCTION = "compute_bounding_box"
    CATEGORY = "image/processing"
    
    def compute_bounding_box(self, mask_bounding_box, min_width, min_height, image_mapped, threshold):
        mask_above_threshold = mask_bounding_box > threshold
        # ComfyUI supplies [batch, height, width]. The node returns one box, so
        # include selected pixels from every mask in the batch.
        end_offset = 0
        if mask_above_threshold.ndim == 3:
            mask_above_threshold = mask_above_threshold.any(dim=0)
            # Include the last selected pixel in the slice. Keep the original
            # endpoint convention for callers passing a two-dimensional mask.
            end_offset = 1

        height, width = mask_above_threshold.shape
        min_width = max(0, min(int(min_width), width))
        min_height = max(0, min(int(min_height), height))

        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(image_mapped))

        min_x = int(torch.min(non_zero_positions[:, 1]))
        max_x = int(torch.max(non_zero_positions[:, 1])) + end_offset
        min_y = int(torch.min(non_zero_positions[:, 0]))
        max_y = int(torch.max(non_zero_positions[:, 0])) + end_offset

        while(max_x - min_x < min_width):
            if max_x < width:
                max_x+=1
            if min_x > 0:
                min_x-=1

        while(max_y - min_y < min_height):
            if max_y < height:
                max_y+=1
            if min_y > 0:
                min_y-=1

        # Extract raw bounded mask
        raw_bb = mask_bounding_box[..., min_y:max_y, min_x:max_x]
        raw_img = image_mapped[:,int(min_y):int(max_y),int(min_x):int(max_x),:]

        return (int(min_x), int(max_x), int(min_y), int(max_y), int(max_x-min_x), int(max_y-min_y), raw_bb, raw_img)


NODE_CLASS_MAPPINGS = {
    "Mask Bounding Box": MaskBoundingBox,
}

