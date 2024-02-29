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
                "threshold": ("FLOAT", {"default": 0}),
                "pad_left": ("FLOAT", {"default": 0}),
                "pad_top": ("FLOAT", {"default": 0}),
                "pad_right": ("FLOAT", {"default": 0}),
                "pad_bottom": ("FLOAT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",  "MASK", "IMAGE")
    RETURN_NAMES = ("X1","X2", "Y1","Y2", "width", "height", "bounded mask", "bounded image")
    FUNCTION = "compute_bounding_box"
    CATEGORY = "image/processing"
    
    def compute_bounding_box(self, mask_bounding_box, min_width, min_height, image_mapped, threshold, pad_left, pad_top, pad_right, pad_bottom):
        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold

        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold, as_tuple=False)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(image_mapped))

        if (len(non_zero_positions[0]) == 3):
            min_x = int(torch.min(non_zero_positions[:, 2]))
            max_x = int(torch.max(non_zero_positions[:, 2]))
            min_y = int(torch.min(non_zero_positions[:, 1]))
            max_y = int(torch.max(non_zero_positions[:, 1]))
            raw_bb = mask_bounding_box[:,int(min_y):int(max_y), int(min_x):int(max_x)] 
        else:
            min_x = int(torch.min(non_zero_positions[:, 1]))
            max_x = int(torch.max(non_zero_positions[:, 1]))
            min_y = int(torch.min(non_zero_positions[:, 0]))
            max_y = int(torch.max(non_zero_positions[:, 0]))
            raw_bb = mask_bounding_box[int(min_y):int(max_y), int(min_x):int(max_x)]

        cx = (max_x+min_x)//2
        cy = (max_y+min_y)//2

        while(max_x - min_x < min_width):
            if max_x < mask_bounding_box.shape[1] or min_x <= 0:
                max_x+=1
            else:
                min_x-=1

        while(max_y - min_y < min_height):
            if max_y < mask_bounding_box.shape[0] or min_y <= 0:
                max_y+=1
            else:
                min_y-=1

        min_x -= pad_left
        max_x += pad_right
        min_y -= pad_top
        max_y += pad_bottom
        
        raw_img = image_mapped[:,int(min_y):int(max_y),int(min_x):int(max_x),:]

        return (int(min_x), int(max_x), int(min_y), int(max_y), int(max_x-min_x), int(max_y-min_y), raw_bb, raw_img)


NODE_CLASS_MAPPINGS = {
    "Mask Bounding Box": MaskBoundingBox,
}
