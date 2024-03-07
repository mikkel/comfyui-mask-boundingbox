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

        len_zeros = len(non_zero_positions[0])
        min_x = int(torch.min(non_zero_positions[:, len_zeros - 1]))
        max_x = int(torch.max(non_zero_positions[:, len_zeros - 1]))
        min_y = int(torch.min(non_zero_positions[:, len_zeros - 2]))
        max_y = int(torch.max(non_zero_positions[:, len_zeros - 2]))
        
        if (len_zeros == 2):
            raw_bb = mask_bounding_box[int(min_y):int(max_y), int(min_x):int(max_x)]
        elif (len_zeros == 3):
            raw_bb = mask_bounding_box[:,int(min_y):int(max_y), int(min_x):int(max_x)] 
        elif (len_zeros == 4):
            raw_bb = mask_bounding_box[:,:,int(min_y):int(max_y), int(min_x):int(max_x)] 

        print (raw_bb == mask_bounding_box[...,int(min_y):int(max_y), int(min_x):int(max_x)])
                    
        print("here")
        print(min_x, max_x, min_y, max_y)
        print(len(non_zero_positions[0]))
        print(non_zero_positions[:, 2])
        for i in range(len_zeros):
            print(str(i), mask_bounding_box.shape[i])
            #array_to_text_file(non_zero_positions[:, i], "boudning_box_error_" + str(i) + ".txt")
      
        pad_x = max(0, (min_width - (max_x - min_x)) / 2)
        pad_y = max(0, (min_height - (max_y - min_y)) / 2)
        
        pad_left = max(pad_left, pad_x)
        pad_right = max(pad_right, pad_x)
        pad_top = max(pad_top, pad_y)
        pad_bottom = max(pad_bottom, pad_y)

        width = mask_bounding_box.shape[len_zeros - 1]
        height = mask_bounding_box.shape[len_zeros - 2]

        min_x = max(min_x - pad_left, 0)
        max_x = min(max_x + pad_right, width)
        min_y = max(min_y - pad_top, 0)
        max_y = min(max_y + pad_bottom, height)
        
        raw_img = image_mapped[:,int(min_y):int(max_y),int(min_x):int(max_x),:]

        return (int(min_x), int(max_x), int(min_y), int(max_y), int(max_x-min_x), int(max_y-min_y), raw_bb, raw_img)


NODE_CLASS_MAPPINGS = {
    "Mask Bounding Box": MaskBoundingBox,
}
