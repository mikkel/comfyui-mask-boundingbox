import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def center_pad_tensor(src_tensor, target_height, target_width):
    '''Function to center pad a tensor (mask or image) to target dimensions.'''
    is_image = len(src_tensor.shape) == 4  # Checking if the tensor is an image based on its shape

    if is_image:
        batch, src_height, src_width, channels = src_tensor.shape
        padding_left = (target_width - src_width) // 2
        padding_right = target_width - padding_left - src_width
        padding_top = (target_height - src_height) // 2
        padding_bottom = target_height - padding_top - src_height
        padded_tensor = torch.nn.functional.pad(src_tensor, (0, 0, padding_left, padding_right, padding_top, padding_bottom))
    else:
        src_height, src_width = src_tensor.shape
        padding_left = (target_width - src_width) // 2
        padding_right = target_width - padding_left - src_width
        padding_top = (target_height - src_height) // 2
        padding_bottom = target_height - padding_top - src_height
        padded_tensor = torch.nn.functional.pad(src_tensor, (padding_left, padding_right, padding_top, padding_bottom))

    return padded_tensor


def extract_bounding_box_with_aspect_ratio(mask_or_image, min_y, max_y, min_x, max_x, target_aspect_ratio):
    '''Function to extract bounding box with a specific aspect ratio and pad with zeros if out-of-bounds for both masks and images.'''
    bb_width = max_x - min_x + 1
    bb_height = max_y - min_y + 1
    bb_aspect_ratio = bb_width / bb_height

    is_image = len(mask_or_image.shape) == 4  # Checking if the input is an image based on the shape

    if bb_aspect_ratio > target_aspect_ratio:
        # Adjust height based on width
        new_height = int(bb_width / target_aspect_ratio)
        height_diff = new_height - bb_height
        min_y -= height_diff // 2
        max_y += height_diff - (height_diff // 2)
    else:
        # Adjust width based on height
        new_width = int(bb_height * target_aspect_ratio)
        width_diff = new_width - bb_width
        min_x -= width_diff // 2
        max_x += width_diff - (width_diff // 2)

    # Extract with padding for out-of-bounds areas
    if is_image:
        extracted = mask_or_image[:, max(0, min_y):min(mask_or_image.shape[1], max_y+1), max(0, min_x):min(mask_or_image.shape[2], max_x+1), :]
    else:
        extracted = mask_or_image[max(0, min_y):min(mask_or_image.shape[0], max_y+1), max(0, min_x):min(mask_or_image.shape[1], max_x+1)]

    target_shape = (max_y-min_y+1, max_x-min_x+1, 3) if is_image else (max_y-min_y+1, max_x-min_x+1)
    if min_x < 0:
        max_x -= min_x
        min_x = 0
    if min_y < 0:
        max_y -= max_y
        min_y = 0
    #padded = extracted
    padded = center_pad_tensor(extracted, *target_shape[:2])

    return min_x, max_x, min_y, max_y, padded


class MaskBoundingBox:
    def __init__(self, device="cpu"):
        self.device = device
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_bounding_box": ("MASK",),
                "image_mapped": ("IMAGE",),
                "output_width": ("INT", {"default": 512}),
                "output_height": ("INT", {"default": 512}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("X1","X2", "Y1","Y2", "width", "height", "resized mask", "raw mask", "resized image", "raw image")
    FUNCTION = "compute_bounding_box"
    CATEGORY = "image/processing"
    
    def compute_bounding_box(self, mask_bounding_box, image_mapped, output_width, output_height, threshold):
        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold

        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, torch.zeros((output_width, output_height)), mask_bounding_box)

        min_x = torch.min(non_zero_positions[:, 1])
        max_x = torch.max(non_zero_positions[:, 1])
        min_y = torch.min(non_zero_positions[:, 0])
        max_y = torch.max(non_zero_positions[:, 0])

        # Calculate aspect ratio of bounding box and output size
        bb_width = max_x - min_x + 1
        bb_height = max_y - min_y + 1
        bb_aspect_ratio = bb_width / bb_height
        output_aspect_ratio = output_width / output_height
        
        # Determine which dimension (width or height) should match the output size
        if bb_aspect_ratio > output_aspect_ratio:
            # Width is the dominant dimension
            new_width = output_width
            new_height = int(output_width / bb_aspect_ratio)
        else:
            # Height is the dominant dimension
            new_height = output_height
            new_width = int(output_height * bb_aspect_ratio)

        # Extract raw bounded mask
        ax1,ax2,ay1,ay2, raw_bounded_mask = extract_bounding_box_with_aspect_ratio(mask_bounding_box, min_y, max_y, min_x, max_x, output_aspect_ratio)
        
        # Resize the bounded mask
        resized_bounded_mask = torch.nn.functional.interpolate(
            raw_bounded_mask.unsqueeze(0).unsqueeze(0),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)
        bx1, bx2, by1, by2, raw_image = extract_bounding_box_with_aspect_ratio(image_mapped, min_y, max_y, min_x, max_x, output_aspect_ratio)
        resized_image = torch.nn.functional.interpolate(
            raw_image.permute(0, 3, 1, 2),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)


        return (int(ax1), int(ax2), int(ay1), int(ay2), int(max_x-min_x), int(max_y-min_y), resized_bounded_mask, raw_bounded_mask, resized_image, raw_image)


NODE_CLASS_MAPPINGS = {
    "Mask Bounding Box": MaskBoundingBox,
}

