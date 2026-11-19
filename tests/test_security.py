import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location("bounding", Path(__file__).parents[1]/"mask_boundingbox.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_oversized_minimum_cannot_spin_forever():
    mask=torch.ones(8,8)
    result=module.MaskBoundingBox().compute_bounding_box(mask,512,512,torch.ones(1,8,8,3),0.5)
    assert result[4:6] == (8,8)
    assert result[-1].shape == (1,8,8,3)

def test_normal_bounding_and_empty_masks():
    node=module.MaskBoundingBox()
    mask=torch.zeros(16,16);mask[2:6,3:7]=1
    assert node.compute_bounding_box(mask,4,4,torch.ones(1,16,16,3),0.5)[4:6]==(5,5)
    assert node.compute_bounding_box(torch.zeros(16,16),4,4,torch.ones(1,16,16,3),0.5)[4:6]==(0,0)


def test_comfyui_mask_batch_crops_spatial_dimensions():
    mask = torch.zeros(2, 8, 12)
    mask[0, 1:3, 2:5] = 1
    mask[1, 5:8, 9:12] = 1
    image = torch.arange(2 * 8 * 12 * 3).reshape(2, 8, 12, 3)

    result = module.MaskBoundingBox().compute_bounding_box(mask, 1, 1, image, 0.5)

    assert result[:6] == (2, 12, 1, 8, 10, 7)
    assert torch.equal(result[6], mask[:, 1:8, 2:12])
    assert torch.equal(result[7], image[:, 1:8, 2:12, :])


def test_comfyui_single_pixel_mask_is_retained_at_the_edge():
    mask = torch.zeros(1, 8, 12)
    mask[0, -1, -1] = 1
    image = torch.ones(1, 8, 12, 3)

    result = module.MaskBoundingBox().compute_bounding_box(mask, 0, 0, image, 0.5)

    assert result[:6] == (11, 12, 7, 8, 1, 1)
    assert torch.equal(result[6], torch.ones(1, 1, 1))
    assert result[7].shape == (1, 1, 1, 3)
