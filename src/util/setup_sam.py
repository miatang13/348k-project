import os
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

checkpoint_dir = "/home/miatang/projects/stroke-label/data/ckpts"
"""
Segmentation 
"""


def setup_sam_mask_generator():
    sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    sam_checkpoint = os.path.abspath(sam_checkpoint)
    print(f"Loading SAM model from {sam_checkpoint}")
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def setup_mask_annotator():
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    return mask_annotator
