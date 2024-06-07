"""
Given a stroke, and a list of segmentation masks, we want to find the majority voting segmentation mask.

We define a stroke by a list of sampled 2D points. 
"""

import numpy as np


def majority_voting_seg_masks(seg_masks):
    """
    Args:
     stroke_points: list of 2D points
     seg_masks: list of segmentation masks, each of shape (H, W)
    """

    # We get majority mask for different LoDs (levels of detail) by merging different sizes
    # of segmentation masks from each point in the stroke

    majority_mask = np.zeros(seg_masks[0].shape, dtype=np.uint8)
    for seg_mask in seg_masks:
        majority_mask += seg_mask
    normalize_mask = majority_mask / len(seg_masks)
    threhold = 0.2
    majority_mask = (normalize_mask > threhold).astype(np.uint8)
    return majority_mask
