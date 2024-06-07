import numpy as np
from scipy import ndimage
import cv2


def dilate_strokes(binary_img, iterations=15):
    """
    sketch_img: np.array (W, H), should be white strokes on black background
    """
    dilated = ndimage.binary_dilation(binary_img, iterations=iterations)
    return dilated


def color_stroke_from_ann_img(annotated_image, sketch_image):
    if not isinstance(annotated_image, np.ndarray):
        annotated_image = np.array(annotated_image)
    dilated_sketch_img = (
        dilate_strokes(sketch_image, iterations=3).astype(np.uint8)
    ) * 255
    annotated_strokes = np.ones_like(annotated_image) * 255
    annotated_strokes[dilated_sketch_img == 255] = annotated_image[
        dilated_sketch_img == 255
    ]

    return annotated_strokes


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sketch_img = cv2.imread(
        "/home/miatang/projects/stroke-label/experiments/2D_circle_sqr_occlude/segment/0/auto_sam/indiv_masks/mask_1.jpg", cv2.IMREAD_GRAYSCALE)
    dilated_img = dilate_strokes(sketch_img)
    cv2.imwrite("colored_strokes.png", dilated_img)
