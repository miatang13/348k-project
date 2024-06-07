import os
import supervision as sv
import numpy as np
import json
import subprocess
import glob

# own modules
from util.img import save_image_to_path, load_sketch, load_sqr_image
from util.stroke import color_stroke_from_ann_img

cur_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = "/home/miatang/projects/stroke-label/data/ckpts"
GEN_NEW_MASK = True

"""
VISUALIZATION
"""


def find_sam_seg_gaps(sam_results):
    masks = [result["segmentation"] for result in sam_results]
    all_mask = np.zeros_like(masks[0])
    for mask in masks:
        all_mask += mask
    all_mask = all_mask.astype("uint8") * 255
    return all_mask


"""
SEGMENTATION
"""

grounded_dino_bash = "/home/miatang/projects/stroke-label/scripts/run_grounded_dino_sam.sh"
auto_ram_bash = "/home/miatang/projects/stroke-label/scripts/run_auto_label_ram.sh"

USE_AUTO = True


def segment_image_and_annotate_strokes(
    gen_image_path,
    image_to_segment,
    sketch_image,
    output_dir,
    text_prompt=None,
):
    '''
    Run script to do segmentaion
    '''
    clean_annotation_image_path = os.path.join(
        output_dir, "gen_clean_annotation")
    input_image_path = gen_image_path
    if USE_AUTO:
        seg_output_dir = os.path.join(output_dir, "auto_sam")
        img_seg_path = os.path.join(seg_output_dir, "auto_sam_output.jpg")
    else:
        seg_output_dir = os.path.join(output_dir, "grounded_sam")
        img_seg_path = os.path.join(seg_output_dir, "grounded_sam_output.jpg")
    indiv_masks_dir = os.path.join(seg_output_dir, "indiv_masks")
    mask_output_path = os.path.join(seg_output_dir, "mask.jpg")
    num_exist_masks = len(glob.glob(os.path.join(indiv_masks_dir, "*.jpg")))
    if GEN_NEW_MASK or not os.path.exists(mask_output_path) or num_exist_masks == 0:
        if USE_AUTO:
            cmd = f"bash {auto_ram_bash} {input_image_path} {seg_output_dir}"
        else:
            cmd = f"bash {grounded_dino_bash} {input_image_path} {seg_output_dir} '{text_prompt}'"
        subprocess.run(cmd, shell=True)
        if not os.path.exists(mask_output_path):
            raise ValueError(
                f"Grounded segmentation failed. \n Can't find {mask_output_path}")

    '''
    Get sam results from the masks
    '''
    clean_annotation_image = load_sqr_image(mask_output_path)
    clean_annotation_image_path = save_image_to_path(
        clean_annotation_image, clean_annotation_image_path)

    '''
    Color strokes 
    '''
    seg_stroke_image = color_stroke_from_ann_img(
        sketch_image=sketch_image,
        annotated_image=clean_annotation_image,
    )
    sketch_seg_path = save_image_to_path(
        seg_stroke_image,
        output_path=os.path.join(output_dir, "seg_stroke"),
    )
    paths = {
        "img_seg_path": img_seg_path,
        "sketch_seg_path": sketch_seg_path,
        "clean_ann_path": clean_annotation_image_path
    }

    return paths


def segment_and_annotate_image(
    input_img, mask_generator, mask_annotator, seg_method="naive"
):
    if not isinstance(input_img, np.ndarray):
        input_img = np.array(input_img)
    if seg_method == "naive":
        sam_result = naive_segmentation(mask_generator, input_img)
    else:
        raise NotImplementedError(
            f"Segmentation method {seg_method} not implemented")

    detections = sv.Detections.from_sam(sam_result)
    annotated_image = mask_annotator.annotate(input_img, detections)
    return sam_result, annotated_image, detections


def naive_segmentation(mask_generator, input_img):
    """
    SamAutomaticMaskGenerator returns a list of masks, where each mask is a dict
    containing various information about the mask:

    segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
    area - [int] - the area of the mask in pixels
    bbox - [List[int]] - the boundary box of the mask in xywh format
    predicted_iou - [float] - the model's own prediction for the quality of the mask
    point_coords - [List[List[float]]] - the sampled input point that generated this mask
    stability_score - [float] - an additional measure of mask quality
    crop_box - List[int] - the crop of the image used to generate this mask in xywh format
    """
    if not isinstance(input_img, np.ndarray):
        input_img = np.array(input_img)
    sam_result = mask_generator.generate(input_img)
    return sam_result


"""
SET UP
"""


def main():
    debug_config = "/home/miatang/projects/stroke-label/configs/test_segment.json"
    config = json.load(open(debug_config, "r"))
    sketch = load_sketch(config["sketch_path"])
    gen_img = load_sqr_image(config["gen_paths"][0])
    seg_img = load_sqr_image(config["clean_gen_ann_paths"][0])
    # mask_generator = setup_sam_mask_generator()
    # mask = find_sam_seg_gaps(naive_segmentation(mask_generator, gen_img))
    output_img = color_stroke_from_ann_img(sketch, seg_img)
    output_dir = "/home/miatang/projects/stroke-label/experiments/tmp"
    save_image_to_path(sketch, f"{output_dir}/sketch.png")
    save_image_to_path(output_img, f"{output_dir}/test_dilate_rgb.png")


if __name__ == "__main__":
    main()
