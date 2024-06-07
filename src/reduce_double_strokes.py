import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import supervision as sv
from util.img import (
    save_image_to_path,
    load_sketch,
    load_sqr_image,
    parse_masks_from_colorful_annotations,
    save_masks_to_path,
    load_mask_np_bool_array
)
from util.timing import get_timestamp
from util.path import get_auto_sam_masks_paths, get_sample_subdirs
from util.setup_sam import setup_mask_annotator
from util.stroke import color_stroke_from_ann_img, dilate_strokes
from util.masks import merge_similar_masks
from util.visualize import masks_to_sam_clean_annotation


def check_in_bg_mask(cur_mask, bg_mask_rgba):
    # Extract the alpha channel
    if not isinstance(bg_mask_rgba, np.ndarray):
        bg_mask_rgba = np.array(bg_mask_rgba)
    alpha_channel = bg_mask_rgba[:, :, 3]

    # Create the mask where alpha channel is 0
    mask = (alpha_channel == 0).astype(np.uint8)

    # Check if the current mask is in the background mask
    num_same = np.where(cur_mask == mask)[0].shape[0]
    total_pixels = cur_mask.shape[0] * cur_mask.shape[1]
    if num_same / total_pixels >= 0.9:
        return True
    else:
        return False


def plot_reduce_process(
    sketch_image,
    seg_mask,
    dilated_mask,
    stroke_pixel_in_mask,
    i,
    dispose_mask,
    output_dir,
    num_stroke_pixels,
    axes,
    fig,
):
    axes[0].imshow(sketch_image, cmap="gray")
    axes[0].set_title("Sketch")
    axes[1].imshow(seg_mask, cmap="gray")
    axes[1].set_title(f"Original mask {i} ")
    axes[2].imshow(dilated_mask, cmap="gray")
    axes[2].set_title(f"Dilated mask")
    axes[3].imshow(stroke_pixel_in_mask, cmap="gray")
    axes[3].set_title(f"Seg {i} with {num_stroke_pixels} stroke pixels")
    show_text = "Dispose" if dispose_mask else "Keep"
    fig.suptitle(f"Segment {i}: {show_text}")
    fig.savefig(f"{output_dir}/seg_{i}.png")


def reduce_double_stroke(
    sketch_image,
    seg_masks,
    output_dir,
    bg_mask_rgba=None
):
    enhanced_sam_path = os.path.join(output_dir, "enhanced_sam_results.png")
    enhanced_strokes_path = os.path.join(
        output_dir, "enhanced_annotated_strokes.png")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Has {len(seg_masks)} segmentation masks")
    if not isinstance(sketch_image, np.ndarray):
        sketch_image = np.array(sketch_image)
    in_same_seg_mask = np.zeros(
        (len(seg_masks), *sketch_image.shape), dtype=np.uint8)
    # sketch_image is white strokes on black background
    total_pixels = np.where(sketch_image)[0].shape[0]
    print(f"Total pixels: {total_pixels}")
    pixel_masks = []
    dilated_masks = []
    masks_to_keep = []

    mask_subdir = f"{output_dir}/kept_masks"
    if os.path.exists(mask_subdir):
        files = os.listdir(mask_subdir)
        for file in files:
            os.remove(f"{mask_subdir}/{file}")
    if not os.path.exists(mask_subdir):
        os.makedirs(mask_subdir, exist_ok=True)

    seg_masks = merge_similar_masks(seg_masks, no_save=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))

    """
    Need to remove dupe seg masks
    """

    for i, seg_mask in enumerate(seg_masks):
        dilated_mask = dilate_strokes(seg_mask, iterations=8)
        dilated_masks.append(dilated_mask)
        stroke_pixel_in_mask = sketch_image * dilated_mask
        num_stroke_pixels = np.where(stroke_pixel_in_mask)[0].shape[0]
        print(f"Segment {i} has {num_stroke_pixels} stroke pixels")
        in_same_seg_mask[i] = stroke_pixel_in_mask
        pixel_masks.append(stroke_pixel_in_mask)

        no_strokes = num_stroke_pixels == 0
        if bg_mask_rgba is None:
            in_bg_mask = False
        else:
            in_bg_mask = check_in_bg_mask(seg_mask, bg_mask_rgba)
        dispose_mask = no_strokes or in_bg_mask
        if not dispose_mask:
            masks_to_keep.append(dilated_mask)

        plot_reduce_process(
            sketch_image,
            seg_mask,
            dilated_mask,
            stroke_pixel_in_mask,
            i,
            dispose_mask,
            output_dir,
            num_stroke_pixels,
            axes,
            fig,
        )

    if len(masks_to_keep) == 0:
        masks_to_keep = dilated_masks

    # Save a plot with all the kept masks
    fig, axes = plt.subplots(1, len(masks_to_keep) + 1, figsize=(20, 7))

    axes[0].imshow(sketch_image, cmap="gray")
    axes[0].set_title("Sketch")
    if len(masks_to_keep) == 0:
        axes[0].set_title("No masks to keep")

    for i, mask in enumerate(masks_to_keep):
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Kept mask {i}")
        save_image_to_path(mask, f"{mask_subdir}/{i}.png")
    kept_masks_paths = [
        f"{mask_subdir}/{i}.png" for i in range(len(masks_to_keep))]
    fig.savefig(f"{output_dir}/kept_masks.png")

    # With the kept segments, we color them
    mask_annotator = setup_mask_annotator()
    annotated = masks_to_sam_clean_annotation(masks_to_keep, mask_annotator)
    save_image_to_path(annotated, enhanced_sam_path)
    annotated_strokes = color_stroke_from_ann_img(
        annotated_image=annotated, sketch_image=sketch_image
    )
    save_image_to_path(annotated_strokes, enhanced_strokes_path)

    paths = {
        "kept_masks_paths": kept_masks_paths,
        "enhanced_sam_path": enhanced_sam_path,
        "enhanced_strokes_path": enhanced_strokes_path
    }
    return paths


def run_on_config(args):
    config = json.load(open(args.config, "r"))
    config_name = os.path.basename(args.config).split(".")[0]
    new_data = []

    for item in config:
        sketch = load_sketch(item["sketch_path"])
        new_item = item.copy()
        enhanced_strokes_paths = []
        enhanced_sam_paths = []

        for i, clean_gen_path in enumerate(item["clean_gen_ann_paths"]):
            clean_gen = load_sqr_image(clean_gen_path)
            decomposed_masks = parse_masks_from_colorful_annotations(clean_gen)
            enhanced_strokes_path, sam_output_path = reduce_double_stroke(
                sketch,
                decomposed_masks,
                f"{os.path.dirname(clean_gen_path)}/dilated",
            )
            enhanced_strokes_paths.append(enhanced_strokes_path)
            enhanced_sam_paths.append(sam_output_path)

        new_item["enhanced_strokes_paths"] = enhanced_strokes_paths
        new_item["enhanced_sam_paths"] = enhanced_sam_paths
        new_item["timestamp"] = get_timestamp()
        new_data.append(new_item)

    output_name = f"{config_name}_reduced.json"
    output_path = os.path.join(os.path.dirname(args.config), output_name)
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved dilated config to {output_path}")


def reduce_sketch_seg_double_strokes(sketch_dir):
    sketch_image = load_sketch(os.path.join(sketch_dir, "input_sketch.png"))
    segment_dir = os.path.join(sketch_dir, "segment")
    samples_dir = get_sample_subdirs(segment_dir)
    print(f"Found {len(samples_dir)} samples")

    for sample_dir in samples_dir:
        masks_paths = get_auto_sam_masks_paths(sample_dir)
        masks = [load_mask_np_bool_array(mask_path)
                 for mask_path in masks_paths]
        if len(masks) == 0:
            print(f"No masks found in {sample_dir}")
            breakpoint()

        print(f"Found {len(masks)} masks")
        reduce_double_stroke(
            sketch_image=sketch_image,
            seg_masks=masks,
            bg_mask_rgba=None,
            output_dir=os.path.join(sample_dir, "dilated"),
        )


def main():
    # sketch_name = "3D_tri_on_cube"
    # debug_dir = "/home/miatang/projects/stroke-label/experiments/tmp"
    # debug_config = "/home/miatang/projects/stroke-label/configs/test_segment.json"
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--sketch_name", type=str, default=sketch_name)
    # argparser.add_argument("--sample", type=int, default=0)
    # argparser.add_argument("--output_dir", type=str, default=debug_dir)
    # argparser.add_argument("--config", type=str, default=debug_config)
    # argparser.add_argument("--debug", action="store_true", default=False)

    # args = argparser.parse_args()

    # if args.debug:
    #     sketch_path, mask_colorful_path = get_sketch_clean_ann_paths(
    #         args.sketch_name, args.sample
    #     )
    #     run_reduce(args, sketch_path, mask_colorful_path, args.sample)
    # else:
    #     run_on_config(args)
    reduce_sketch_seg_double_strokes(
        "/home/miatang/projects/stroke-label/experiments/2D_circle_sqr_separate")


if __name__ == "__main__":
    main()
