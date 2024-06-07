
import supervision as sv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/miatang/projects/stroke-label/")
if True:
    from src.util.svg import path_obj_to_points
    from src.util.img import calculate_bbox

sys.path.append("/home/miatang/projects/stroke-label/")

"""
SAM helpers
"""


def masks_to_sam_clean_annotation(masks, mask_annotator):
    # breakpoint()
    sam_results = convert_masks_to_sam_results(masks)
    white_image = (
        np.ones((masks[0].shape[0], masks[0].shape[1], 3),
                dtype=np.uint8) * 255
    )
    annotated = sam_results_to_annotated_image(
        white_image, mask_annotator, sam_results)
    return annotated


def sam_results_to_annotated_image(image, mask_annotator, sam_results):
    detections = sv.Detections.from_sam(sam_results)
    annotated_image = mask_annotator.annotate(image, detections)
    return annotated_image


def convert_masks_to_sam_results(masks):
    """
    Args:
        masks: list of masks (np.array)
    """
    # used for visualization
    bboxes = [calculate_bbox(mask) for mask in masks]
    areas = [bbox[2] * bbox[3] for bbox in bboxes]
    sam_results = [
        {"segmentation": mask, "bbox": bbox, "area": area}
        for mask, bbox, area in zip(masks, bboxes, areas)
    ]
    return sam_results


def show_mask(mask, ax, random_color=False):
    # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green",
                      facecolor=(0, 0, 0, 0), lw=2)
    )


"""
Segment SVG helpers
"""


def visualize_point_mask_results(
    masks, scores, image, input_point, input_label, output_dir
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    print(f"Visualizing {len(masks)} masks")
    vis_paths = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt_output_path = os.path.abspath(
            os.path.join(output_dir, f"mask_{i+1}.png"))
        plt.savefig(plt_output_path)
        plt.close()
        vis_paths.append(plt_output_path)
    return vis_paths


def draw_input_point(input_point, input_label, image, DEBUG_OUTPUT_DIR):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Input Point", fontsize=18)
    show_points(input_point, input_label, plt.gca())
    plt.axis("on")
    plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "input_point.png"))
    plt.close()


def visualize_path(path_obj, image, output_dir, file_name):
    # visualize the path via drawing the points
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    points = path_obj_to_points(path_obj)
    show_points(points, np.ones(len(points)), plt.gca())
    plt.title(f"Points on path", fontsize=18)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"path_{file_name}.png"))
    plt.close()
