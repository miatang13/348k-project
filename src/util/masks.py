
from itertools import product
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
import math
import glob
from sklearn.decomposition import PCA
import sys
import cv2

sys.path.append("/home/miatang/projects/stroke-label/src")
if True:
    from util.stroke import color_stroke_from_ann_img
    from util.vis_data import vis_consensus_masks
    from util.path import get_sample_subdirs
    from util.setup_sam import setup_mask_annotator
    from util.visualize import masks_to_sam_clean_annotation
    from util.img import (
        bw_img_path_to_bool_mask_array,
        save_image_to_path,
        load_mask_np_bool_array,
        load_sketch,
        save_binary_mask_to_path
    )


def compute_l2_loss(mask1, mask2):
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape")
    if not isinstance(mask1, np.ndarray):
        mask1 = np.array(mask1)
    if not isinstance(mask2, np.ndarray):
        mask2 = np.array(mask2)
    # return np.linalg.norm(mask1 - mask2)
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    return np.sum((mask1 - mask2) ** 2)


def compute_iou(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) of two binary masks.

    Parameters:
    - mask1: numpy array, binary mask where 1 indicates the object and 0 is the background.
    - mask2: numpy array, binary mask where 1 indicates the object and 0 is the background.

    Returns:
    - iou: float, Intersection over Union metric.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def is_mask_contained(mask1, mask2):
    """
    Check if mask1 is contained within mask2.

    Args:
    mask1 (numpy.ndarray): The first mask.
    mask2 (numpy.ndarray): The second mask.

    Returns:
    bool: True if mask1 is contained within mask2, False otherwise.
    """
    # Ensure the masks are binary (0 or 1)
    mask1_binary = mask1 > 0
    mask2_binary = mask2 > 0

    num_contained_1 = np.sum(mask1_binary[mask2_binary])
    num_contained_2 = np.sum(mask2_binary[mask1_binary])

    ratio_1 = num_contained_1 / np.sum(mask1_binary)
    ratio_2 = num_contained_2 / np.sum(mask2_binary)
    print(f"Containment: Ratio 1: {ratio_1}, Ratio 2: {ratio_2}")

    is_contained = ratio_1 > 0.9 or ratio_2 > 0.9
    has_full_contained = ratio_1 == 1 or ratio_2 == 1

    # Check if all non-zero pixels in mask1 are also non-zero in mask2
    return is_contained, has_full_contained


def merge_similar_masks(masks, output_dir=None, iou_threshold=0.6, no_save=False):
    """
    Merges similar masks in an array based on the IoU threshold.

    Parameters:
    - masks: list of numpy arrays, each being a binary mask.
    - iou_threshold: float, the threshold above which two masks are considered similar enough to merge.

    Returns:
    - list of numpy arrays, merged masks.
    """
    merged = []
    while masks:
        current = masks.pop(0)
        indices_to_merge = []

        # Check all remaining masks to see if they should merge with 'current'
        for i, mask in enumerate(masks):
            iou = compute_iou(current, mask)
            # print(f"IoU between {i} and {i+1}: {iou}")

            containment, has_full_contained = is_mask_contained(current, mask)
            l2_loss = compute_l2_loss(current, mask)
            if containment:
                print("l2: ", l2_loss)
            is_contained_artifact = (
                containment and l2_loss < 50000) or has_full_contained

            if iou > iou_threshold or is_contained_artifact:
                indices_to_merge.append(i)

        # Merge the masks
        for i in sorted(indices_to_merge, reverse=True):
            current = np.logical_or(current, masks.pop(i))

        merged.append(current)

    # plot the merged masks
    if not no_save:
        if output_dir is None:
            masks_dir = os.path.dirname(masks[0])
            output_dir = os.path.join(masks_dir, "merged")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory {output_dir}")
        for i, mask in enumerate(merged):
            save_binary_mask_to_path(mask, f"{output_dir}/merged_mask_{i}.png")

    return merged


def fill_holes_in_masks(masks, output_dir=None):
    """
    Fills holes in binary masks.

    Parameters:
    - masks: list of numpy arrays, each being a binary mask.

    Returns:
    - list of numpy arrays, masks with holes filled.
    """
    filled_masks = []
    for mask in masks:
        mask = np.logical_not(mask)
        mask = mask.astype(np.uint8)
        # contours, _ = cv2.findContours(
        #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(mask, contours, 0, 255, -1)
        # filled_masks.append(mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        morphed = morphed.astype(bool)
        morphed = np.logical_not(morphed)

        to_dilate = morphed.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(to_dilate, kernel, iterations=1)
        dilated = dilated.astype(bool)

        res = dilated

        filled_masks.append(res)

    # Save filled masks
    if output_dir is not None:
        for i, mask in enumerate(filled_masks):
            save_binary_mask_to_path(mask, f"{output_dir}/filled_mask_{i}.png")

    return filled_masks


def count_disjoint_components(mask):
    # Check if the mask is empty
    if mask.size == 0:
        raise ValueError("The mask is empty.")

    # Convert the mask to uint8 if it's not
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask = mask * 255

    # Threshold the mask to ensure it is binary
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8)

    # Return the number of components minus the background
    return num_labels - 1


def is_sparse(mask):
    # Count components
    component_count = count_disjoint_components(mask)
    print(f"Number of components: {component_count}")

    # Check if the number of components is above the threshold
    threshold = 3
    return component_count > threshold


def find_consensus_masks(mask_sets, output_dir, filter_by_cluster=True, use_IoU=False,
                         iou_threshold=0.4):
    subdir = os.path.join(output_dir, "consensus_masks_process")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    num_sets = len(mask_sets)
    consensus_masks = []

    # Create a list of tuples (set_index, mask_index) for all masks
    indices = [(i, j) for i, masks in enumerate(mask_sets)
               for j in range(len(masks))]

    # Compute L2 loss between each pair of masks across all sets
    if use_IoU:
        IoU_scores = {}
        for (set1_idx, mask1_idx), (set2_idx, mask2_idx) in product(indices, repeat=2):
            if set1_idx != set2_idx:
                mask1 = mask_sets[set1_idx][mask1_idx]
                mask2 = mask_sets[set2_idx][mask2_idx]
                iou_score = compute_iou(mask1, mask2)
                IoU_scores[(set1_idx, mask1_idx, set2_idx,
                            mask2_idx)] = iou_score
    else:
        l2_losses = {}
        for (set1_idx, mask1_idx), (set2_idx, mask2_idx) in product(indices, repeat=2):
            if set1_idx != set2_idx:  # Avoid self-comparison
                mask1 = mask_sets[set1_idx][mask1_idx]
                mask2 = mask_sets[set2_idx][mask2_idx]
                l2_loss = compute_l2_loss(mask1, mask2)
                l2_losses[(set1_idx, mask1_idx, set2_idx, mask2_idx)] = l2_loss

        all_losses = list(l2_losses.values())
        all_losses.sort()
        print(f"Min loss: {all_losses[0]}, Max loss: {all_losses[-1]}")
        percentile = 0.35  # 15th percentile
        threshold = all_losses[int(len(all_losses) * percentile)]

    # Identify masks with similar matches in multiple sets
    for set1_idx, mask1_idx in indices:
        similar_count = 0
        for set2_idx in range(num_sets):
            if set1_idx == set2_idx:
                continue

            for mask2_idx in range(len(mask_sets[set2_idx])):
                if use_IoU:
                    cur_pair_score = IoU_scores[(
                        set1_idx, mask1_idx, set2_idx, mask2_idx)]
                    if cur_pair_score < iou_threshold:
                        continue
                    other_pairs_within_set1 = [
                        (set1_idx, other_mask_idx, set2_idx, mask2_idx)
                        for other_mask_idx in range(len(mask_sets[set1_idx]))
                        if other_mask_idx != mask1_idx
                    ]
                    has_better_match = any(
                        IoU_scores[pair] > cur_pair_score for pair in other_pairs_within_set1
                    )
                    if has_better_match:
                        continue
                else:  # use L2
                    cur_pair_loss = l2_losses[(
                        set1_idx, mask1_idx, set2_idx, mask2_idx)]

                    """
                    First we check if the current pair is similar enough
                    """
                    if cur_pair_loss > threshold:
                        continue

                    """
                    Second we check if any other mask in set 1 can outperform the current pair
                    """
                    other_pairs_within_set1 = [
                        (set1_idx, other_mask_idx, set2_idx, mask2_idx)
                        for other_mask_idx in range(len(mask_sets[set1_idx]))
                        if other_mask_idx != mask1_idx
                    ]
                    has_better_match = any(
                        l2_losses[pair] < cur_pair_loss for pair in other_pairs_within_set1
                    )
                    if has_better_match:
                        continue

                similar_count += 1
        similar_pct = similar_count / (num_sets - 1)
        # print(
        #     f"Mask {mask1_idx} in set {set1_idx} has {similar_pct} similar matches")
        if similar_pct >= 0.5:  # 75% of the sets
            consensus_masks.append((set1_idx, mask1_idx))

    if filter_by_cluster and len(consensus_masks) > 1:
        print(f"Filtering {len(consensus_masks)} consensus masks")
        if use_IoU:
            filtered_consensus_masks = merge_similar_masks(
                consensus_masks, output_dir=None, no_save=True)
            return filtered_consensus_masks, True
        else:  # use L2
            # Extract consensus masks
            consensus_mask_images = [
                mask_sets[set_idx][mask_idx] for set_idx, mask_idx in consensus_masks
            ]

            # Cluster similar consensus masks to avoid duplicates
            # Need to compute cluster_eps -- maximum L2 loss for masks to be considered in the same cluster
            consensus_mask_vectors = [mask.flatten()
                                      for mask in consensus_mask_images]
            pairwise_distances = []
            for i, j in product(range(len(consensus_mask_vectors)), repeat=2):
                if i < j:
                    distance = compute_l2_loss(
                        consensus_mask_vectors[i], consensus_mask_vectors[j]
                    )
                    pairwise_distances.append(distance)

            # Visualize pairwise distances for debugging
            plt.hist(pairwise_distances, bins=30)
            plt.title("Pairwise L2 Distances Distribution")
            plt.xlabel("L2 Distance")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(subdir, "pairwise_distances.png"))
            plt.clf()

            # Determine cluster_eps dynamically
            if pairwise_distances:
                p25 = np.percentile(pairwise_distances, 25)
                p50 = np.percentile(pairwise_distances, 50)
                p75 = np.percentile(pairwise_distances, 75)
                print(
                    f"25th percentile: {p25}, 50th percentile: {p50}, 75th percentile: {p75}"
                )
                cluster_eps = np.percentile(pairwise_distances, 5)
            else:
                cluster_eps = 10  # Default fallback value
            # breakpoint()
            # Print computed cluster_eps for debugging
            print("Computed cluster_eps:", cluster_eps)

            # Cluster similar consensus masks to avoid duplicates
            clustering = DBSCAN(eps=60, min_samples=1).fit(
                consensus_mask_vectors)
            unique_clusters = set(clustering.labels_)
            # visualize_clustering(consensus_mask_vectors,
            #                      clustering.labels_, subdir)

            # Print clustering labels for debugging
            print("Clustering labels:", clustering.labels_)

            filtered_consensus_masks = []
            for cluster in unique_clusters:
                cluster_indices = [
                    i for i, label in enumerate(clustering.labels_) if label == cluster
                ]
                representative_index = cluster_indices[0]
                filtered_consensus_masks.append(
                    consensus_masks[representative_index])

            return filtered_consensus_masks, True
        # filtered_consensus_masks = merge_similar_masks(
        #     consensus_masks, output_dir=None, no_save=True)
        # return filtered_consensus_masks, True
        # return consensus_masks, True
    else:
        print(f"Not filtering {len(consensus_masks)} consensus masks")
        return consensus_masks, False


def run_find_rep_masks_on_base_dir(base_dir, no_filtering=False):
    # we loop over all samples by globbing subdirectories
    subdirs = get_sample_subdirs(base_dir)
    subdir_kept_masks = [
        os.path.join(subdir, "dilated/kept_masks") for subdir in subdirs
    ]
    mask_sets = []
    for subdir in subdir_kept_masks:
        masks = []
        for file in os.listdir(subdir):
            masks.append(bw_img_path_to_bool_mask_array(f"{subdir}/{file}"))
        mask_sets.append(masks)
        # breakpoint()
    if len(mask_sets) < 2:
        print(f"Found {len(mask_sets)} sets of masks")
        breakpoint()

    output_dir = os.path.join(base_dir, "match_segs")
    consensus_masks, filtered = find_consensus_masks(
        mask_sets, output_dir, filter_by_cluster=not no_filtering
    )
    consensus_mask_arr = consensus_mask_set_to_array(
        consensus_masks, mask_sets)
    vis_consensus_masks(consensus_mask_arr, output_dir, no_filtering=False,
                        fig_title="Initial Representative Masks", mask_output_name="init_consensus_mask")
    no_dup_consensus_masks_arr = remove_dupe_consensus_masks(
        consensus_mask_arr, output_dir)
    no_dup_consensus_masks_arr = fill_holes_in_masks(
        no_dup_consensus_masks_arr, output_dir)
    no_dup_consensus_masks_arr = [
        mask for mask in no_dup_consensus_masks_arr if not is_sparse(mask)]
    print(
        f"Found {len(no_dup_consensus_masks_arr)} consensus masks from {len(mask_sets)} sets")
    mask_plot_path = vis_consensus_masks(
        no_dup_consensus_masks_arr, output_dir, not filtered
    )
    masks_paths = save_each_rep_mask(
        no_dup_consensus_masks_arr, output_dir, not filtered
    )
    return mask_plot_path


def save_each_rep_mask(consensus_masks, output_dir, no_filtering):
    # Save each individual mask as a png
    mask_output_dir = "rep_masks"
    if no_filtering:
        mask_output_dir += "_no_filtering"
    mask_output_dir = os.path.join(output_dir, mask_output_dir)
    if os.path.exists(mask_output_dir):
        # remove all files
        for file in os.listdir(mask_output_dir):
            if file.endswith(".png"):
                os.remove(os.path.join(mask_output_dir, file))
    else:
        os.makedirs(mask_output_dir, exist_ok=True)
        print(f"Created directory {mask_output_dir}")

    masks_paths = []
    for i in range(len(consensus_masks)):
        mask = consensus_masks[i]
        mask_img = (mask.astype(np.uint8)) * 255
        path = os.path.join(mask_output_dir, f"mask_{i}.png")
        save_image_to_path(mask_img, path)
        masks_paths.append(path)

    return masks_paths


def consensus_mask_set_to_array(consensus_masks, mask_sets):
    consensus_masks_arr = []
    for i, (set_idx, mask_idx) in enumerate(consensus_masks):
        mask = mask_sets[set_idx][mask_idx]
        consensus_masks_arr.append(mask)
    return consensus_masks_arr


def remove_dupe_consensus_masks(consensus_mask_arr, output_dir):
    merged_consensus_masks_arr = merge_similar_masks(
        consensus_mask_arr, output_dir, iou_threshold=0.4, no_save=False)

    return merged_consensus_masks_arr


def run_turn_rep_masks_to_sam_images(base_dir, sketch_image):
    match_seg_dir = f"{base_dir}/match_segs"
    if not os.path.exists(match_seg_dir):
        print(f"Directory {match_seg_dir} does not exist")
        exit(1)

    rep_masks = glob.glob(f"{match_seg_dir}/rep_masks/*.png")
    if len(rep_masks) == 0:
        rep_masks = glob.glob(f"{match_seg_dir}/rep_masks_no_filtering/*.png")
    rep_masks = [load_mask_np_bool_array(mask) for mask in rep_masks]
    mask_annotator = setup_mask_annotator()
    clean_ann = masks_to_sam_clean_annotation(rep_masks, mask_annotator)
    sam_output_path = f"{match_seg_dir}/rep_mask_annotated.png"
    save_image_to_path(clean_ann, sam_output_path)
    ann_strokes = color_stroke_from_ann_img(
        annotated_image=clean_ann, sketch_image=sketch_image
    )
    stroke_output_path = f"{match_seg_dir}/rep_mask_annotated_stroke.png"
    save_image_to_path(ann_strokes, stroke_output_path)

    return sam_output_path, stroke_output_path


def test_match_masks():
    argparser = argparse.ArgumentParser()
    base_path = "/home/miatang/projects/stroke-label/experiments/segmentation/geometry/3D_tri_on_cube"
    dir1_idx = 0
    dir2_idx = 1
    dir3_idx = 2
    default_dir1 = f"{base_path}/{dir1_idx}/dilated/kept_masks"
    default_dir2 = f"{base_path}/{dir2_idx}/dilated/kept_masks"
    default_dir3 = f"{base_path}/{dir3_idx}/dilated/kept_masks"
    base_dir_def = "/home/miatang/projects/stroke-label/experiments/segmentation/geometry/3D_all_separate"
    argparser.add_argument("--base_dir", type=str, default=base_dir_def)
    argparser.add_argument("--masks_dir1", type=str, default=default_dir1)
    argparser.add_argument("--masks_dir2", type=str, default=default_dir2)
    argparser.add_argument("--masks_dir3", type=str, default=default_dir3)
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=f"{base_path}/match_segs_{dir1_idx}_{dir2_idx}_{dir3_idx}",
    )
    argparser.add_argument(
        "--no_filtering", action="store_true", default=False)
    args = argparser.parse_args()

    if args.base_dir is not None:
        run_find_rep_masks_on_base_dir(args.base_dir, args.no_filtering)
    else:
        masks_set1 = []
        masks_set2 = []
        masks_set3 = []
        for file in os.listdir(args.masks_dir1):
            masks_set1.append(
                bw_img_path_to_bool_mask_array(f"{args.masks_dir1}/{file}")
            )
        for file in os.listdir(args.masks_dir2):
            masks_set2.append(
                bw_img_path_to_bool_mask_array(f"{args.masks_dir2}/{file}")
            )
        for file in os.listdir(args.masks_dir3):
            masks_set3.append(
                bw_img_path_to_bool_mask_array(f"{args.masks_dir3}/{file}")
            )
        mask_sets = [masks_set1, masks_set2, masks_set3]
        output_dir = os.path.join(args.base_dir, "match_segs")
        consensus_masks, filtered = find_consensus_masks(
            mask_sets, output_dir, filter_by_cluster=not args.no_filtering
        )
        print(
            f"Found {len(consensus_masks)} consensus masks from {len(mask_sets)} sets"
        )
        vis_consensus_masks(consensus_masks, mask_sets,
                            output_dir, not filtered)
        save_each_rep_mask(consensus_masks, mask_sets,
                           output_dir, not filtered)


if __name__ == "__main__":
    # test_match_masks()
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--base_dir", type=str)
    # argparser.add_argument("--sketch", type=str)
    # args = argparser.parse_args()
    # # run_find_rep_masks_on_base_dir(base_dir=args.base_dir)
    # sketch_image = load_sketch(args.sketch)
    # # save it for sanity check
    # save_image_to_path(
    #     sketch_image, "/home/miatang/projects/stroke-label/experiments/tmp/sketch.png"
    # )
    # run_turn_rep_masks_to_sam_images(
    #     base_dir=args.base_dir, sketch_image=sketch_image)
    mask1 = "/home/miatang/projects/stroke-label/experiments/muscular_sheeps/final_masks/mask_5.png"
    # mask2 = "/home/miatang/projects/stroke-label/experiments/3D_all_slight_occlude/segment/match_segs/rep_masks/mask_3.png"
    mask1 = load_mask_np_bool_array(mask1)
    # mask2 = load_mask_np_bool_array(mask2)
    # print(compute_iou(mask1, mask2))
    # print(is_mask_contained(mask1, mask2))
    print(is_sparse(mask1))
