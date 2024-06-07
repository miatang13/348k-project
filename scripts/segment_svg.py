from segment_anything import sam_model_registry, SamPredictor
import os
import numpy as np
import argparse
import json
from util.img import load_sqr_image, save_image_to_path
from util.svg import load_svg, save_svg, path_obj_to_points, color_specific_path
from util.visualize import visualize_point_mask_results
from util.majority_voting import majority_voting
import cv2

DEBUG_OUTPUT_DIR = "/home/miatang/projects/stroke-label/experiments/features/sam_point"

if not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    print(f"Created directory: {DEBUG_OUTPUT_DIR}")


def load_sam_predictor():
    sam_checkpoint = (
        "/home/miatang/projects/stroke-label/data/ckpts/sam_vit_h_4b8939.pth"
    )
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def query_predictor_w_point(predictor, image, input_point):
    # Process the image to produce an image embedding by calling
    predictor.set_image(image)
    #  labels 1 (foreground point) or 0 (background point)
    input_label = np.array([1])
    # draw_input_point(input_point, input_label, image, DEBUG_OUTPUT_DIR)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks, scores, logits


def run_single_point_query(args):
    gen_img = load_sqr_image(args.gen_img_path)
    gen_img = np.array(gen_img)
    paths, attributes, svg_attributes = load_svg(args.svg_path)
    predictor = load_sam_predictor()

    paths = paths[0]

    for idx, path in enumerate(paths):
        output_dir = os.path.join(DEBUG_OUTPUT_DIR, f"path_{idx}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        # visualize_path(path, gen_img, output_dir, idx)
        save_svg(
            [path], attributes, svg_attributes, os.path.join(output_dir, "path.svg")
        )
        points = path_obj_to_points(path)
        input_point = points[50].reshape(1, 2)
        masks, scores, logits = query_predictor_w_point(predictor, gen_img, input_point)

        visualize_point_mask_results(
            masks, scores, gen_img, input_point, np.array([1]), output_dir
        )
        print(f"Processed polyline {idx} with point {input_point}")


def run_majority_voting(svg_path, gen_img_path, output_dir=DEBUG_OUTPUT_DIR):
    gen_img = load_sqr_image(gen_img_path)
    gen_img = np.array(gen_img)
    paths, attributes, svg_attributes = load_svg(svg_path)
    predictor = load_sam_predictor()

    paths = paths[0]
    paths = paths[:1]  # debugging
    res_log = []

    print(f"Processing {len(paths)} polylines")

    subdir = os.path.join(output_dir, os.path.basename(svg_path).split(".")[0])

    for idx, path in enumerate(paths):
        output_dir = os.path.join(subdir, f"path_{idx}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        else:
            # remove existing files
            for f in os.listdir(output_dir):
                # if is directory we ignore
                if os.path.isdir(os.path.join(output_dir, f)):
                    continue
                os.remove(os.path.join(output_dir, f))

        save_svg(
            [path], attributes, svg_attributes, os.path.join(output_dir, "path.svg")
        )
        points = path_obj_to_points(path, num_samples=5)
        # seg_result_per_point = []
        path_log = {}
        color_specific_path(svg_path, idx, output_dir)

        print(f"Processing polyline {idx} with {len(points)} points")
        seg_masks = []
        for idx, input_point in enumerate(points):
            masks, scores, logits = query_predictor_w_point(
                predictor, gen_img, input_point.reshape(1, 2)
            )
            point_output_dir = os.path.join(output_dir, f"point_{idx}")
            vis_paths = visualize_point_mask_results(
                masks,
                scores,
                gen_img,
                input_point.reshape(1, 2),
                np.array([1]),
                point_output_dir,
            )
            point_log = {"vis_paths": vis_paths}
            path_log["point_logs"] = point_log
            seg_masks.extend(masks)

        # save seg_masks
        np.save(os.path.join(output_dir, "seg_masks.npy"), seg_masks)

        # save images
        majority_mask = majority_voting(seg_masks)
        binary_maj_mask = majority_mask * 255
        # overlay this mask on gen image
        w, h = gen_img.shape[:2]
        rgba_maj_mask = np.zeros((w, h, 4), dtype=np.uint8)
        rgba_maj_mask[:, :, 0] = binary_maj_mask
        rgba_maj_mask = rgba_maj_mask.astype(np.uint8)
        rgba_gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2RGBA)
        rgba_gen_img[:, :, 3] = 255
        overlay_img = cv2.addWeighted(rgba_gen_img, 0.6, rgba_maj_mask, 0.4, 0)
        overlay_img[:, :, 3] = 255

        path_log["majority_mask_path"] = save_image_to_path(
            binary_maj_mask, os.path.join(output_dir, "majority_mask.png")
        )
        path_log["overlay_on_gen_path"] = save_image_to_path(
            overlay_img, os.path.join(output_dir, "overlay_on_gen.png")
        )
        path_log["gen_img_path"] = save_image_to_path(
            rgba_gen_img, os.path.join(output_dir, "gen.png")
        )
        path_log["svg_path"] = svg_path
        path_log["rgba_maj_mask_path"] = save_image_to_path(
            rgba_maj_mask, os.path.join(output_dir, "rgba_maj_mask.png")
        )
        res_log.append(path_log)

    # save log
    log_path = os.path.join(subdir, "majority_voting_log.json")
    with open(log_path, "w") as f:
        json.dump(res_log, f, indent=4)
    print(f"Saved majority voting log to {log_path}")


def main():
    argparser = argparse.ArgumentParser()
    base_dir = "/home/miatang/projects/stroke-label"
    test_svg = f"{base_dir}/data/images/sketch/baby_penguin/penguin_style2.svg"
    test_gen = f"{base_dir}/experiments/sketch_2_img/baby_penguin/penguin_style2/a-realistic-image-of-a-penguin-standing-on-a-snowy-surface-1.png"
    argparser.add_argument("--svg_path", type=str, default=test_svg)
    argparser.add_argument("--gen_img_path", type=str, default=test_gen)
    args = argparser.parse_args()
    run_majority_voting(args.svg_path, args.gen_img_path)


if __name__ == "__main__":
    main()
