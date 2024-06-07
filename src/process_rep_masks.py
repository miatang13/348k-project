import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util.img import load_sketch, save_binary_mask_to_path, \
    save_sketch_to_path, save_image_to_path, parse_masks_from_colorful_annotations, load_sqr_image
from util.colors import generate_pastel_colors
from util.path import clear_dir


def binary_mask_to_rgba(mask, color, init_image=None):
    # Create an empty RGBA image with the same height and width as the mask
    if init_image is not None:
        rgba_image = init_image
    else:
        rgba_image = np.zeros(
            (mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    # Set the RGB values for the True values in the mask
    rgba_image[mask, 0] = color[0]  # Red
    rgba_image[mask, 1] = color[1]  # Green
    rgba_image[mask, 2] = color[2]  # Blue
    rgba_image[mask, 3] = 255       # Alpha for full opacity

    return rgba_image


def process_rep_masks(rep_mask_path, sketch_path, output_dir=None, stroke_output_dir=None):
    clear_dir(output_dir)
    colorful_sam_mask = load_sqr_image(rep_mask_path)
    merged_masks = parse_masks_from_colorful_annotations(colorful_sam_mask)
    print(f"Found {len(merged_masks)} masks")

    '''
    Merge dupe masks 
    '''
    os.makedirs(output_dir, exist_ok=True)
    clear_dir(output_dir)

    '''
    Get covered stroke pixels for each mask, all pixels that are not covered will be "background"
    '''
    if stroke_output_dir is None:
        stroke_layers_dir = os.path.join(output_dir, "stroke_layers")
        stroke_output_dir = stroke_layers_dir
    os.makedirs(stroke_output_dir, exist_ok=True)
    clear_dir(stroke_output_dir)
    sketch_image = load_sketch(sketch_path)
    save_sketch_to_path(sketch_image, os.path.join(
        output_dir, "sketch_image.png"))

    '''
    Get stroke masks for each mask
    '''
    foreground_mask = np.zeros_like(sketch_image)
    for i in range(len(merged_masks)):
        mask = merged_masks[i]
        foreground_mask[mask] = 1
    foreground_mask = foreground_mask.astype('uint8')
    save_binary_mask_to_path(foreground_mask, os.path.join(
        output_dir, "foreground_mask.png"))
    # we can also get the leftover strokes for background stroke
    background_mask = (1 - foreground_mask)
    save_binary_mask_to_path(background_mask.astype('uint8'), os.path.join(
        output_dir, "background_mask.png"))
    # for background stroke pixels, we find that if it's close to a foreground pixel, we should include it
    collect_loose = False
    if collect_loose:
        collected_loose = 0
        search_range = 5
        for x in range(search_range, sketch_image.shape[0] - search_range):
            for y in range(search_range, sketch_image.shape[1] - search_range):
                if background_mask[x, y]:
                    start_x = max(0, x - search_range)
                    end_x = min(sketch_image.shape[0], x + search_range)
                    start_y = max(0, y - search_range)
                    end_y = min(sketch_image.shape[1], y + search_range)
                    search_window = foreground_mask[start_x:end_x,
                                                    start_y:end_y]
                    if np.any(search_window):
                        background_mask[x, y] = False
                        # print(f"Found a loose pixel at {x}, {y}")

                        # find closest mask within a threshold
                        closest_mask_idx = None
                        closest_dist = 50
                        for i, mask in enumerate(merged_masks):
                            mask_idxs = np.where(mask)
                            dist = np.linalg.norm(
                                np.array([x, y]) - np.array([np.mean(mask_idxs[0]), np.mean(mask_idxs[1])]))
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_mask_idx = i
                        if closest_mask_idx is not None:
                            merged_masks[closest_mask_idx][x, y] = True
                            print(
                                f"Assigned loose pixel to mask {closest_mask_idx}")
                            collected_loose += 1
                            background_mask[x, y] = False

        print(f"Collected {collected_loose} loose pixels")
    merged_masks.append(background_mask == 1)

    print(f"Has {len(merged_masks)} masks including background")
    stroke_layers = []
    total_pixels = sketch_image.shape[0] * sketch_image.shape[1]
    stroke_layer_masks = []  # later used to set white pixels within region
    for mask_i in range(len(merged_masks)):
        mask = merged_masks[mask_i]
        stroke_layer = np.zeros_like(sketch_image)
        stroke_layer[mask] = sketch_image[mask]
        # check if the stroke layer is almost empty
        num_stroke_pixels = np.sum(stroke_layer)
        ratio = num_stroke_pixels / total_pixels
        print(f"Num stroke pixels: {num_stroke_pixels}")
        if ratio < 0.1:
            print(
                f"Skipping stroke layer with {num_stroke_pixels} pixels")
            continue
        stroke_layers.append(stroke_layer)
        stroke_layer_masks.append(mask)

    b_w_stroke_output_dir = os.path.join(stroke_output_dir, "bw")
    clear_dir(b_w_stroke_output_dir)
    for i, stroke_layer in enumerate(stroke_layers):
        save_sketch_to_path(stroke_layer, os.path.join(
            b_w_stroke_output_dir, f"{i}.png"), verbose=False)

    '''
    Color each stroke mask with a pretty color
    '''
    hex_colors = generate_pastel_colors(len(stroke_layers))
    colored_stroke_layers = []
    final_colored_sketch = np.ones(
        (sketch_image.shape[0], sketch_image.shape[1], 3)) * 255
    print(f"Shapes: {sketch_image.shape}, {final_colored_sketch.shape}")
    print(f"Has {len(stroke_layers)} stroke layers")
    for i, stroke_layer in enumerate(stroke_layers):
        stroke_layer = stroke_layer == 255
        hex_color = hex_colors[i]
        rgb_color = np.array(list(bytes.fromhex(hex_color[1:])))
        mask_for_layer = stroke_layer_masks[i]
        rgba_stroke_image = np.zeros(
            (mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_stroke_image[mask_for_layer] = (255, 255, 255, 255)
        rgba_stroke_image = binary_mask_to_rgba(
            stroke_layer, rgb_color, init_image=rgba_stroke_image)

        colored_stroke_layers.append(rgba_stroke_image)
        save_image_to_path(rgba_stroke_image, os.path.join(
            stroke_output_dir, f"colored_{i}.png"), verbose=True)
        save_binary_mask_to_path(mask_for_layer, os.path.join(
            output_dir, f"mask_{i}.png"))
        stroke_layer_idxs = np.where(stroke_layer)
        final_colored_sketch[stroke_layer_idxs] = rgb_color

    save_image_to_path(final_colored_sketch.astype('uint8'), os.path.join(
        output_dir, "final_colored_sketch.png"))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True)
    argparser.add_argument("--sketch_path", type=str, required=True)
    argparser.add_argument("--output_dir", default=None)
    argparser.add_argument("--stroke_output_dir", default=None)
    args = argparser.parse_args()
    process_rep_masks(args.dir, args.sketch_path,
                      args.output_dir, args.stroke_output_dir)


if __name__ == "__main__":
    main()
