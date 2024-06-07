from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import requests
import os
import numpy as np
import argparse

img_res = 1024
img_size = (img_res, img_res)


def calculate_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]


def invert_sketch(image):
    return ImageOps.invert(image)


def load_mask_image(image_path, verbose=False):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize(img_size)
    if verbose:
        print(f"Loaded mask from path: {image_path}")
    return image


def load_mask_np_bool_array(image_path, shape=None, verbose=False):
    image = Image.open(image_path)
    # image = image.resize(img_size)
    image = image.convert("L")
    if shape is None:
        image = image.resize(img_size)
    else:
        image = image.resize(shape)
    if verbose:
        print(f"Loaded mask from path: {image_path}")
    image = np.array(image)
    image = image == 255
    return image


def load_sketch(image_path, shape=None, verbose=False, invert=True):
    image = Image.open(image_path)
    image = image.convert("L")
    if shape is None:
        image = image.resize(img_size)
    else:
        image = image.resize(shape)
    if verbose:
        print(f"Loaded sketch from path: {image_path}")
    if invert:
        image = invert_sketch(image)
    image = np.array(image)
    return image


def load_rgba_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGBA")
    image = image.resize(img_size)
    print(f"Loaded image from path: {image_path}")
    return image


def load_sqr_image(image_path):
    # can be b&w so check for that
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize(img_size)
    print(f"Loaded image from path: {image_path}")
    # image = np.array(image)
    return image


def load_url_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image


def bw_img_path_to_bool_mask_array(image_path):
    bw_image = load_sketch(image_path, invert=False)
    bw_image = np.array(bw_image)
    bw_image = bw_image > 0
    return bw_image


def save_image_to_path(image, output_path, verbose=True):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Created output subdirectory: {os.path.dirname(output_path)}")
    if not output_path.endswith(".png") and not output_path.endswith(".jpg"):
        output_path = f"{output_path}.png"
    output_path = os.path.abspath(output_path)
    image.save(output_path)
    if verbose:
        print(f"Saved image to {output_path}")

    return output_path


def save_binary_mask_to_path(mask, output_path, verbose=True):
    mask = mask.astype(np.uint8) * 255
    save_image_to_path(mask, output_path, verbose)
    return output_path


def save_sketch_to_path(sketch, output_path, verbose=True):
    sketch = sketch.astype(np.uint8)
    save_image_to_path(sketch, output_path, verbose)
    return output_path


def parse_masks_from_colorful_annotations(annotated_image):
    # find all colors in the image, and then make masks for each color
    if not isinstance(annotated_image, np.ndarray):
        annotated_image = np.array(annotated_image)
    unique_colors = np.unique(
        annotated_image.reshape(-1, annotated_image.shape[2]), axis=0
    )
    masks = []
    img_h, img_w = annotated_image.shape[:2]
    corners = [(0, 0), (0, img_w - 1), (img_h - 1, 0), (img_h - 1, img_w - 1)]
    for color in unique_colors:
        # if it's very close to white, we ignore
        # if np.all(color > 200):
        #     continue
        mask = np.all(annotated_image == color, axis=-1)
        # check if background mask by checking it contains 4 corners
        if np.all([mask[corner] for corner in corners]):
            continue

        masks.append(mask)
    print(f"Found {len(masks)} masks")
    # need to invert the masks
    # masks = [np.logical_not(mask) for mask in masks]
    return masks


def save_masks_to_path(masks, output_dir, verbose=True):
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            os.remove(f"{output_dir}/{file}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output subdirectory: {output_dir}")
    for i, mask in enumerate(masks):
        save_binary_mask_to_path(
            mask, f"{output_dir}/mask-{i}.png", verbose=verbose)


"""
Testing functions 
"""


def test_ann_to_masks():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ann_img", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()
    annotated_image = load_sqr_image(args.ann_img)
    masks = parse_masks_from_colorful_annotations(annotated_image)
    for i, mask in enumerate(masks):
        save_binary_mask_to_path(mask, f"{args.output_dir}/mask-{i}.png")
    save_image_to_path(
        annotated_image, f"{args.output_dir}/annotated_image.png")
