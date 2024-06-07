from util.setup_inpaint import setup_inpaint_model, inpaint_blur_mask
from util.img import save_binary_mask_to_path, load_mask_np_bool_array, save_image_to_path
from util.path import clear_dir
from diffusers.utils import load_image
import json
import numpy as np
import glob
import cv2
import argparse
from sketch_inpainting import align_inpaint

negative_prompt = "bad anatomy, deformed, ugly, disfigured"
interface_data_base_dir = "/home/miatang/projects/stroke-label/interfaces/sketch_edit/static/uploads"

USE_ALIGN_INPAINT = True


def gen_caption_to_inpaint_caption(gen_caption):
    inpaint_caption = gen_caption.replace("A realistic image", "A sketch")
    return inpaint_caption


def inpaint_region(init_image_path, mask_image_path, prompt, style_prompt=None, pipeline=None):
    mask_image = load_image(mask_image_path)
    blurred_mask_image = inpaint_blur_mask(mask_image)
    blurred_mask_path = mask_image_path.replace(".png", "_blurred.png")
    save_image_to_path(blurred_mask_image, blurred_mask_path)
    if USE_ALIGN_INPAINT:
        ref_prompt = f"{prompt}, {style_prompt} style."
        target_prompt = f"{style_prompt} style."
        print(f"Using prompts: {ref_prompt}, {target_prompt}")
        image, mask_image = align_inpaint(ref_img_path=init_image_path, ref_prompt=ref_prompt,
                                          mask_path=blurred_mask_path, init_img_path=init_image_path, target_prompt=target_prompt)
        return image, blurred_mask_path
    else:
        if pipeline is None:
            pipeline = setup_inpaint_model()
        init_image = load_image(init_image_path)
        image = pipeline(prompt="",  negative_prompt=negative_prompt,
                         image=init_image, mask_image=mask_image).images[0]
        return image


def find_hole_mask(layerData, sketch_dir):
    # each layerData object has a "image_src" field that has the path to an image,
    # and we want to sort them by the path name
    layerData = sorted(
        layerData, key=lambda x: x["image_src"].split("/")[-1])
    layer_shape = (layerData[0]["height"], layerData[0]["width"])
    hole_mask = np.zeros(layer_shape)
    layer_masks_base_path = f"{sketch_dir}/final_masks"
    layer_masks_paths = glob.glob(f"{layer_masks_base_path}/*.png")
    layer_masks_paths = [
        mask for mask in layer_masks_paths if "mask_" in mask]
    layer_masks_paths.sort()
    layer_masks = [load_mask_np_bool_array(
        mask, shape=layer_shape) for mask in layer_masks_paths]
    # layer_masks.append(load_mask_np_bool_array(
    #     f"{layer_masks_base_path}/background_mask.png", shape=layer_shape))
    print(f"Loaded {len(layer_masks)} masks")
    if len(layer_masks) != len(layerData):
        print(
            f"Error: Number of masks ({len(layer_masks)}) does not match number of layers ({len(layerData)})")
        exit(1)

    # Get original no offset combined mask
    no_offset_combined_mask = np.zeros(layer_shape)
    for layer_mask in layer_masks:
        no_offset_combined_mask = np.logical_or(
            no_offset_combined_mask, layer_mask)
    inpaint_dir = f"{sketch_dir}/inpainting"
    clear_dir(inpaint_dir)
    save_binary_mask_to_path(
        no_offset_combined_mask, f"{inpaint_dir}/no_offset_combined_mask.png")

    # Get offset combined mask
    offset_combined_mask = np.zeros(layer_shape)
    for layer_i in range(len(layer_masks)):
        offset_x = layerData[layer_i]["x"]
        offset_y = layerData[layer_i]["y"]
        layer_mask = layer_masks[layer_i]

        # shift the mask and clip things out of bounds
        shifted_mask = np.roll(layer_mask, offset_x, axis=1)
        shifted_mask = np.roll(shifted_mask, offset_y, axis=0)
        shifted_mask = np.clip(shifted_mask, 0, 1)
        save_binary_mask_to_path(
            shifted_mask, f"{inpaint_dir}/shifted_mask_{layer_i}.png")

        offset_combined_mask = np.logical_or(
            offset_combined_mask, shifted_mask)

    # erode it a bit so we don't inpaint the edges
    save_binary_mask_to_path(
        offset_combined_mask, f"{inpaint_dir}/offset_combined_mask.png")
    # Get hole mask
    hole_mask = np.logical_xor(offset_combined_mask, no_offset_combined_mask)
    hole_mask = cv2.erode(
        hole_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    hole_mask_path = f"{inpaint_dir}/hole_mask.png"
    save_binary_mask_to_path(hole_mask, hole_mask_path)
    return hole_mask_path


def perform_inpainting(layerData):
    # sketch will be located 2 levels up, so we get substring before that
    # layer_shape = (layerData[0]["height"], layerData[0]["width"])
    sketch_name = layerData[0]["image_src"].split("/")[-3]
    sketch_dir = f"{interface_data_base_dir}/{sketch_name}"
    sketch_path = f"{sketch_dir}/edited_sketch/latest.png"
    caption_json_path = f"{sketch_dir}/caption.json"
    with open(caption_json_path, "r") as f:
        gen_caption = json.load(f)
    gen_caption = gen_caption['caption']

    '''Calculate inpainting mask'''
    hole_mask_path = find_hole_mask(layerData, sketch_dir)

    '''Set up params for inpainting'''
    inpaint_caption = gen_caption_to_inpaint_caption(gen_caption)
    inpaint_dir = f"{sketch_dir}/inpainting"
    inpainted_img, blurred_mask_path = inpaint_region(init_image_path=sketch_path,
                                                      mask_image_path=hole_mask_path, prompt=inpaint_caption, style_prompt='quick sketch')
    output_path = f"{sketch_dir}/inpainting.png"
    save_image_to_path(inpainted_img, output_path)
    return output_path, blurred_mask_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)  # input image
    parser.add_argument("--m", type=str)  # mask
    parser.add_argument("--c", type=str)  # caption
    args = parser.parse_args()
    res_img, blurred_mask = inpaint_region(args.i, args.m, args.c)
    save_image_to_path(res_img, "./inpaint.png")
    save_image_to_path(blurred_mask, "./blurred_mask.png")
