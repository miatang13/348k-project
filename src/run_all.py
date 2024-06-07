'''
Script to run the whole pipeline given a single sketch image
'''

import argparse
import os
import json
import time
import gc
import torch
from caption import run_on_image as caption_image
from gen_img import run_sketch_path_to_images
from segment import Segmenter
from src.util.path import get_sketch_name
from vote_majority import run_majority_voting_on_sketch
from process_rep_masks import process_rep_masks
from reduce_double_strokes import reduce_sketch_seg_double_strokes


def perform_captioning(sketch_path, sketch_name, output_dir):
    caption_json_path = os.path.join(output_dir, "caption.json")
    if not os.path.exists(caption_json_path):
        caption_res_obj = caption_image(sketch_path)
        sketch_caption = caption_res_obj["caption"]
        with open(caption_json_path, "w") as f:
            json.dump(caption_res_obj, f, indent=4)
    else:
        print(f"Already has caption.json for {sketch_name}")
        with open(caption_json_path, "r") as f:
            caption_res_obj = json.load(f)
        sketch_caption = caption_res_obj["caption"]
    print(f"Sketch caption: {sketch_caption}")
    return sketch_caption


def perform_gen(sketch_path, sketch_name, sketch_caption, output_dir, gen_new):

    gen_img_output_dir = os.path.join(output_dir, "gen_imgs")
    if not os.path.exists(gen_img_output_dir):
        os.makedirs(gen_img_output_dir, exist_ok=True)
        existing_images = []
    else:
        existing_images = os.listdir(gen_img_output_dir)
    if len(existing_images) == 0 or gen_new:
        gen_paths = run_sketch_path_to_images(
            sketch_path, sketch_caption, gen_img_output_dir)
        print(f"Generated images: {gen_paths}")
    else:
        print(f"Already has generated images for {sketch_name}")
        gen_paths = [os.path.join(gen_img_output_dir, img)
                     for img in existing_images]
    return gen_paths


def perform_segmentation(output_dir, gen_paths, sketch_path, sketch_name, seg_new):
    seg_output_dir = os.path.join(output_dir, "segment")
    num_gen = len(gen_paths)

    is_all_segmented = True
    for i in range(num_gen):
        seg_subdir = os.path.join(seg_output_dir, f"{i}")
        if not os.path.exists(os.path.join(seg_subdir, "dilated/kept_masks.png")):
            is_all_segmented = False
            break
    if seg_new or not is_all_segmented:
        segmenter = Segmenter()
        os.makedirs(seg_output_dir, exist_ok=True)
        segmenter.run_on_single_sketch(
            sketch_path, gen_paths, seg_output_dir)
        print(f"Segmented images saved to {seg_output_dir}")
    else:
        print(f"Already has segmented images for {sketch_name}")
    return seg_output_dir


def run_process_rep_masks(sketch_path, seg_output_dir, output_dir):
    rep_mask_path = os.path.join(
        seg_output_dir, "match_segs/rep_mask_annotated.png")
    final_masks_dir = os.path.join(output_dir, "final_masks")
    final_stroke_layers_dir = os.path.join(output_dir, "stroke_layers")
    process_rep_masks(rep_mask_path, sketch_path, output_dir=final_masks_dir,
                      stroke_output_dir=final_stroke_layers_dir)


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def run_on_sketch(sketch_path, gen_new=False, seg_new=False, vote_new=False):
    if not os.path.exists(sketch_path):
        print(f"Sketch path {sketch_path} does not exist")
        exit(1)
    start_time = time.time()
    sketch_name = get_sketch_name(sketch_path)
    base_output_dir = "/home/miatang/projects/stroke-label/experiments"
    output_dir = os.path.join(base_output_dir, sketch_name)
    os.makedirs(output_dir, exist_ok=True)
    # copy over sketch
    os.system(f"cp {sketch_path} {output_dir}/input_sketch.png")

    sketch_caption = perform_captioning(sketch_path, sketch_name, output_dir)
    gen_paths = perform_gen(sketch_path, sketch_name,
                            sketch_caption, output_dir, gen_new)

    clear_cache()

    seg_output_dir = perform_segmentation(
        output_dir, gen_paths, sketch_path, sketch_name, seg_new)
    reduce_sketch_seg_double_strokes(sketch_dir=output_dir)

    clear_cache()

    if vote_new:
        run_majority_voting_on_sketch(sketch_path, seg_output_dir)

    run_process_rep_masks(sketch_path, seg_output_dir, output_dir)

    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time} seconds")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sketch", type=str, required=True)
    argparser.add_argument("--gen", action="store_true", default=False)
    argparser.add_argument("--seg", action="store_true", default=False)
    argparser.add_argument("--vote", action="store_true", default=False)

    args = argparser.parse_args()
    sketch_base_dir = "/home/miatang/projects/stroke-label/data/sketch"
    sketch_path = os.path.join(sketch_base_dir, f"{args.sketch}.png")
    run_on_sketch(sketch_path, gen_new=args.gen,
                  seg_new=args.seg, vote_new=args.vote)


if __name__ == "__main__":
    main()
