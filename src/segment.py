import supervision as sv
import numpy as np
import os
import argparse
import json
import time
from util.setup_sam import setup_sam_mask_generator, setup_mask_annotator
from util.img import load_sqr_image, load_sketch, save_image_to_path, load_rgba_image
from util.timing import get_timestamp
from util.sam import segment_image_and_annotate_strokes
from util.bria import run_inference_bria, setup_bria_model
from util.masks import run_find_rep_masks_on_base_dir, run_turn_rep_masks_to_sam_images
from util.path import get_sketch_name, get_gen_img_sample_idx, get_sketch_style
cur_dir = os.path.dirname(os.path.abspath(__file__))


class Segmenter:
    def __init__(self, crop_segment=False):
        self.crop_segment = crop_segment

    def remove_bg(self, output_subdir, bria_model, gen_image_path):
        gen_image_no_bg_path = os.path.join(output_subdir, "gen_no_bg.png")
        if not os.path.exists(gen_image_no_bg_path):
            gen_image_no_bg_rgba = run_inference_bria(
                bria_model, gen_image_path
            )
            save_image_to_path(gen_image_no_bg_rgba,
                               gen_image_no_bg_path)
        else:
            gen_image_no_bg_rgba = load_rgba_image(
                gen_image_no_bg_path)

        return gen_image_no_bg_rgba, gen_image_no_bg_path

    def update_paths_per_gen(self, item_paths_acc, paths_per_gen):
        for key in list(paths_per_gen.keys()):
            if key not in item_paths_acc:
                item_paths_acc[key] = []
            item_paths_acc[key].append(paths_per_gen[key])
        return item_paths_acc

    def save_result_data(self, config_json_path, output_dir, res_data):
        input_config_name = os.path.basename(config_json_path).split(".")[0]
        json_output_dir = os.path.join(output_dir, "json")
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir, exist_ok=True)
        output_json_path = os.path.join(
            json_output_dir, f"{input_config_name}_seg.json")
        with open(output_json_path, "w") as f:
            json.dump(res_data, f, indent=4)
        return output_json_path

    def run_seg_on_gen_image(self, gen_image_path, sketch_path, sketch_style, sketch_name,
                             bria_model, output_dir):
        sample_idx = get_gen_img_sample_idx(gen_image_path)
        output_subdir = os.path.join(
            output_dir, sample_idx
        )

        '''
        Remove bg
        '''
        gen_image_no_bg_rgba, gen_image_no_bg_path = self.remove_bg(output_subdir=output_subdir,
                                                                    bria_model=bria_model, gen_image_path=gen_image_path)
        gen_image_no_bg_rgb = gen_image_no_bg_rgba.convert("RGB")

        '''
        Run segmentation
        '''
        sketch_image = load_sketch(sketch_path)
        gen_image = load_sqr_image(gen_image_path)
        seg_paths = segment_image_and_annotate_strokes(
            image_to_segment=gen_image_no_bg_rgb,
            sketch_image=sketch_image,
            output_dir=output_subdir,
            gen_image_path=gen_image_no_bg_path,
            text_prompt=None
        )

        return seg_paths

    def run_seg_pipeline_on_sketch(self, item, sketch_path, gen_paths, output_dir, bria_model):
        # item will be empty if we are running on a single image
        new_item = item.copy()
        sketch_name = get_sketch_name(sketch_path)
        sketch_style = get_sketch_style(sketch_path)
        item_paths_acc = {}
        # gen_paths = [gen_paths[0]]

        for gen_image_path in gen_paths:
            seg_paths = self.run_seg_on_gen_image(
                gen_image_path=gen_image_path, sketch_path=sketch_path, sketch_style=sketch_style,
                sketch_name=sketch_name, bria_model=bria_model, output_dir=output_dir
            )
            self.update_paths_per_gen(item_paths_acc, seg_paths)

        # Update the item with the paths
        for key in list(item_paths_acc.keys()):
            item_key = "all_" + key
            new_item[item_key] = item_paths_acc[key]

        return new_item

    def run_on_config(self, config_json_path, output_dir, debug=False, debug_idx=0):
        print(f"Using config JSON: {config_json_path}")
        config_data = json.load(open(config_json_path))
        if debug:
            config_data = [config_data[debug_idx]]
        res_data = []
        bria_model = setup_bria_model()

        for item_idx in range(len(config_data)):
            item = config_data[item_idx]
            new_item = self.run_seg_pipeline_on_sketch(
                item, item['sketch_path'], item['gen_paths'], output_dir, bria_model)
            res_data.append(new_item)

        output_json_path = self.save_result_data(
            config_json_path, output_dir, res_data)
        print(f"Saved results to {output_json_path}")

    def run_update_rep_on_existing_config(self, config_json_path, output_dir, debug=False):
        config_data = json.load(open(config_json_path))
        new_config = []
        for item_idx in range(len(config_data)):
            item = config_data[item_idx].copy()
            sketch_name = os.path.basename(item["sketch_path"]).split(".")[0]
            sketch_style = item["style"]
            samples_base_dir = os.path.join(
                output_dir, sketch_style, sketch_name)
            _, mask_plot_path = run_find_rep_masks_on_base_dir(
                base_dir=samples_base_dir
            )
            item["rep_masks_plot_path"] = mask_plot_path
            item["timestamp"] = get_timestamp()

            # We get the sam images from the representative masks and strokes
            sketch_image = load_sketch(item["sketch_path"])
            rep_sam_output_path, rep_stroke_output_path = (
                run_turn_rep_masks_to_sam_images(
                    base_dir=samples_base_dir, sketch_image=sketch_image
                )
            )
            item["rep_sam_output_path"] = rep_sam_output_path
            item["rep_stroke_output_path"] = rep_stroke_output_path
            new_config.append(item)

        # save results to json
        output_config_path = config_json_path.replace(".json", "_rep.json")
        with open(output_config_path, "w") as f:
            json.dump(new_config, f, indent=4)

    def run_on_single_sketch(self, sketch_path, gen_paths, output_dir):
        bria_model = setup_bria_model()
        res_obj = self.run_seg_pipeline_on_sketch(
            {}, sketch_path, gen_paths, output_dir, bria_model)
        return res_obj


def main():
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    default_output_dir = os.path.join(
        cur_dir, "../experiments/segmentation_ram")
    default_sketch_config = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/batch_captions_geometry_gen_results.json"
    default_sketch_config = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/batch_captions_scene_complex_occlude_gen_results_segprompts.json"
    # default_sketch_config = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/batch_captions_scene_no_occlude_gen_results_segprompts.json"
    default_sketch_config = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/batch_captions_scene_slight_occlude_gen_results_segprompts.json"
    all_sketch_config = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/all_captions_gen_results.json"
    argparser.add_argument("--gen_path", type=str, default=None)
    argparser.add_argument("--sketch_path", type=str, default=None)
    argparser.add_argument("--prompt", type=str, default=None)
    argparser.add_argument("--config", type=str, default=default_sketch_config)
    argparser.add_argument("--output_dir", type=str,
                           default=default_output_dir)
    argparser.add_argument("--crop", action="store_true", default=False)
    argparser.add_argument("--all", action="store_true", default=False)
    argparser.add_argument("--debug", action="store_true", default=False)
    argparser.add_argument("--debug_idx", type=int, default=0)
    argparser.add_argument("--update_rep", action="store_true", default=False)
    argparser.add_argument("--no_grounded", action="store_true", default=False)
    args = argparser.parse_args()

    if args.update_rep:
        segmenter = Segmenter(crop_segment=args.crop)
        segmenter.run_update_rep_on_existing_config(
            config_json_path=args.config, output_dir=args.output_dir, debug=args.debug
        )
        return

    if args.gen_path and args.sketch_path:
        print(f"Single image segmentation")
        segmenter = Segmenter(crop_segment=args.crop)
        gen_image = load_sqr_image(args.gen_path)
        sketch_image = load_sketch(args.sketch_path)
        sketch_name = os.path.basename(args.sketch_path).split(".")[0]
        output_subdir = os.path.join(
            args.output_dir, "single_image_debug", sketch_name)
        segment_image_and_annotate_strokes(
            image_to_segment=gen_image,
            sketch_image=sketch_image,
            output_dir=output_subdir,
            gen_image_path=args.gen_path,
            text_prompt=args.prompt,
        )
        return

    if args.all:
        args.config = all_sketch_config

    segmenter = Segmenter(crop_segment=args.crop)
    segmenter.run_on_config(
        config_json_path=args.config, output_dir=args.output_dir,
        debug=args.debug, debug_idx=args.debug_idx
    )

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Done!")


if __name__ == "__main__":
    main()
