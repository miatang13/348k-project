
from util.img import load_sketch
from util.path import get_sketch_name, get_sketch_style
from util.masks import run_find_rep_masks_on_base_dir, run_turn_rep_masks_to_sam_images
from util.timing import get_timestamp


def run_majority_voting_on_sketch(sketch_path, samples_base_dir, new_item={}):
    sketch_image = load_sketch(sketch_path)
    sketch_name = get_sketch_name(sketch_path)
    sketch_style = get_sketch_style(sketch_path)

    '''
    Run majority voting over all samples
    '''
    mask_plot_path = run_find_rep_masks_on_base_dir(
        base_dir=samples_base_dir
    )
    rep_sam_output_path, rep_stroke_output_path = (
        run_turn_rep_masks_to_sam_images(
            base_dir=samples_base_dir, sketch_image=sketch_image)
    )

    new_item["rep_masks_plot_path"] = mask_plot_path
    new_item["timestamp"] = get_timestamp()
    new_item["rep_sam_output_path"] = rep_sam_output_path
    new_item["rep_stroke_output_path"] = rep_stroke_output_path

    return new_item
