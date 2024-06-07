import json
import argparse
import sys
import os

sys.path.append("/home/miatang/projects/stroke-label/src")
from util.img import load_sketch, load_sqr_image, save_image_to_path
from util.sam import color_sketch_strokes_by_ann_img


def main():
    debug_config = "/home/miatang/projects/stroke-label/configs/test_segment.json"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default=debug_config)
    args = argparser.parse_args()

    config = json.load(open(args.config, "r"))
    config_name = os.path.basename(args.config).split(".")[0]
    new_data = []

    for item in config:
        sketch = load_sketch(item["sketch_path"])
        dilated_sketch_paths = []
        new_item = item.copy()

        for i in range(len(item["gen_paths"])):
            seg_img = load_sqr_image(item["clean_gen_ann_paths"][i])
            output_img = color_sketch_strokes_by_ann_img(sketch, seg_img)
            output_dir = os.path.dirname(item["clean_gen_ann_paths"][i])
            dilated = save_image_to_path(
                output_img, f"{output_dir}/seg_stroke_dilated.png"
            )
            dilated_sketch_paths.append(dilated)

        new_item["dilated_seg_stroke_paths"] = dilated_sketch_paths
        new_data.append(new_item)

    output_name = f"{config_name}_dilated.json"
    output_path = os.path.join(os.path.dirname(args.config), output_name)
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved dilated config to {output_path}")


if __name__ == "__main__":
    main()
