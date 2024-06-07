"""
Background Removal
"""

import numpy as np
from skimage import io
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import argparse
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Main funcs
"""


def setup_bria_model():
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True
    )
    model.to(device)
    return model


def run_inference_bria(model, image_path):
    # prepare input
    # image_path = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/simple_nature/tree_mountain_center/a-realistic-image-of-a-tree-on-a-hill-0.png"
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image_bria(orig_im, orig_im_size).to(device)

    # inference
    result = model(image)

    # post process
    result_image = postprocess_image_bria(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(image_path)
    no_bg_image.paste(orig_image, mask=pil_im)

    # save image
    # no_bg_image.save("no_bg_image.png")
    return no_bg_image


"""
Utils
"""


def preprocess_image_bria(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    )
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image_bria(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


def bria_input_to_output_path(image_path):
    output_path = image_path.replace(".png", "_no_bg.png")
    return output_path


def run_on_image(image_path):
    model = setup_bria_model()
    no_bg_image = run_inference_bria(model, image_path)
    output_path = bria_input_to_output_path(image_path)
    no_bg_image.save(output_path)
    print(f"Saved no bg image to {output_path}")
    return output_path


def run_on_config(config_json_path, debug=False):
    config_data = json.load(open(config_json_path, "r")).copy()
    model = setup_bria_model()
    output_paths = []
    for item_idx in range(len(config_data)):
        item = config_data[item_idx]
        gen_paths = item["gen_paths"]
        for gen_image_path in gen_paths:
            run_inference_bria(model, gen_image_path)
            output_path = bria_input_to_output_path(gen_image_path)
            output_paths.append(output_path)

        config_data[item_idx]["remove_bg_gen_paths"] = output_paths

    output_config_path = config_json_path.replace(".json", "_no_bg.json")
    with open(output_config_path, "w") as f:
        json.dump(config_data, f, indent=4)


def main():
    argparser = argparse.ArgumentParser()
    def_img = "/home/miatang/projects/stroke-label/experiments/sketch_2_img/geometry/2D_circle_sqr_occlude/a-realistic-image-of-a-white-circle-and-a-white-square-0.png"
    argparser.add_argument("--img", type=str, default=def_img)
    argparser.add_argument("--config", type=str, default=None)
    args = argparser.parse_args()
    if args.config:
        run_on_config(args.config)
        return

    output_path = run_on_image(args.img)
    print(f"Saved no bg image to {output_path}")


if __name__ == "__main__":
    main()
