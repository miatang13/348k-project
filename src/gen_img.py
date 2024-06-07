import torch
import argparse
import os
import json
from slugify import slugify
from util.timing import get_timestamp
from util.img import load_sqr_image, save_image_to_path, invert_sketch, img_res

cur_dir = os.path.dirname(os.path.abspath(__file__))
NUM_SAMPLES = 8


def setup_pipe_and_pidinet():
    print(f"Setting up pipe and pidinet")
    from diffusers import (
        StableDiffusionXLAdapterPipeline,
        T2IAdapter,
        EulerAncestralDiscreteScheduler,
        AutoencoderKL,
    )
    from controlnet_aux.pidi import PidiNetDetector
    # load adapter
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        torch_dtype=torch.float16,
        varient="fp16",
    ).to("cuda")

    # load euler_a scheduler
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id,
        vae=vae,
        adapter=adapter,
        scheduler=euler_a,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    # pipe.enable_xformers_memory_efficient_attention()

    pidinet = PidiNetDetector.from_pretrained(
        "lllyasviel/Annotators").to("cuda")
    return pipe, pidinet


def process_sketch_with_pidinet(image_path, pidinet):
    # make sure to check this processed output (took out bear eyes for example)
    image = load_sqr_image(image_path)
    image = pidinet(
        image, detect_resolution=img_res, image_resolution=img_re, apply_filter=True
    )
    return image


def process_sketch_naive(image_path):
    image = load_sqr_image(image_path)
    return invert_sketch(image)


def run_pipe_on_sketch(pipe, sketch_image, prompt, num_images_per_prompt=NUM_SAMPLES):
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
    seed = torch.randint(0, 1000000, (1,)).item()
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sketch_image,
        num_inference_steps=30,
        adapter_conditioning_scale=0.9,
        guidance_scale=7.5,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )

    return output


def run_sketch_to_images(pipe, processed_image, prompt, output_dir, num_images_per_prompt=NUM_SAMPLES):
    output = run_pipe_on_sketch(
        pipe, processed_image, prompt, num_images_per_prompt)
    output_paths = []
    for idx, image in enumerate(output.images):
        image_name = f"{slugify(prompt)}-{idx}.png"
        image_path = os.path.join(output_dir, image_name)
        save_image_to_path(image, image_path)
        output_paths.append(image_path)
    return output_paths


def run_sketch_path_to_single_image_path(sketch_path, prompt, output_dir):
    pipe, _ = setup_pipe_and_pidinet()
    processed_image = process_sketch_naive(sketch_path)
    paths = run_sketch_to_images(
        pipe, processed_image, prompt, output_dir, num_images_per_prompt=1)
    return paths[0]


def run_sketch_path_to_images(sketch_path, prompt, output_dir):
    pipe, pidinet = setup_pipe_and_pidinet()
    processed_image = process_sketch_naive(sketch_path)
    output_paths = run_sketch_to_images(
        pipe, processed_image, prompt, output_dir)
    return output_paths


def run_on_config(args, pipe):
    print(f"Using config JSON: {args.config}")
    config_data = json.load(open(args.config))
    res_data = []  # store results paths

    for item_idx in range(len(config_data)):
        item = config_data[item_idx]
        image_path = item["sketch_path"]
        image_name = os.path.basename(image_path).split(".")[0]
        image_dir = os.path.dirname(image_path).split("/")[-1]
        if args.sketch_name and args.sketch_name not in image_path:
            continue
        output_subdir = os.path.join(args.output_dir, image_dir, image_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)
            print(f"Created output subdirectory: {output_subdir}")

        if args.uniform_prompt:
            caption = "A realistic image"
        else:
            caption = item["caption"]

        print(f"Processing image {image_path} with caption: {caption}")
        processed_image = process_sketch_naive(image_path)
        processed_img_path = os.path.join(
            output_subdir, "processed_sketch", f"{image_name}_processed.png"
        )
        save_image_to_path(processed_image, processed_img_path)

        output_paths = run_sketch_to_images(
            pipe, processed_image, caption, output_subdir
        )
        res_obj = {
            "sketch_path": image_path,
            "processed_sketch_path": processed_img_path,
            "caption": caption,
            "gen_paths": output_paths,
            "timestamp": get_timestamp(),
            "style": image_dir,
        }
        res_data.append(res_obj)

    # save results to json
    input_config_name = os.path.basename(args.config).split(".")[0]
    output_json = os.path.join(
        args.output_dir, f"{input_config_name}_gen_results.json")
    # if already exists, we update the file
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            existing_data = json.load(f)
        # look for duped image paths
        existing_data = [
            x for x in existing_data if x["sketch_path"] not in res_data]
        res_data = existing_data + res_data

    with open(output_json, "w") as f:
        json.dump(res_data, f, indent=4)
    print(f"Saved results to {output_json}")


def main():
    argparser = argparse.ArgumentParser()
    test_config = "/home/miatang/projects/stroke-label/experiments/captioning/test.json"
    default_output_dir = os.path.join(cur_dir, "../experiments/sketch_2_img")
    argparser.add_argument("--config", type=str, default=test_config)
    argparser.add_argument("--output_dir", type=str,
                           default=default_output_dir)
    argparser.add_argument("--sketch_name", type=str, default=None)
    argparser.add_argument(
        "--uniform_prompt", action="store_true", default=False)
    argparser.add_argument("--all", action="store_true", default=False)

    # for debugging
    argparser.add_argument("--sketch_path", type=str, default=None)
    argparser.add_argument("--caption", type=str, default=None)
    args = argparser.parse_args()

    if args.sketch_path and args.caption:
        pipe, pidinet = setup_pipe_and_pidinet()
        processed_image = process_sketch_naive(args.sketch_path)
        output_paths = run_sketch_to_images(
            pipe, processed_image, args.caption, args.output_dir)
        return output_paths

    if args.uniform_prompt:
        args.output_dir = f"{args.output_dir}_uniform_prompt"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    if args.all:
        args.config = test_config.replace("test", "all_captions")

    """
    Run pipeline
    """
    pipe, pidinet = setup_pipe_and_pidinet()
    run_on_config(args, pipe)


if __name__ == "__main__":
    main()
