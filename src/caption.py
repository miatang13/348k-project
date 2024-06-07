from util.setup_llava import load_llava_model_and_processor
from PIL import Image
import argparse
import os
import glob
import json
from tqdm import tqdm
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
prompt = "USER: <image>\nWhat text prompt can you use along with the sketch to generate a high quality realistic image? Please give me a one sentence prompt in the format of 'A realistic image of ...'. Do not mention colors. ASSISTANT:"


def run_llava_on_image(model, processor, prompt, image):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    result = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"Generated caption: {result}")
    return result


def parse_llava_output(raw_output):
    """
    Example:
    USER: ... ASSITANT:{CONTENT_TO_SAVE}
    """
    parts = raw_output.split("ASSISTANT:")
    if len(parts) != 2:
        return None
    return parts[1].strip()


def caption_image(image_path, model, processor, prompt):
    # image_path = images_to_caption_paths[image_path_i]
    image = Image.open(image_path)
    raw_output = run_llava_on_image(
        model, processor, prompt=prompt, image=image)
    parsed_caption = parse_llava_output(raw_output)
    res_obj = {
        "sketch_path": image_path,
        "caption": parsed_caption,
        "raw_output": raw_output,
    }
    return res_obj


def run_on_dir(args):
    args.dir = os.path.abspath(args.dir)
    print(f"Running LLAVA on images in directory: {args.dir}")
    images_to_caption_paths = glob.glob(os.path.join(args.dir, "*.png")) + glob.glob(
        os.path.join(args.dir, "*.jpg")
    )
    print(f"Found {len(images_paths)} images in directory")
    model, processor = load_llava_model_and_processor()

    res_data = []
    for image_path_i in tqdm(range(len(images_to_caption_paths))):
        res_obj = caption_image(
            images_to_caption_paths[image_path_i], model, processor, prompt)
        res_data.append(res_obj)

    dir_name = os.path.basename(args.dir)
    output_file_name = f"batch_captions_{dir_name}.json"

    json_output_path = os.path.join(output_dir, output_file_name)
    with open(json_output_path, "w") as f:
        json.dump(res_data, f, indent=4)
    print(f"Saved captions to {json_output_path}")

    return res_data


def run_on_image(image_path):
    image_path = os.path.abspath(image_path)
    print(f"Running LLAVA on image path: {image_path}")
    model, processor = load_llava_model_and_processor()
    res_obj = caption_image(image_path, model, processor, prompt)
    return res_obj


def main():
    argparser = argparse.ArgumentParser()
    test_img_path = os.path.join(
        cur_dir, "../data/images/sketch/complex/sketch_bear_contour.png"
    )
    default_output_dir = os.path.join(cur_dir, "../experiments/captioning")

    argparser.add_argument("--image_path", type=str, default=test_img_path)
    argparser.add_argument("--dir", type=str, default=None)  # batch
    argparser.add_argument("--caption_output_dir",
                           type=str, default=default_output_dir)
    args = argparser.parse_args()
    output_dir = os.path.abspath(args.caption_output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    print(f"Args: {args}")

    """
    Load the image(s) and run LLAVA on it
    """
    if args.dir is not None:
        run_on_dir(args)
    else:
        run_on_image(args.image_path)


if __name__ == "__main__":
    main()
