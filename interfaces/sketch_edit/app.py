from flask import Flask, jsonify, send_from_directory, render_template, request
import glob
import os
import sys
import base64
from PIL import Image, ImageOps
import io

sys.path.append("/home/miatang/projects/stroke-label/src")
if True:
    from inpaint import perform_inpainting
app = Flask(__name__)


@app.route('/')
def index():
    return render_template(
        "index.html",
    )


demos_base_dir = "/home/miatang/projects/stroke-label/interfaces/sketch_edit/static/uploads"
demos = os.listdir(demos_base_dir)
demos = [demo for demo in demos if os.path.isdir(f"{demos_base_dir}/{demo}")]

demos_to_skip = ["bird_branch_simple", "joon_man", "two_camels",  "two_meerkats_fur", "plant_sofa",
                 "pears_scene", "shrimp_lemon_dish", "two_meerkats", "street_light", "yael_penguin"]
demos = [demo for demo in demos if demo not in demos_to_skip]

base_dir = "static/uploads"
glob_base_dir = "/home/miatang/projects/stroke-label/interfaces/sketch_edit/static/uploads"


@app.route('/get-demos')
def get_demos():
    # This could be dynamically generated from database or configuration
    return jsonify(demos)


@app.route('/get-images/<demo_name>')
def get_images(demo_name):
    image_urls = {}
    for sketch in demos:
        sketch_dir_path = f"{glob_base_dir}/{sketch}/stroke_layers"
        abs_path_urls = glob.glob(f"{sketch_dir_path}/*.png")
        abs_path_urls = sorted(
            abs_path_urls, key=lambda x: x.split("/")[-1])
        # reverse the order of the layers
        abs_path_urls = abs_path_urls[::-1]
        image_urls[sketch] = [url.replace(
            glob_base_dir, base_dir) for url in abs_path_urls]
        print(f"{sketch}:", image_urls[sketch], f"in {sketch_dir_path}")

    print("result:", image_urls.get(demo_name, []))
    return jsonify(image_urls.get(demo_name, []))


@app.route('/inpaint-layers', methods=['POST'])
def inpaint_layers():
    data = request.json
    layers = data['layers']
    # Assuming `perform_inpainting` is your function that handles the inpainting logic
    inpainted_path, mask_path = perform_inpainting(layers)
    assert inpainted_path is not None
    assert mask_path is not None
    inpainted_path = inpainted_path.replace(glob_base_dir, base_dir)
    mask_path = mask_path.replace(glob_base_dir, base_dir)
    return jsonify({'inpainted_path': inpainted_path, 'mask_path': mask_path})


@app.route('/save-bw-image', methods=['POST'])
def save_bw_image():
    data = request.json
    image_data = data['image']
    sketch_name = data['name']
    # Remove the "data:image/png;base64," part
    image_data = image_data.split(",")[1]

    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    image = image.convert('L')

    # Save the image
    sketch_dir = f"{demos_base_dir}/{sketch_name}"
    edit_sketch_dir = f"{sketch_dir}/edited_sketch"
    os.makedirs(edit_sketch_dir, exist_ok=True)
    image.save(f'{sketch_dir}/edited_sketch/latest.png')

    return jsonify({'message': 'Image saved successfully'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2345, debug=True)
