from flask import Flask, render_template, request, send_from_directory
import os
import glob
from datetime import datetime
app = Flask(__name__)

# Base directory containing subfolders with images
BASE_DIR = '/home/miatang/projects/stroke-label/experiments'


@app.route('/')
def index():
    sections = [f.name for f in os.scandir(BASE_DIR) if f.is_dir()]
    # sort by latest updated
    sections.sort(key=lambda x: os.path.getmtime(
        os.path.join(BASE_DIR, x)), reverse=True)
    sections_time = [os.path.getmtime(os.path.join(BASE_DIR, x))
                     for x in sections]
    sections_time = [
        f"{ datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')}" for x in sections_time]
    zip_sections = zip(sections, sections_time)
    return render_template('index.html', zip_sections=zip_sections)


@app.route("/images/<path:filename>")
def serve_image(filename):
    filename = f"/{filename}"  # add leading slash
    print(f"Requested image: {filename}")
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


@app.route('/display')
def display():
    selected_sections = request.args.getlist(
        'section')  # Get sections from query parameters
    images_with_captions = []

    base_caption_arr = [
        'Generated Image', 'Remove BG', 'Annotations', "Dilated SAM"]

    for selected_section in selected_sections:
        full_path = os.path.join(BASE_DIR, selected_section)
        sketch = os.path.join(full_path, 'input_sketch.png')
        sketch = f"/images/{sketch}"
        final_result = os.path.join(
            full_path, 'segment/match_segs/rep_mask_annotated.png')
        final_result = f"/images/{final_result}"
        init_consensus_masks = os.path.join(
            full_path, 'segment/match_segs/init_consensus_mask.png')
        init_consensus_masks = f"/images/{init_consensus_masks}"
        consensus_masks = os.path.join(
            full_path, 'segment/match_segs/consensus_mask.png')
        consensus_masks = f"/images/{consensus_masks}"
        final_colored_sketch = os.path.join(
            full_path, 'final_masks/final_colored_sketch.png')
        final_colored_sketch = f"/images/{final_colored_sketch}"
        # , consensus_masks]
        large_images = [sketch, final_colored_sketch, final_result,
                        init_consensus_masks, consensus_masks]
        # final_with_captions = (f"/images/{final_results}", "Final Results")

        gen_imgs_dir = os.path.join(full_path, 'gen_imgs')
        seg_dir = os.path.join(full_path, 'segment')
        gen_imgs_sorted = sorted(glob.glob(gen_imgs_dir + '/*.png'))
        num_gen = len(os.listdir(gen_imgs_dir))
        for gen_i in range(num_gen):
            gen_img = gen_imgs_sorted[gen_i]
            seg_subdir = os.path.join(seg_dir, str(gen_i))
            gen_no_bg = os.path.join(seg_subdir, 'gen_no_bg.png')
            gen_clean_annotation = os.path.join(
                seg_subdir, 'gen_clean_annotation.png')
            kept_masks = os.path.join(
                seg_subdir, 'dilated/enhanced_sam_results.png')

            images_arr = [gen_img, gen_no_bg, gen_clean_annotation, kept_masks]
            images_paths = [
                f"/images/{img}" for img in images_arr]
            captions_arr = [f"{gen_i}__{base_caption_arr[i]}" for i in range(
                len(base_caption_arr))]
            zipped_pairs = zip(images_paths, captions_arr)
            print(f"zipped_pairs: {zipped_pairs}")
            images_with_captions.extend(zipped_pairs)

        final_masks_dir = os.path.join(full_path, 'final_masks')

    return render_template('display.html', images_with_captions=images_with_captions,
                           subdir=selected_section, large_images=large_images)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8011)
