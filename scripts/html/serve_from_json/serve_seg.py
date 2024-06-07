
from flask import Flask, send_from_directory, render_template
import json
import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
if True:
    from src.util.timing import get_last_modified_timestamp
app = Flask(__name__)
cur_dir = os.path.dirname(os.path.abspath(__file__))

argparser = argparse.ArgumentParser()
default_json = "/home/miatang/projects/stroke-label/experiments/json_for_display/scene_no_occlude.json"
argparser.add_argument("--json_path", type=str, default=default_json)
argparser.add_argument("--port", type=int, default=8033)
args = argparser.parse_args()
json_path = os.path.abspath(args.json_path)
timestamp = get_last_modified_timestamp(json_path)
print(f"Using JSON file: {json_path} (last modified: {timestamp})")


@app.route("/")
def index():
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Generate HTML content
    html_output = generate_html_from_json(json_data)
    return f"<html><body>{html_output}</body></html>"


# Route to serve images
@app.route("/images/<path:filename>")
def serve_image(filename):
    filename = f"/{filename}"  # add leading slash
    print(f"Requested image: {filename}")
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


def generate_html_from_json(json_data):

    html_content = ""
    num_cols = 5
    json_content = f"""<h2>Serving {json_path}</h2>"""
    json_content += f"""<p>JSON Last modified: {timestamp}</p>"""
    html_content += json_content
    html_content += """<div class="py-4">"""
    img_width = 250
    for item in json_data:
        image_path = item["sketch_path"]

        image_name = os.path.basename(image_path).split(".")[0]
        output_image_paths = item["gen_paths"]
        seg_on_gen_paths = item["all_img_seg_path"]
        seg_stroke_paths = item["all_sketch_seg_path"]
        # seg_on_sketch_paths = item["seg_on_sketch_paths"]
        # seg_stroke_sketch_paths = item["seg_stroke_sketch_paths"]
        clean_gen_ann_paths = item["all_clean_ann_path"]
        # gen_gap_mask_paths = item["gen_gap_mask_paths"]
        enhanced_strokes_paths = item["all_enhanced_strokes_path"]
        enhanced_sam_paths = item["all_enhanced_sam_path"]
        all_kept_masks_paths = item["all_kept_masks_paths"]
        # seg_on_sketch_path = seg_on_sketch_paths[0]
        # seg_stroke_sketch_path = seg_stroke_sketch_paths[0]
        caption = item["caption"]

        all_gen_samples_str = ""
        all_gen_seg_str = ""
        all_clean_gen_ann_str = ""
        all_gen_gap_mask_str = ""
        all_enhanced_strokes_str = ""
        all_enhanced_sam_str = ""
        all_kept_masks_str = ""
        for seg_i in range(len(seg_on_gen_paths)):

            generated_str = f"""
              <figure class="flex">
                <div>
                    <caption> Gen Sample </caption>
                    <img class="border-2" src="/images/{output_image_paths[seg_i]}" alt="{caption}" style="max-width: {img_width//2}px;"/>
                </div>
                <div>
                    <caption> Naive Seg Strokes</caption>
                    <img class="border-2" src="/images/{seg_stroke_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width//2}px;"/>
                </div>
              </figure>
            """
            segmented_gen_str = f"""
               <figure>
                    <img class="border-2" src="/images/{seg_on_gen_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            """
            clean_gen_ann_str = f"""
                <figure>
                    <img class="border-2" src="/images/{clean_gen_ann_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            """
            # gen_gap_mask_str = f"""
            #     <figure>
            #         <img class="border-2" src="/images/{gen_gap_mask_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width}px;"/>
            #     </figure>
            # """
            enhanced_strokes_str = f"""
                <figure>
                    <caption> Dilating Segmentation Result </caption>
                    <img class="border-2" src="/images/{enhanced_strokes_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            """
            enhanced_sam_str = f"""
               <figure class="flex">
                    <div>
                        <caption> Original Sam </caption>
                        <img class="border-2" src="/images/{clean_gen_ann_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width//2}px;"/>
                    </div>
                    <div>
                       <caption> Processed Sam </caption>
                        <img class="border-2" src="/images/{enhanced_sam_paths[seg_i]}" alt="{image_path}" style="max-width: {img_width//2}px;"/>
                    </div>
                </figure>
            """

            all_gen_samples_str += generated_str
            # all_gen_seg_str += segmented_gen_str
            all_clean_gen_ann_str += clean_gen_ann_str
            # all_gen_gap_mask_str += gen_gap_mask_str
            all_enhanced_strokes_str += enhanced_strokes_str
            all_enhanced_sam_str += enhanced_sam_str
            cur_mask_str = f"""
                <div class="grid grid-cols-3">
                    <div class="col-span-3">  <h2> Kept Masks </h2> </div>
                    """
            for mask_i, mask_path in enumerate(all_kept_masks_paths[seg_i]):
                mask_str = f"""
                    <figure>
                        <img class="border-2" src="/images/{mask_path}" alt="{image_path}" style="max-width: {img_width//3}px;"/>
                    </figure>
                """
                cur_mask_str += mask_str
            cur_mask_str += "</div>"
            all_kept_masks_str += cur_mask_str

        # for loop ends

        gen_content = f"""
            {all_gen_samples_str}
            {all_enhanced_sam_str}
            {all_enhanced_strokes_str}
            {all_kept_masks_str}
        """

        rep_mask_str = ""
        if "rep_masks_plot_path" in item:
            rep_mask_path = item["rep_masks_plot_path"]
            rep_mask_str = f"""
             <div class="col-span-{num_cols} flex">
                <figure>
                    <caption class="text-xl"> Rep Masks </caption>
                    <img class="border-2" src="/images/{rep_mask_path}" alt="{caption}" style="max-width: 500px;"/>
                </figure>
            </div>
            """

        div_content = f"""
        <div class="grid grid-cols-{num_cols} border-2 gap-2">
            <h2 class="text-lg" > {image_name} </h2>
            <div class="col-span-{num_cols} flex">
                <figure>
                    <caption class="text-xl"> Input Sketch </caption>
                    <img class="border-2" src="/images/{image_path}" alt="{caption}" style="max-width: {img_width//2}px;"/>
                </figure>
                  {rep_mask_str}
                <figure>
                    <caption class="text-xl"> Enhanced Segments Across Seeds </caption>
                    <img class="border-2" src="/images/{item['rep_sam_output_path']}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
                  <figure>
                    <caption class="text-xl"> Stroke Cluster Across Seeds </caption>
                    <img class="border-2" src="/images/{item['rep_stroke_output_path']}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            </div>
          
            {gen_content}
        </div>
        """
        html_content += div_content
    html_content += "</div>"

    """
      <figure>
                <caption> Seg on Sketch </caption>
                    <img class="border-2" src="/images/{seg_on_sketch_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
                <figure>
                    <caption> Seg Strokes (Sketch)  </caption>
                    <img class="border-2" src="/images/{seg_stroke_sketch_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
    
    """

    return render_template(
        "index.html",
        html_content=html_content,
        title="Segmentation on Img and Strokes",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
