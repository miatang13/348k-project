from flask import Flask, send_from_directory, render_template
import json
import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.util.timing import get_last_modified_timestamp
from exclude import exclude_sketch_names as exclude

app = Flask(__name__)
cur_dir = os.path.dirname(os.path.abspath(__file__))

argparser = argparse.ArgumentParser()
default_json = (
    "/home/miatang/projects/stroke-label/experiments/voting/majority_voting_log.json"
)
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
    num_cols = 3
    json_content = f"""<h2>Serving {json_path}</h2>"""
    json_content += f"""<p>JSON Last modified: {timestamp}</p>"""
    html_content += json_content
    html_content += """<div class="grid grid-cols-2 py-4">"""
    img_width = 200
    for item in json_data:
        point_logs = item["point_logs"]
        point_html = ""

        seg_samples_str = ""
        for seg_i in range(len(seg_on_gen_paths)):
            gen_path = output_image_paths[seg_i]
            seg_on_gen_path = seg_on_gen_paths[seg_i]
            seg_stroke_path = seg_stroke_paths[seg_i]
            seg_samples_str += f"""
            <div class="col-span-{num_cols} flex">
                <figure>
                    <caption> Generated Image </caption>
                    <img class="border-2" src="/images/{gen_path}" alt="{caption}" style="max-width: {img_width}px;"/>
                </figure>
                <figure>
                    <caption> Seg on Gen Image </caption>
                    <img class="border-2" src="/images/{seg_on_gen_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
                <figure>
                    <caption> Seg Strokes (Gen)  </caption>
                    <img class="border-2" src="/images/{seg_stroke_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            </div>
            """

        div_content = f"""
        <div class="grid grid-cols-{num_cols} border-4 gap-4">
            <div class="col-span-{num_cols} flex">
                <figure>
                    <caption class="text-xl"> Input Sketch {image_name}</caption>
                    <img class="border-2" src="/images/{image_path}" alt="{caption}" style="max-width: {img_width}px;"/>
                </figure>
                <figure>
                    <caption> Seg on Sketch </caption>
                    <img class="border-2" src="/images/{seg_on_sketch_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
                <figure>
                    <caption> Seg Strokes (Sketch)  </caption>
                    <img class="border-2" src="/images/{seg_stroke_sketch_path}" alt="{image_path}" style="max-width: {img_width}px;"/>
                </figure>
            </div>
            {seg_samples_str}
        </div>
        """
        html_content += div_content
    html_content += "</div>"

    return render_template(
        "index.html",
        html_content=html_content,
        title="Voting on Segmentation",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
