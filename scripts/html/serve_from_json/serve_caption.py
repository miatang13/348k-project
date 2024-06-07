from flask import Flask, send_from_directory, render_template
import json
import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.util.timing import get_last_modified_timestamp

app = Flask(__name__)
cur_dir = os.path.dirname(os.path.abspath(__file__))

argparser = argparse.ArgumentParser()
default_json = (
    "/home/miatang/projects/stroke-label/experiments/captioning/all_captions.json"
)
argparser.add_argument("--json_path", type=str, default=default_json)
argparser.add_argument("--port", type=int, default=8080)
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

    json_content = f"""<h2>Serving {json_path}</h2>"""
    json_content += f"""<p>JSON Last modified: {timestamp}</p>"""
    html_content += json_content
    html_content += """<div class="grid grid-cols-2 gap-4 py-4">"""
    for item in json_data:
        image_path = item["sketch_path"]
        caption = item["caption"]
        raw_output = item["raw_output"]

        div_content = f"""
        <div class="border-2 p-2">
            <img class="border-2" src="/images/{image_path}" alt="{caption}" style="max-width: 300px;"/>
            <p><strong>Caption:</strong> {caption}</p>
            <p><strong>Raw Output:</strong> {raw_output}</p>
        </div>
        """
        html_content += div_content
    html_content += "</div>"

    return render_template(
        "index.html",
        html_content=html_content,
        title="Captioning Sketch for Sketch2Img",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
