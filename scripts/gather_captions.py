"""'
Util script to gather all the captions from the dataset and save them in a single file.
"""

import os
import json
import glob


def main():
    res_data = []
    captioning_dir = "/home/miatang/projects/stroke-label/experiments/captioning"
    caption_files = glob.glob(os.path.join(captioning_dir, "*.json"))
    print(f"Found {len(caption_files)} caption files")
    output_json_path = os.path.join(captioning_dir, "all_captions.json")
    if os.path.exists(output_json_path):
        print(f"Removing existing file: {output_json_path}")
        os.remove(output_json_path)

    for caption_file in caption_files:
        print(f"Loading captions from {caption_file}")
        with open(caption_file, "r") as f:
            json_data = json.load(f)
            res_data.extend(json_data)
    print(f"Found {len(res_data)} captions")
    # Save all captions to a single file
    with open(output_json_path, "w") as f:
        json.dump(res_data, f, indent=4)
    print(f"Saved all captions to {output_json_path}")


if __name__ == "__main__":
    main()
