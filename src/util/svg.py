from svgpathtools import svg2paths2, wsvg
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse


def set_paths_attributes_for_highlight(paths, attributes, path_idx):
    HIGHLIGHT_COLOR = "red"
    NONE_HIGHLIGHT_COLOR = "black"
    for i, path in enumerate(paths):
        if i == path_idx:
            attributes[i]["stroke"] = HIGHLIGHT_COLOR
            attributes[i]["fill"] = HIGHLIGHT_COLOR
        else:
            attributes[i]["stroke"] = NONE_HIGHLIGHT_COLOR
            attributes[i]["fill"] = NONE_HIGHLIGHT_COLOR
    return attributes


def color_paths_one_by_one(svg_path, output_dir):
    paths, attributes, svg_attributes = load_svg(svg_path)
    for i, path in enumerate(paths):
        colored_attributes = set_paths_attributes_for_highlight(paths, attributes, i)
        output_path = os.path.join(output_dir, f"{os.path.basename(svg_path)}-p{i}.svg")
        save_svg(paths, colored_attributes, svg_attributes, output_path)
    print(f"Saved {len(paths)} SVGs to {output_dir}")


def color_specific_path(svg_path, path_idx, output_dir):
    paths, attributes, svg_attributes = load_svg(svg_path)
    colored_attributes = set_paths_attributes_for_highlight(paths, attributes, path_idx)
    output_path = os.path.join(
        output_dir, f"{os.path.basename(svg_path)}-p{path_idx}.svg"
    )
    save_svg(paths, colored_attributes, svg_attributes, output_path)


def plot_polyline(polyline):
    # from a CubicBezier object we can get .poly() which returns a numpy poly1d object
    # polyline is of type poly1d (numpy)
    x = np.linspace(0, 1, 100)
    y = polyline(x)
    plt.plot(x, y)
    plt.savefig("/home/miatang/projects/stroke-label/data/svgs/polyline.png")


def save_svg(paths, attributes, svg_attributes, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        print(f"Created directory: {os.path.dirname(output_path)}")

    wsvg(
        paths,
        attributes=attributes,
        svg_attributes=svg_attributes,
        filename=output_path,
    )


def load_svg(svg_path):
    # https://pypi.org/project/svgpathtools/
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    return paths, attributes, svg_attributes


def path_obj_to_points(path_obj, num_samples=100):
    points = []
    for i in range(num_samples):
        t = i / num_samples
        point = path_obj.point(t)
        point = np.array([point.real, point.imag])
        points.append(point)
    return points


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    test_svg = "/home/miatang/projects/stroke-label/data/images/sketch/baby_penguin/penguin_style2.svg"
    experiment_dir = "/home/miatang/projects/stroke-label/experiments/svgs"
    argparser.add_argument("--svg", type=str, default=test_svg)
    argparser.add_argument("--output_dir", type=str, default=experiment_dir)
    args = argparser.parse_args()

    paths, attributes, svg_attributes = load_svg(args.svg)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    svg_name = os.path.basename(args.svg).split(".")[0]
    print(f"Processing {svg_name}")
    output_subdir = os.path.join(experiment_dir, f"{svg_name}-highlight")
    if os.path.exists(output_subdir):
        # remove files
        for file in os.listdir(output_subdir):
            os.remove(os.path.join(output_subdir, file))
    color_paths_one_by_one(args.svg, output_subdir)
