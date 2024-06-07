
import glob
import os
import sys
sys.path.append("/home/miatang/projects/stroke-label/src")


def clear_dir(dir_path):
    images = glob.glob(f"{dir_path}/*.png") + \
        glob.glob(f"{dir_path}/*.jpg") + glob.glob(f"{dir_path}/*.jpeg")
    for image in images:
        os.remove(image)


def get_sketch_clean_ann_paths(sketch_name, sample_idx):
    base_path = "/home/miatang/projects/stroke-label"
    sketch_path = f"{base_path}/data/images/sketch/geometry/{sketch_name}.png"
    mask_colorful_path = f"{base_path}/experiments/segmentation/geometry/{sketch_name}/{sample_idx}/gen_clean_annotation.png"
    return sketch_path, mask_colorful_path


def get_sketch_name(sketch_path):
    return os.path.basename(sketch_path).split(".")[0]


def get_sketch_style(sketch_path):
    return os.path.dirname(sketch_path).split("/")[-1]


def get_gen_img_sample_idx(gen_img_path):
    sample_idx = gen_img_path.split(
        "/")[-1].split(".")[0].split("-")[-1]
    return sample_idx


def get_sample_subdirs(base_dir):
    # we loop over all samples by globbing subdirectories
    subdirs = glob.glob(f"{base_dir}/*")
    # we only want dirs that have a single int
    subdirs = [
        subdir
        for subdir in subdirs
        if os.path.basename(subdir).isdigit() and len(os.path.basename(subdir)) == 1
    ]
    return subdirs


def get_auto_sam_masks_paths(sample_dir):
    masks_dir = os.path.join(sample_dir, "auto_sam", "indiv_masks")
    masks_paths = glob.glob(f"{masks_dir}/*.jpg")
    return masks_paths
