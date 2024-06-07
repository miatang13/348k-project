from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import ImageFilter
from PIL import Image


def inpaint_blur_mask(image, blur_factor=15):
    # Applies Gaussian blur to an image.
    # copied from VaeImageProcessor's blur
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_factor))
    combined_image = Image.blend(image, blurred_image, 0.5)
    return combined_image


def setup_inpaint_model():
    # pipeline = AutoPipelineForInpainting.from_pretrained(
    #     "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
    # )
    # pipeline.enable_model_cpu_offload()
    # return pipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")
    return pipe
