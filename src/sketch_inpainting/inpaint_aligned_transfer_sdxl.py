import numpy as np
from diffusers.utils import load_image
from diffusers import DDIMScheduler, StableDiffusionXLInpaintPipeline
import torch
import sys
import os

# local moduels
sys.path.append("/home/miatang/projects/stroke-label/src/sketch_inpainting")
if True:
    from .util import prepare_mask
    from .sa_handler import Handler, StyleAlignedArgs
    from .inversion import ddim_inversion, make_inversion_inpaint_callback


def align_inpaint(ref_img_path, ref_prompt, mask_path, init_img_path, target_prompt=""):

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False)

    # https://medium.com/aibygroup/lets-understand-stable-diffusion-inpainting-fdd0b1c3a925
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        use_safetensors=True,
        variant="fp16").to("cuda")

    # DDIM inversion

    # ref_img_path = "/home/miatang/projects/stroke-label/data/sketch/two_meerkats_fur.png"
    # ref_prompt = "meerkats, quick sketch style."

    num_inference_steps = 50
    x0 = np.array(load_image(ref_img_path).resize((1024, 1024)))
    zts = ddim_inversion(
        pipeline, x0, ref_prompt, num_inference_steps, 2)

    # some parameters you can adjust to control fidelity to reference
    # higher value induces higher fidelity, set 0 for no shift
    shared_score_shift = np.log(2)
    shared_score_scale = 1.0  # higher value induces higher, set 1 for no rescale

    # target_prompt = "cute tiger, quick sketch style."
    prompts = [ref_prompt, target_prompt]

    handler = Handler(pipeline)
    sa_args = StyleAlignedArgs(
        share_group_norm=True, share_layer_norm=True, share_attention=True,
        adain_queries=True, adain_keys=True, adain_values=False,
        shared_score_shift=shared_score_shift, shared_score_scale=shared_score_scale,)
    handler.register(sa_args)

    # mask_path = "/mnt/sda/miatang/external_git/style-aligned/two_meerkats_mask_test1.png"
    print(f"Using mask from {mask_path}")
    mask_image = load_image(mask_path).resize((1024, 1024))
    mask = prepare_mask(mask=mask_image)

    height, width = mask.shape[-2:]

    mask = torch.nn.functional.interpolate(
        mask, size=(
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor
        )
    ).to(pipeline.device)
    print(f"Mask is on device: {mask.device}")

    zT, inversion_callback = make_inversion_inpaint_callback(
        zts, mask, offset=5)

    # Create random latents
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(10)

    latents = torch.randn(len(prompts), 4, 128, 128, device='cpu', generator=g_cpu,
                          dtype=pipeline.unet.dtype,).to('cuda:0')
    latents[0] = zT

    # init_img_path = "/home/miatang/projects/stroke-label/data/sketch/two_meerkats_fur.png"
    image = load_image(init_img_path).resize((1024, 1024))

    images_a = pipeline(
        prompt=prompts,
        latents=latents,
        image=image,
        callback_on_step_end=inversion_callback,
        mask_image=mask_image,
        guidance_scale=10.0,
        num_inference_steps=50,
        strength=0.99,
    ).images

    handler.remove()
    return images_a[1], mask
