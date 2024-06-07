from typing import Union
import PIL
import torch
import numpy as np


def prepare_mask(
    mask: Union[PIL.Image.Image, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    # From https://github.com/Vadbeg/diffusers-inpainting/blob/3efd045e431ddfb40019809554285c5d3e62722e/pipelines/pipeline_stable_diffusion_img2img_simple_inpaint.py
    """
    Prepares a mask to be consumed by the Stable Diffusion pipeline. This means that this input will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``1`` for
    the ``mask``.

    The ``mask`` will be binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        torch.Tensor: mask as ``torch.Tensor`` with 4 dimensions: ``batch x channels x height x width``.
    """
    if isinstance(mask, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not"
            )

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
            )
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    return mask
