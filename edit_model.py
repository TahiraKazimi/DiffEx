import torch
from diffusers import LEditsPPPipelineStableDiffusion
from diffusers.utils import load_image
from PIL import Image
from typing import Union


_LEDits_PIPE = None


def get_ledits_pipeline(device: str = "cuda"):
    global _LEDits_PIPE
    if _LEDits_PIPE is None:
        pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        else:
            print("[warn] VAE tiling not supported in this diffusers version/pipeline. Continuing without it.")
        pipe = pipe.to(device)
        _LEDits_PIPE = pipe
    return _LEDits_PIPE




@torch.no_grad()
def ledits_edit(
    image: Union[str, Image.Image],
    keyword: str,
    *,
    device: str = "cuda",
    image_size: int = 512,
    num_inversion_steps: int = 50,
    skip: float = 0.1,
    edit_guidance_scale: float = 10.0,
    edit_threshold: float = 0.75,
) -> Image.Image:
    """
    Apply Ledits++ editing to an input image using a single keyword.

    Args:
        image: PIL.Image or image path/URL
        keyword: editing keyword (e.g., "sunglasses", "newborn", "cherry blossom")
        device: cuda or cpu
        image_size: resize input to square (default 512)
        num_inversion_steps: inversion steps for Ledits++
        skip: skip ratio for inversion
        edit_guidance_scale: edit strength
        edit_threshold: mask threshold

    Returns:
        Edited PIL.Image
    """

    pipe = get_ledits_pipeline(device)

    # Load image if path/URL is given
    if isinstance(image, str):
        image = load_image(image)

    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image or a path/URL string")

    image = image.convert("RGB").resize((image_size, image_size))

    # 1) Invert the image (stores latents inside the pipeline)
    _ = pipe.invert(
        image=image,
        num_inversion_steps=num_inversion_steps,
        skip=skip,
    )

    # 2) Apply edit
    out = pipe(
        editing_prompt=[keyword],
        edit_guidance_scale=edit_guidance_scale,
        edit_threshold=edit_threshold,
    )

    return out.images[0]
# import torch

# from diffusers import LEditsPPPipelineStableDiffusion
# from diffusers.utils import load_image

# pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
#     "stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16
# )
# # Enable VAE tiling if available (diffusers version differences)
# if hasattr(pipe, "enable_vae_tiling"):
#     pipe.enable_vae_tiling()
# elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
#     pipe.vae.enable_tiling()
# else:
#     print("[warn] VAE tiling not supported in this diffusers version/pipeline. Continuing without it.")

# pipe = pipe.to("cuda")

# img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/cherry_blossom.png"
# image = load_image(img_url).resize((512, 512))

# _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.1)

# edited_image = pipe(
#     editing_prompt=["cherry blossom"], edit_guidance_scale=10.0, edit_threshold=0.75
# ).images[0]
# edited_image.save("edited.png")