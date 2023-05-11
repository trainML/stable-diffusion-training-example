import os
import argparse
import logging
import sys

from diffusers import StableDiffusionPipeline
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

model_id = os.environ.get("TRAINML_CHECKPOINT_PATH")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a custom model prompt generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A photo of sks dog in jumping over the moon",
        help="the prompt to render",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_args()
    prompt = [args.prompt] * args.n_samples
    pipe.enable_xformers_memory_efficient_attention()
    image_count = 0
    os.makedirs(f"{os.environ.get('TRAINML_OUTPUT_PATH')}", exist_ok=True)
    for i in range(args.n_iter):
        output = pipe(
            [args.prompt] * args.n_samples,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
        )
        for image in output.images:
            image.save(
                f"{os.environ.get('TRAINML_OUTPUT_PATH')}/output-{image_count:05d}.png"
            )
            image_count += 1
