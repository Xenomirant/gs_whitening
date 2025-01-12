import os
import tqdm
import argparse

import torch
import torch.backends.cuda
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers

from gs_attention_processor import GSOFTCrossAttnProcessor, DoubleGSOFTCrossAttnProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the exp folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save generated images"
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=10,
        help="Number of generated images for each prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--gsoft_nblocks",
        required=False,
        type=int,
        help="The number of GSORT blocks",
    )
    parser.add_argument(
        "--gsoft_scale",
        action='store_true',
        default=True,
    )
    parser.add_argument(
        "--gsoft_method",
        type=str,
        default='cayley',
    )
    parser.add_argument(
        "--double_gsoft",
        action="store_true",
        default=False,
        help="Whether to use Double GSOFT",
    )
    return parser.parse_args()


def main(args):
    checkpoint_path = os.path.join(args.input_dir, f'checkpoint-{args.checkpoint_idx}')

    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    attention_processor = DoubleGSOFTCrossAttnProcessor if args.double_gsoft else GSOFTCrossAttnProcessor
    gsoft_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        gsoft_attn_procs[name] = attention_processor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=args.gsoft_nblocks,
            method=args.gsoft_method, scale=args.gsoft_scale
        )

    unet.set_attn_processor(gsoft_attn_procs)
    gsoft_layers = AttnProcsLayers(unet.attn_processors)
    gsoft_layers.load_state_dict(load_file(os.path.join(checkpoint_path, 'pytorch_lora_weights.safetensors')))

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        scheduler=scheduler,
        torch_dtype=torch.float32).to(args.device)
    pipe.safety_checker = None

    prompt_set = args.prompts.split('#')

    for prompt in tqdm.tqdm(prompt_set):
        generator = torch.Generator(device=args.device)
        generator = generator.manual_seed(args.seed)
        samples_path = os.path.join(args.output_dir, 'samples', prompt)
        n_batches = (args.num_images_per_prompt - 1) // args.batch_size + 1
        images = []
        for i in range(n_batches):
            images_batch = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, num_images_per_prompt=args.batch_size,
                generator=generator
            ).images
            images += images_batch

        os.makedirs(samples_path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(samples_path, f'{idx}.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
