import os
from flask import Flask, request, jsonify
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
import torch
from PIL import Image, ImageOps

import io
import base64

# Function to parse command-line arguments
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion Server')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Model name for DiffusionPipeline')
    parser.add_argument('--unet', type=str, default='', help='UNet model name')
    parser.add_argument('--lora_dirs', type=str, default='', help='Colon-separated list of LoRA directories')
    parser.add_argument('--lora_scales', type=str, default='', help='Colon-separated list of LoRA scales')
    parser.add_argument('--scheduler', type=str, default='euler_a', help='Scheduler')
    return parser.parse_args()

# Simple class to mimic argparse.Namespace behavior
class Args:
    def __init__(self, model, unet, lora_dirs, lora_scales, scheduler):
        self.model = model
        self.unet = unet
        self.lora_dirs = lora_dirs
        self.lora_scales = lora_scales
        self.scheduler = scheduler

# Check if running under Gunicorn
is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")

if not is_gunicorn:
    args = parse_args()
else:
    # Create Args from environment variables
    args = Args(
        model=os.getenv('MODEL_NAME', 'stabilityai/stable-diffusion-xl-base-1.0'),
        unet=os.getenv('UNET_MODEL', ''),
        lora_dirs=os.getenv('LORA_DIRS', ''),
        lora_scales=os.getenv('LORA_SCALES', ''),
        scheduler=os.getenv('SCHEDULER', 'euler_a'),
    )

app = Flask(__name__)

# Load the models
if args.unet == '':
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16")
else:
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float16, variant="fp16")
    pipe = DiffusionPipeline.from_pretrained(args.model, unet=unet, torch_dtype=torch.float16, variant="fp16")

# Process and load LoRA weights and scales
lora_dirs = args.lora_dirs.split(':') if args.lora_dirs else []
lora_scales = [float(scale) for scale in args.lora_scales.split(':')] if args.lora_scales else []

if len(lora_dirs) != len(lora_scales):
    raise ValueError("The number of LoRA directories must match the number of scales")

for ldir, lsc in zip(lora_dirs, lora_scales):
    pipe.load_lora_weights(ldir)
    pipe.fuse_lora(lora_scale=lsc)

if args.scheduler == "lcm":
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "euler":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "euler_a":
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to("cuda")


@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    negative_prompt = data.get("negative_prompt", None)  # New negative prompt parameter
    num_inference_steps = data.get("num_inference_steps", 4)
    guidance_scale = data.get("guidance_scale", 1.0)
    seed = data.get("seed", None)  # Retrieve seed from request; default is None
    image_format = data.get("format", "jpeg") # Default format is jpeg

    original_width = data.get("width", 1024)
    original_height = data.get("height", 1024)
    
    # Adjust width and height to the nearest multiple of 8
    width = ((original_width + 7) // 8) * 8
    height = ((original_height + 7) // 8) * 8
    
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
        
    # Validation for width and height
    if width % 8 != 0 or height % 8 != 0:
        return jsonify({"error": "Width and height must be divisible by 8"}), 400

    # Validation for image format
    if image_format not in ["jpeg", "png"]:
        return jsonify({"error": "Invalid image format. Choose 'jpeg' or 'png'."}), 400

    # Image generation
    image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    if (width != original_width) or (height != original_height):
        # Calculate cropping dimensions
        left = (width - original_width) // 2
        top = (height - original_height) // 2
        right = left + original_width
        bottom = top + original_height

        image = image.crop((left, top, right, bottom))

    # Convert image to Data URI
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"
    data_uri = "data:" + mime_type + ";base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({"image": data_uri})

if __name__ == '__main__':
    app.run(port=3101)
