import os
from flask import Flask, request, jsonify
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
import torch
from PIL import Image, ImageOps
import io
import base64
import logging

from utils import parse_args, Args, is_local_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")

if not is_gunicorn:
    args = parse_args()
else:
    args = Args(
        model=os.getenv('MODEL_NAME', 'stabilityai/stable-diffusion-xl-base-1.0'),
        unet=os.getenv('UNET_MODEL', ''),
        lora_dirs=os.getenv('LORA_DIRS', ''),
        lora_scales=os.getenv('LORA_SCALES', ''),
        scheduler=os.getenv('SCHEDULER', 'euler_a'),
    )


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16)

def load_models():
    print("Loading models...")
    if args.unet == '':
        if is_local_file(args.model):
            pipe = StableDiffusionXLPipeline.from_single_file(args.model, vae=vae, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True)
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        else:    
            pipe = StableDiffusionXLPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16, variant="fp16")
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    else:
        unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.bfloat16, variant="fp16")
        if is_local_file(args.model):
            pipe = StableDiffusionXLPipeline.from_single_file(args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True)
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16")
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)

    lora_dirs = args.lora_dirs.split(':') if args.lora_dirs else []
    lora_scales = [float(scale) for scale in args.lora_scales.split(':')] if args.lora_scales else []

    if len(lora_dirs) != len(lora_scales):
        raise ValueError("The number of LoRA directories must match the number of scales")

    for ldir, lsc in zip(lora_dirs, lora_scales):
        pipe.load_lora_weights(ldir)
        pipe.fuse_lora(lora_scale=lsc)
        img2img_pipe.load_lora_weights(ldir)
        img2img_pipe.fuse_lora(lora_scale=lsc)

    if args.scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        img2img_pipe.scheduler = EulerDiscreteScheduler.from_config(img2img_pipe.scheduler.config)
    elif args.scheduler == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        img2img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(img2img_pipe.scheduler.config)

    
    pipe.to("cuda")
    img2img_pipe.to("cuda")

    print("Models loaded")
    return pipe, img2img_pipe

pipe, img2img_pipe = load_models()

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", None)
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", None)
        image_format = data.get("format", "jpeg").lower()

        original_width = data.get("width", 1024)
        original_height = data.get("height", 1024)

        width = ((original_width + 7) // 8) * 8
        height = ((original_height + 7) // 8) * 8

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        if image_format not in ["jpeg", "png"]:
            return jsonify({"error": "Invalid image format. Choose 'jpeg' or 'png'."}), 400

        image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]

        if (width != original_width) or (height != original_height):
            left = (width - original_width) // 2
            top = (height - original_height) // 2
            right = left + original_width
            bottom = top + original_height

            image = image.crop((left, top, right, bottom))

        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"
        data_uri = "data:" + mime_type + ";base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"image": data_uri})

    except Exception as e:
        logger.exception("Error generating image")
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate-img2img', methods=['POST'])
def generate_img2img():
    try:
        data = request.json
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", None)
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", None)
        image_format = data.get("format", "jpeg").lower()
        strength = data.get("strength", 0.8)
        extract_mask = data.get("extract_mask", False)
        extract_color = data.get("extract_color", (0, 0, 0, 0))

        original_width = data.get("width", 1024)
        original_height = data.get("height", 1024)

        width = ((original_width + 7) // 8) * 8
        height = ((original_height + 7) // 8) * 8

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        if image_format not in ["jpeg", "png"]:
            return jsonify({"error": "Invalid image format. Choose 'jpeg' or 'png'."}), 400

        images_data = data.get("images", [])
        masks_data = data.get("masks", [])

        if not images_data:
            return jsonify({"error": "At least one image is required for img2img."}), 400

        def process_image_data(image_data):
            if isinstance(image_data, list):
                return [process_image_data(img) for img in image_data]
            elif isinstance(image_data, dict):
                image = base64.b64decode(image_data["image"].split(",")[1])
                image = Image.open(io.BytesIO(image)).convert("RGBA")
                return {
                    "x": image_data.get("x", 0),
                    "y": image_data.get("y", 0),
                    "sx": image_data.get("sx", 1),
                    "sy": image_data.get("sy", 1),
                    "image": image
                }
            else:
                image = base64.b64decode(image_data.split(",")[1])
                return Image.open(io.BytesIO(image)).convert("RGBA")

        images = process_image_data(images_data)
        masks = process_image_data(masks_data) if masks_data else None

        def compose_images(images, width, height):
            composite_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for image_data in images:
                if isinstance(image_data, dict):
                    image = image_data["image"]
                    x = image_data["x"]
                    y = image_data["y"]
                    sx = image_data["sx"]
                    sy = image_data["sy"]
                    image = image.resize((int(image.width * sx), int(image.height * sy)))
                    composite_image.paste(image, (x, y), mask=image)
                else:
                    composite_image.paste(image_data, (0, 0))
            return composite_image

        composite_image = compose_images(images, width, height)
        composite_mask = compose_images(masks, width, height) if masks else None
        composite_image = composite_image.convert("L") if composite_mask is not None else composite_image

        # Generate the image using the composite image and mask
        generated_image = img2img_pipe(prompt, negative_prompt=negative_prompt, image=composite_image, mask_image=composite_mask, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]

        if extract_mask and composite_mask is not None:
            # Extract the generated content using the mask
            extracted_image = Image.composite(generated_image, Image.new("RGBA", generated_image.size, extract_color), composite_mask)
        else:
            extracted_image = generated_image

        if (width != original_width) or (height != original_height):
            left = (width - original_width) // 2
            top = (height - original_height) // 2
            right = left + original_width
            bottom = top + original_height

            extracted_image = extracted_image.crop((left, top, right, bottom))

        buffer = io.BytesIO()
        extracted_image.save(buffer, format=image_format)
        mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"
        data_uri = "data:" + mime_type + ";base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"image": data_uri})

    except Exception as e:
        logger.exception("Error generating img2img")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=3101)