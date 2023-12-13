# Simple Diffusion Server

## Introduction
`simple-diffusion-server` is a lightweight Flask-based server designed to generate images using the Stable Diffusion model. It offers a simple API endpoint to receive image generation requests and supports custom UNet models and LoRA adjustments.

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda environment (recommended)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/x0rb0t/simple-diffusion-server.git
   cd simple-diffusion-server
   ```

2. **Create and activate a Conda environment (optional but recommended):**
   ```bash
   conda create -n diffusion-server python=3.9
   conda activate diffusion-server
   ```

3. **Install dependencies:**
   ```bash
   pip install flask diffusers torch pillow
   ```

   If using Gunicorn for production:
   ```bash
   pip install gunicorn
   ```

4. **Environment Variables:**
   Optionally, set the environment variables for the default model, UNet model, LoRA directories, and scales:
   ```bash
   export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
   export UNET_MODEL="latent-consistency/lcm-sdxl" # Or use can use "default" to use default unet of your model
   export LORA_DIRS="lora1.safetensors:lora2.safetensors"
   export LORA_SCALES="0.7:0.7"
   ```

## Usage

### Starting the Server
Run the server using Flask (for development):
```bash
python server.py
```

Or using Gunicorn (for production):
```bash
gunicorn -b 0.0.0.0:3101 server:app
```

### Making Requests
To generate an image, send a POST request to the `/generate-image` endpoint with the desired parameters. Example using `curl`:

```bash
curl -X POST http://localhost:3101/generate-image \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "a surreal landscape, digital art",
         "num_inference_steps": 4,
         "guidance_scale": 1,
         "width": 1024,
         "height": 1024,
         "format": "jpeg"
         }'
```

Note: We are using latent-consistency/lcm-sdxl, so num_inference_steps is small

### Command-Line Arguments
You can also start the server with specific model configurations using command-line arguments:

```bash
python server.py --model "your_model_name" --unet "your_unet_model" --lora_dirs "dir1:dir2" --lora_scales "0.7:0.8"
```

## Additional Notes
- Ensure that the models and LoRA directories you specify are compatible with the `diffusers` library.
- For production deployment, consider using a reverse proxy like Nginx and setting up HTTPS.
