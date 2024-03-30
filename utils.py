import os
import argparse
# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion Server')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Model name for DiffusionPipeline')
    parser.add_argument('--unet', type=str, default='', help='UNet model name')
    parser.add_argument('--lora_dirs', type=str, default='', help='Colon-separated list of LoRA directories')
    parser.add_argument('--lora_scales', type=str, default='', help='Colon-separated list of LoRA scales')
    parser.add_argument('--scheduler', type=str, default='euler_a', help='Scheduler')
    return parser.parse_args()

def is_local_file(path):
    return os.path.isfile(path)

class Args:
    def __init__(self, model, unet, lora_dirs, lora_scales, scheduler):
        self.model = model
        self.unet = unet
        self.lora_dirs = lora_dirs
        self.lora_scales = lora_scales
        self.scheduler = scheduler
