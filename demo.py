import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import imageio
from trellis.pipelines import TrellisE2EInterleaveResCondPipeline
from trellis.utils import render_utils, postprocessing_utils

import pandas as pd
import numpy as np
import random
import time
import hashlib

from PIL import Image
import torch
import utils3d.torch
import cv2


ENABLE_PIPELINE_OFFLOAD = os.environ.get('TIGON_ENABLE_OFFLOAD', '1').lower() not in {'0', 'false', 'no'}

# Load pipeline
pipeline = TrellisE2EInterleaveResCondPipeline.from_pretrained("./mix_e2e_pipe")
if ENABLE_PIPELINE_OFFLOAD:
    pipeline.enable_sequential_offload()
else:
    pipeline.cuda()

def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr

def render_4_views(reps, device: torch.device):
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitches = [np.pi / 6 for _ in range(4)]

    exts, ints = [], []
    for yaw, pitch in zip(yaws, pitches):
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ], device=device).float() * 2
        fov = torch.deg2rad(torch.tensor(40, device=device))
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0], device=device).float(),
            torch.tensor([0, 0, 1], device=device).float()
        )
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        exts.append(extrinsics)
        ints.append(intrinsics)

    rendered_images = {}
    for name, representations in reps.items():
        if len(representations) == 0:
            continue
        images = []
        renderer = render_utils.get_renderer(representations[0], resolution=1024, bg_color=(1,1,1))
        for rep in representations:
            image_sample = []
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr)
                image = res['color']  # [3,H,W], 0..1
                image_sample.append(image)
            images.append(torch.stack(image_sample))   # [V,3,H,W]
        rendered_images[name] = torch.stack(images)   # [B,V,3,H,W]
    return rendered_images

def get_seed_input():
    """Get random seed from user input"""
    print("\n--- Random Seed Settings ---")
    print("Options:")
    print("1. Enter a specific number as seed (e.g., 42)")
    print("2. Enter 'r' or 'random' to use a random seed")
    print("3. Press Enter to use default seed (42)")
    
    seed_input = input("Please enter random seed: ").strip()
    
    if seed_input.lower() in ['', 'default']:
        seed = 42
        print(f"Using default seed: {seed}")
    elif seed_input.lower() in ['r', 'random']:
        seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {seed}")
    else:
        try:
            seed = int(seed_input)
            print(f"Using specified seed: {seed}")
        except ValueError:
            print("Invalid input, using default seed: 42")
            seed = 42
    
    return seed

def get_user_input():
    """Get text prompt and image paths from user"""
    print("\n=== 3D Generation Input ===")
    
    # Get seed
    seed = get_seed_input()
    
    # Get text prompt
    text_prompt = input("Enter text prompt (press Enter to skip): ").strip()
    if text_prompt.lower() == 'q':
        text_prompt = ''
    
    # Get image paths
    image_paths = []
    while True:
        image_path = input("Enter image path (input 'q' to skip, 'done' to finish): ").strip()
        
        if image_path.lower() == 'q':
            image_paths = []
            break
        elif image_path.lower() == 'done':
            break
        elif image_path and os.path.exists(image_path):
            image_paths.append(image_path)
            print(f"Added image: {image_path}")
            break
        else:
            print("Path does not exist, please re-enter")
    
    return seed, text_prompt, image_paths

def process_generation(seed, text_prompt, image_paths, output_dir="interactive_output"):
    """Process a single generation task"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input images
    cond_images = []
    if image_paths:
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                cond_images.append(img)
                print(f"Successfully loaded image: {img_path}")
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
    
    # Determine mode
    if text_prompt and cond_images:
        mode = "interleave"
        sub_dir = "interleave"
    elif text_prompt and not cond_images:
        mode = "text"
        sub_dir = "text"
    elif not text_prompt and cond_images:
        mode = "image"
        sub_dir = "image"
    else:
        print("Error: must provide text or image condition")
        return False
    
    sub_output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    
    # Unique ID
    timestamp = str(int(time.time()))
    input_hash = hashlib.md5(f"{seed}_{text_prompt}_{'_'.join(image_paths)}".encode()).hexdigest()[:8]
    unique_id = f"{timestamp}_{input_hash}"
    
    print(f"Start generation... Mode: {mode}, Seed: {seed}")
    try:
        outputs = pipeline.run(
            text_prompt,
            cond_images,
            seed=seed,
            sparse_structure_sampler_params={
                "steps": 35,
                "cfg_strength": 3,
            },
            formats=['gaussian'],
            save_cond_path=os.path.join(sub_output_dir, f"{unique_id}_ref.png")
        )
        
        # Save input info
        info_file = os.path.join(sub_output_dir, f"{unique_id}_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"ID: {unique_id}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Text prompt: {text_prompt}\n")
            f.write(f"Image paths: {', '.join(image_paths) if image_paths else 'None'}\n")
            f.write(f"Time: {time.ctime()}\n")
        
        # Render video
        if 'gaussian' in outputs and outputs['gaussian']:
            if getattr(pipeline, 'offload_enabled', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            video = render_utils.render_video(outputs['gaussian'][0], bg_color=(1,1,1))['color']
            video_path = os.path.join(sub_output_dir, f"{unique_id}_3d.mp4")
            
            height, width = video[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            for frame in video:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"✅ Generation complete! Video saved to: {video_path}")

            # Render 4 views
            reps = {'gaussian': outputs.get('gaussian', [])}
            sample_vis = render_4_views(reps, device=pipeline.execution_device)

            for rep_name, rep_images in sample_vis.items():
                B, V = rep_images.shape[0], rep_images.shape[1]
                for b in range(B):
                    for v in range(V):
                        img = rep_images[b, v].detach().cpu().numpy()
                        img = np.transpose(img, (1, 2, 0))
                        cv2.imwrite(
                            os.path.join(sub_output_dir, f"{unique_id}_{v}.png"),
                            cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
                        )
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def main():
    """Main interactive loop"""
    print("=" * 50)
    print("TRELLIS 3D Generation Interactive Tool")
    print("=" * 50)
    print("Instructions:")
    print("- Enter text prompt or image path for 3D generation")
    print("- Text: input description directly")
    print("- Image: input file path")
    print("- Seed: input number or random")
    print("- Input 'q' to skip condition")
    print("- Input 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            seed, text_prompt, image_paths = get_user_input()
            
            if text_prompt.lower() == 'quit' and not image_paths:
                print("Exiting...")
                break
            
            if text_prompt or image_paths:
                success = process_generation(seed, text_prompt, image_paths)
                if success:
                    print("🎉 Success!")
            else:
                print("⚠️ No input provided, skipping")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()