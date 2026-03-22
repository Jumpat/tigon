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

# 加载pipeline
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
    """获取用户输入的随机数种子"""
    print("\n--- 随机数种子设置 ---")
    print("可选选项:")
    print("1. 输入特定数字作为种子 (如: 42)")
    print("2. 输入 'r' 或 'random' 使用随机种子")
    print("3. 直接回车使用默认种子 (42)")
    
    seed_input = input("请输入随机数种子: ").strip()
    
    if seed_input.lower() in ['', 'default']:
        seed = 42
        print(f"使用默认种子: {seed}")
    elif seed_input.lower() in ['r', 'random']:
        seed = random.randint(0, 2**32 - 1)
        print(f"使用随机种子: {seed}")
    else:
        try:
            seed = int(seed_input)
            print(f"使用指定种子: {seed}")
        except ValueError:
            print("输入无效，使用默认种子: 42")
            seed = 42
    
    return seed

def get_user_input():
    """获取用户输入的文本提示和图像路径"""
    print("\n=== 3D生成参数输入 ===")
    
    # 获取随机数种子
    seed = get_seed_input()
    
    # 获取文本提示
    text_prompt = input("请输入文本提示 (直接回车跳过文本条件): ").strip()
    if text_prompt.lower() == 'q':
        text_prompt = ''
    
    # 获取图像路径
    image_paths = []
    while True:
        image_path = input("请输入图像路径 (输入'q'跳过图像条件，输入'done'结束输入): ").strip()
        
        if image_path.lower() == 'q':
            image_paths = []
            break
        elif image_path.lower() == 'done':
            break
        elif image_path and os.path.exists(image_path):
            image_paths.append(image_path)
            print(f"已添加图像: {image_path}")
            # more = input("是否继续添加图像? (y/n): ").strip().lower()
            # if more not in ['y', 'yes', '是']:
            #     break
            break
        else:
            print("路径不存在，请重新输入")
    
    return seed, text_prompt, image_paths

def process_generation(seed, text_prompt, image_paths, output_dir="interactive_output"):
    """处理单次生成任务"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入图像
    cond_images = []
    if image_paths:
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                cond_images.append(img)
                print(f"成功加载图像: {img_path}")
            except Exception as e:
                print(f"加载图像失败 {img_path}: {e}")
    
    # 确定生成模式
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
        print("错误：必须提供文本或图像条件")
        return False
    
    # 创建子目录
    sub_output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    
    # 生成唯一标识
    timestamp = str(int(time.time()))
    input_hash = hashlib.md5(f"{seed}_{text_prompt}_{'_'.join(image_paths)}".encode()).hexdigest()[:8]
    unique_id = f"{timestamp}_{input_hash}"
    
    # 运行生成
    print(f"开始生成... 模式: {mode}, 种子: {seed}")
    try:
        outputs = pipeline.run(
            text_prompt,
            cond_images,
            seed=seed,  # 使用用户指定的种子
            sparse_structure_sampler_params={
                "steps": 35,
                "cfg_strength": 3,
            },
            formats=['gaussian'],
            save_cond_path = os.path.join(sub_output_dir, f"{unique_id}_ref.png")
        )
        
        # 保存输入信息
        info_file = os.path.join(sub_output_dir, f"{unique_id}_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"生成ID: {unique_id}\n")
            f.write(f"模式: {mode}\n")
            f.write(f"随机数种子: {seed}\n")
            f.write(f"文本提示: {text_prompt}\n")
            f.write(f"图像路径: {', '.join(image_paths) if image_paths else '无'}\n")
            f.write(f"时间: {time.ctime()}\n")
        
        # 渲染并保存结果
        if 'gaussian' in outputs and outputs['gaussian']:
            if getattr(pipeline, 'offload_enabled', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            video = render_utils.render_video(outputs['gaussian'][0], bg_color=(1,1,1))['color']
            video_path = os.path.join(sub_output_dir, f"{unique_id}_3d.mp4")
            
            # 获取视频尺寸
            height, width = video[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            # 写入每一帧
            for frame in video:
                # 如果帧是RGB格式，需要转换为BGR（OpenCV使用BGR）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # 释放视频写入器
            out.release()
            
            print(f"✅ 生成完成! 视频保存至: {video_path}")

            # 渲染4视图
            reps = {'gaussian': outputs.get('gaussian', [])}
            sample_vis = render_4_views(reps, device=pipeline.execution_device)

            for rep_name, rep_images in sample_vis.items():      # rep_images: [B,V,3,H,W]
                B, V = rep_images.shape[0], rep_images.shape[1]
                for b in range(B):
                    for v in range(V):
                        img = rep_images[b, v].detach().cpu().numpy()        # [3,H,W], 0..1
                        img = np.transpose(img, (1, 2, 0))          # [H,W,3]
                        cv2.imwrite(
                            os.path.join(sub_output_dir, f"{unique_id}_{v}.png"),
                            cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
                        )
        
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False

def main():
    """主交互循环"""
    print("=" * 50)
    print("TRELLIS 3D生成交互工具")
    print("=" * 50)
    print("使用说明:")
    print("- 输入文本提示或图像路径进行3D生成")
    print("- 文本提示: 直接输入描述文字")
    print("- 图像路径: 输入图像文件路径")
    print("- 随机数种子: 输入数字或使用随机种子")
    print("- 输入'q'跳过对应条件")
    print("- 输入'quit'退出程序")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            seed, text_prompt, image_paths = get_user_input()
            
            # 检查退出条件
            if text_prompt.lower() == 'quit' and not image_paths:
                print("退出程序...")
                break
            
            # 执行生成
            if text_prompt or image_paths:
                success = process_generation(seed, text_prompt, image_paths)
                if success:
                    print("🎉 生成成功!")
            else:
                print("⚠️  未提供任何输入条件，跳过本次生成")
            
            # 询问是否继续
            # continue_input = input("\n是否继续生成? (y/n): ").strip().lower()
            # if continue_input not in ['y', 'yes', '是']:
            #     print("感谢使用!")
            #     break
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            # continue_input = input("是否继续? (y/n): ").strip().lower()
            # if continue_input not in ['y', 'yes', '是']:
            #     break

if __name__ == "__main__":
    main()