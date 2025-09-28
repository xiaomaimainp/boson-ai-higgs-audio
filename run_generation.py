#!/usr/bin/env python3
"""
简单的Higgs Audio生成脚本
交互式逐步配置版本
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# 路径配置
BASE_DIR = Path(__file__).parent.absolute()
HIGGS_DIR = BASE_DIR / "higgs-audio"
MODEL_PATH = BASE_DIR / "model" / "higgs-v2-base"
CONDA_ENV_PATH = HIGGS_DIR / "conda_env"

# 输出音频文件夹
OUTPUT_AUDIO_DIR = BASE_DIR / "generated_audio"

# Tokenizer路径（自动检测）
TOKENIZER_PATH = "/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec"

def create_generation_log(text, output_path, ref_audio, temperature, top_p, max_new_tokens):
    """创建生成记录日志"""
    log_file = OUTPUT_AUDIO_DIR / "generation_log.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
=== 音频生成记录 ===
时间: {timestamp}
文案: {text}
输出文件: {output_path.name}
参考音频: {ref_audio if ref_audio else '无'}
温度参数: {temperature}
Top-p参数: {top_p}
最大Token数: {max_new_tokens}
{'='*50}
"""
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def run_generation(text, output, ref_audio=None, temperature=1.0, top_p=0.95, scene_prompt=None, max_new_tokens=1024):
    """运行音频生成"""
    
    # 创建输出音频文件夹
    OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)
    
    # 确保输出文件保存到指定文件夹
    if not os.path.isabs(output):
        output_path = OUTPUT_AUDIO_DIR / output
    else:
        output_path = Path(output)
    
    # 构建基础命令
    cmd = [
        "python", "examples/generation.py",
        "--model_path", str(MODEL_PATH),
        "--audio_tokenizer", TOKENIZER_PATH,
        "--transcript", text,
        "--out_path", str(output_path),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--device", "cuda",
        "--max_new_tokens", str(max_new_tokens)
    ]
    
    # 添加参考音频（语音克隆）
    if ref_audio and os.path.exists(ref_audio):
        # 如果是完整路径，需要复制到voice_prompts目录并创建对应的txt文件
        ref_audio_name = os.path.splitext(os.path.basename(ref_audio))[0]
        voice_prompts_dir = HIGGS_DIR / "examples" / "voice_prompts"
        voice_prompts_dir.mkdir(exist_ok=True)
        
        target_audio_path = voice_prompts_dir / f"{ref_audio_name}.wav"
        target_text_path = voice_prompts_dir / f"{ref_audio_name}.txt"
        
        # 复制音频文件
        import shutil
        shutil.copy2(ref_audio, target_audio_path)
        
        # 创建对应的文本文件（如果不存在）
        if not target_text_path.exists():
            with open(target_text_path, 'w', encoding='utf-8') as f:
                f.write("This is a reference audio for voice cloning.")
        
        cmd.extend(["--ref_audio", ref_audio_name])
        print(f"🎤 参考音频: {ref_audio} -> {ref_audio_name}")
    
    # 添加场景提示
    if scene_prompt and os.path.exists(scene_prompt):
        cmd.extend(["--scene_prompt", scene_prompt])
        print(f"🎬 场景提示: {scene_prompt}")
    
    print(f"🎵 生成音频: {text}")
    print(f"📁 输出文件: {output_path}")
    print(f"📂 输出目录: {OUTPUT_AUDIO_DIR}")
    print(f"🔧 模型路径: {MODEL_PATH}")
    print(f"🌡️ 温度参数: {temperature}")
    print(f"🎯 Top-p参数: {top_p}")
    print(f"🔢 最大Token数: {max_new_tokens}")
    print("-" * 50)
    
    # 设置环境变量
    env = os.environ.copy()
    if CONDA_ENV_PATH.exists():
        env['PATH'] = f"{CONDA_ENV_PATH}/bin:" + env['PATH']
    
    # 切换到higgs-audio目录并运行
    original_dir = os.getcwd()
    try:
        os.chdir(HIGGS_DIR)
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("✅ 生成成功！")
            print(f"🎵 音频文件已保存到: {output_path}")
            
            # 创建生成记录
            create_generation_log(text, output_path, ref_audio, temperature, top_p, max_new_tokens)
        else:
            print(f"❌ 生成失败，错误码: {result.returncode}")
            
    finally:
        os.chdir(original_dir)


def main():
    print("🎵 Higgs Audio 高级生成工具")
    print("=" * 50)
    print(f"📂 音频输出目录: {OUTPUT_AUDIO_DIR}")
    
    # 创建输出目录
    OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)
    
    # 步骤1: 输入生成文案
    print("\n📝 步骤1: 输入生成文案")
    text = input("请输入要生成的文案: ").strip()
    if not text:
        text = "Hello, this is a test audio generation."
        print(f"使用默认文案: {text}")
    
    # 步骤2: 输入输出文件名
    print("\n📁 步骤2: 设置输出文件")
    print(f"音频文件将保存到: {OUTPUT_AUDIO_DIR}")
    output = input("请输入输出文件名 (默认: output.wav): ").strip()
    if not output:
        # 使用时间戳作为默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"audio_{timestamp}.wav"
        print(f"使用默认文件名: {output}")
    
    # 确保输出文件有.wav扩展名
    if not output.endswith('.wav'):
        output += '.wav'
    
    # 步骤3: 语音克隆设置
    print("\n🎤 步骤3: 语音克隆设置")
    print("如果你有参考音频文件，可以进行语音克隆")
    ref_audio = input("请输入参考音频文件路径 (留空跳过): ").strip()
    if ref_audio:
        # 检查文件是否存在
        if not os.path.exists(ref_audio):
            print(f"⚠️  参考音频文件不存在: {ref_audio}")
            ref_audio = None
        # 检查文件是否为音频格式
        elif not ref_audio.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
            print(f"⚠️  文件不是音频格式: {ref_audio}")
            print("支持的格式: .wav, .mp3, .flac, .m4a, .ogg")
            ref_audio = None
        else:
            print(f"✅ 使用参考音频: {ref_audio}")
    
    # 步骤4: 温度参数设置
    print("\n🌡️ 步骤4: 温度参数设置")
    print("温度参数控制生成的随机性和创造性")
    print("- 较低值(0.1-0.8): 更稳定、更一致")
    print("- 中等值(0.8-1.2): 平衡的创造性")
    print("- 较高值(1.2-2.0): 更有创造性、更多样化")
    temp_input = input("温度参数 (0.1-2.0, 默认1.0): ").strip()
    try:
        temperature = float(temp_input) if temp_input else 1.0
        temperature = max(0.1, min(2.0, temperature))  # 限制范围
    except ValueError:
        temperature = 1.0
        print("输入无效，使用默认温度参数: 1.0")
    
    # 步骤5: Top-p参数设置
    print("\n🎯 步骤5: Top-p参数设置")
    print("Top-p参数控制词汇选择的范围")
    print("- 较低值(0.1-0.7): 更保守、更稳定")
    print("- 中等值(0.7-0.9): 平衡的多样性")
    print("- 较高值(0.9-1.0): 更多样化的选择")
    top_p_input = input("Top-p参数 (0.1-1.0, 默认0.95): ").strip()
    try:
        top_p = float(top_p_input) if top_p_input else 0.95
        top_p = max(0.1, min(1.0, top_p))  # 限制范围
    except ValueError:
        top_p = 0.95
        print("输入无效，使用默认Top-p参数: 0.95")
    
    # 步骤6: 最大Token数设置
    print("\n🔢 步骤6: 最大Token数设置")
    print("控制生成音频的长度，数值越大音频越长")
    print("- 512: 短音频")
    print("- 1024: 中等长度 (推荐)")
    print("- 2048: 长音频 (需要更多内存)")
    tokens_input = input("最大Token数 (默认1024): ").strip()
    try:
        max_new_tokens = int(tokens_input) if tokens_input else 1024
        max_new_tokens = max(128, max_new_tokens)  # 最小值限制
    except ValueError:
        max_new_tokens = 1024
        print("输入无效，使用默认Token数: 1024")
    
    # 步骤7: 场景提示设置
    print("\n🎬 步骤7: 场景提示设置")
    scene_prompt = input("场景提示文件路径 (留空跳过): ").strip()
    if scene_prompt and not os.path.exists(scene_prompt):
        print(f"⚠️  场景提示文件不存在: {scene_prompt}")
        scene_prompt = None
    
    # 配置确认
    print("\n📋 配置确认:")
    print("=" * 30)
    print(f"📝 生成文案: {text}")
    print(f"📁 输出文件: {output}")
    print(f"📂 保存目录: {OUTPUT_AUDIO_DIR}")
    print(f"🎤 参考音频: {ref_audio if ref_audio else '无 (使用默认音色)'}")
    print(f"🌡️ 温度参数: {temperature}")
    print(f"🎯 Top-p参数: {top_p}")
    print(f"🔢 最大Token数: {max_new_tokens}")
    print(f"🎬 场景提示: {scene_prompt if scene_prompt else '无'}")
    
    confirm = input("\n确认开始生成? (y/n, 默认y): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 已取消生成")
        sys.exit(0)
    
    print("\n🚀 开始生成...")
    run_generation(text=text, output=output, ref_audio=ref_audio, temperature=temperature, top_p=top_p, scene_prompt=scene_prompt, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    main()