#!/usr/bin/env python3
"""
Higgs Audio API服务
提供HTTP接口用于音频生成
"""

import os
import sys
import json
import uuid
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file

# 路径配置
BASE_DIR = Path(__file__).parent.absolute()
HIGGS_DIR = BASE_DIR / "higgs-audio"
MODEL_PATH = BASE_DIR / "model" / "higgs-v2-base"
CONDA_ENV_PATH = HIGGS_DIR / "conda_env"

# 输出音频文件夹
OUTPUT_AUDIO_DIR = BASE_DIR / "generated_audio"
OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)

# 临时文件夹，用于存储上传的参考音频
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Tokenizer路径（自动检测）
TOKENIZER_PATH = "/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec"

app = Flask(__name__)
# 设置上传文件大小限制（50MB）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def run_generation(text, output_path, ref_audio=None, temperature=1.0, top_p=0.95, scene_prompt=None, max_new_tokens=1024):
    """运行音频生成"""
    
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
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 生成成功！")
            print(f"🎵 音频文件已保存到: {output_path}")
            return True, "生成成功"
        else:
            print(f"❌ 生成失败，错误码: {result.returncode}")
            print(f"错误信息: {result.stderr}")
            return False, f"生成失败: {result.stderr}"
            
    finally:
        os.chdir(original_dir)

@app.route('/generate', methods=['POST'])
def generate_audio():
    """音频生成接口"""
    try:
        # 检查是否是form-data请求
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # 处理form-data请求
            text = request.form.get('text', '').strip()
        else:
            # 处理JSON请求
            data = request.get_json()
            if not data:
                return jsonify({"error": "缺少请求数据"}), 400
            text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "缺少文本内容"}), 400
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        output_filename = f"audio_{file_id}.wav"
        output_path = OUTPUT_AUDIO_DIR / output_filename
        
        # 获取参数（支持form-data和JSON两种方式）
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # form-data方式
            try:
                temperature = float(request.form.get('temperature', 1.0))
            except ValueError:
                temperature = 1.0
                
            try:
                top_p = float(request.form.get('top_p', 0.95))
            except ValueError:
                top_p = 0.95
                
            try:
                max_new_tokens = int(request.form.get('max_new_tokens', 1024))
            except ValueError:
                max_new_tokens = 1024
                
            scene_prompt_path = request.form.get('scene_prompt', None)
            scene_prompt_content = request.form.get('scene_prompt_content', None)
        else:
            # JSON方式
            data = request.get_json()
            temperature = float(data.get('temperature', 1.0)) if 'temperature' in data else 1.0
            top_p = float(data.get('top_p', 0.95)) if 'top_p' in data else 0.95
            max_new_tokens = int(data.get('max_new_tokens', 1024)) if 'max_new_tokens' in data else 1024
            scene_prompt_path = data.get('scene_prompt', None)
            scene_prompt_content = data.get('scene_prompt_content', None)
        
        # 参数范围限制
        temperature = max(0.1, min(2.0, temperature))
        top_p = max(0.1, min(1.0, top_p))
        max_new_tokens = max(128, min(4096, max_new_tokens))
        
        # 处理参考音频文件
        ref_audio_path = None
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # form-data方式上传的参考音频
            if 'ref_audio' in request.files:
                ref_audio_file = request.files['ref_audio']
                if ref_audio_file and ref_audio_file.filename:
                    # 保存上传的音频文件到临时目录
                    ref_audio_filename = f"ref_{file_id}_{ref_audio_file.filename}"
                    ref_audio_path = TEMP_DIR / ref_audio_filename
                    ref_audio_file.save(ref_audio_path)
                    print(f"💾 保存上传的参考音频: {ref_audio_path}")
            
            # 如果没有上传文件，检查是否提供了路径
            if not ref_audio_path:
                ref_audio_path = request.form.get('ref_audio', None)
        else:
            # JSON方式提供的参考音频路径
            data = request.get_json()
            ref_audio_path = data.get('ref_audio', None)
        
        # 处理场景提示
        final_scene_prompt = None
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # form-data方式处理场景提示
            # 优先使用上传的场景提示文件
            if 'scene_prompt_file' in request.files:
                scene_prompt_file = request.files['scene_prompt_file']
                if scene_prompt_file and scene_prompt_file.filename:
                    # 保存上传的场景提示文件到临时目录
                    scene_prompt_filename = f"scene_{file_id}_{scene_prompt_file.filename}"
                    final_scene_prompt = TEMP_DIR / scene_prompt_filename
                    scene_prompt_file.save(final_scene_prompt)
                    print(f"💾 保存上传的场景提示文件: {final_scene_prompt}")
            # 其次使用直接输入的场景提示内容
            elif scene_prompt_content:
                # 创建临时文件保存场景提示内容
                scene_prompt_filename = f"scene_{file_id}.txt"
                final_scene_prompt = TEMP_DIR / scene_prompt_filename
                with open(final_scene_prompt, 'w', encoding='utf-8') as f:
                    f.write(scene_prompt_content)
                print(f"💾 保存场景提示内容: {final_scene_prompt}")
            # 最后使用提供的路径
            elif scene_prompt_path:
                final_scene_prompt = scene_prompt_path
        else:
            # JSON方式处理场景提示
            data = request.get_json()
            if scene_prompt_content:
                # 创建临时文件保存场景提示内容
                scene_prompt_filename = f"scene_{file_id}.txt"
                final_scene_prompt = TEMP_DIR / scene_prompt_filename
                with open(final_scene_prompt, 'w', encoding='utf-8') as f:
                    f.write(scene_prompt_content)
                print(f"💾 保存场景提示内容: {final_scene_prompt}")
            elif scene_prompt_path:
                final_scene_prompt = scene_prompt_path
        
        # 执行生成
        success, message = run_generation(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio_path,
            temperature=temperature,
            top_p=top_p,
            scene_prompt=final_scene_prompt,
            max_new_tokens=max_new_tokens
        )
        
        # 清理上传的参考音频文件
        if ref_audio_path and os.path.exists(ref_audio_path) and str(TEMP_DIR) in str(ref_audio_path):
            try:
                os.remove(ref_audio_path)
                print(f"🧹 清理临时文件: {ref_audio_path}")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
        
        # 清理上传的场景提示文件
        if final_scene_prompt and os.path.exists(final_scene_prompt) and str(TEMP_DIR) in str(final_scene_prompt) and final_scene_prompt != ref_audio_path:
            try:
                os.remove(final_scene_prompt)
                print(f"🧹 清理临时文件: {final_scene_prompt}")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
        
        if success:
            # 构造完整的URL而不是相对路径
            host = request.host_url.rstrip('/')
            return jsonify({
                "status": "success",
                "message": "音频生成成功",
                "file_id": file_id,
                "filename": output_filename,
                "download_url": f"{host}/audio/{output_filename}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": message
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {str(e)}"
        }), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """获取生成的音频文件"""
    try:
        file_path = OUTPUT_AUDIO_DIR / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "文件不存在"}), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取文件时出错: {str(e)}"
        }), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higgs Audio API服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5902, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    print("🎵 Higgs Audio API服务")
    print("=" * 50)
    print(f"📍 监听地址: http://{args.host}:{args.port}")
    print(f"📂 音频输出目录: {OUTPUT_AUDIO_DIR}")
    print(f"📂 临时文件目录: {TEMP_DIR}")
    print(f"🔧 模型路径: {MODEL_PATH}")
    print("🚀 服务已启动...")
    
    app.run(host=args.host, port=args.port, debug=args.debug)