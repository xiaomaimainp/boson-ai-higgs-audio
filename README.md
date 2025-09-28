# Higgs Audio 简化生成工具

这是一个超简化的 Higgs Audio 生成工具，去除了复杂的配置文件，只需要简单配置路径就能使用。

## 特点

- ✅ 自动检测模型和tokenizer路径
- ✅ 简化的命令行接口
- ✅ 一键音频生成功能
- ✅ 无需复杂配置文件
- ✅ 支持中英文文本生成
- ✅ 支持语音克隆和场景提示
- ✅ 提供HTTP API接口

## 使用方法

### 1. 命令行交互式使用

```bash
# 最简单的用法
python run_generation.py
```

交互式脚本将引导您完成以下配置：
- 输入生成文案
- 设置输出文件名
- 配置语音克隆（可选）
- 调整生成参数（温度、top-p等）
- 设置场景提示（可选）

### 2. 命令行直接使用

您也可以参考 [run_generation.py](run_generation.py) 中的代码，直接使用 Higgs Audio 官方的命令行工具：

```bash
# 切换到 higgs-audio 目录
cd higgs-audio

# 基本文本到语音
python examples/generation.py \
--transcript "Hello, welcome to Higgs Audio generation!" \
--out_path output.wav

# 使用参考音频进行语音克隆
python examples/generation.py \
--transcript "This is a test with voice cloning" \
--ref_audio broom_salesman \
--out_path cloned_voice.wav

# 调整生成参数
python examples/generation.py \
--transcript "This is a creative test" \
--temperature 1.2 \
--top_p 0.95 \
--max_new_tokens 1024 \
--out_path creative.wav
```

### 3. HTTP API 服务

项目提供了基于 Flask 的 HTTP API 服务：

```bash
# 启动 API 服务
python higgs_audio_api.py

# 默认监听地址: http://0.0.0.0:5902
```

API 接口：
- `POST /generate` - 生成音频
- `GET /audio/<filename>` - 获取生成的音频文件

使用示例：
```bash
# 使用 curl 发送请求
curl -X POST http://localhost:5902/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test",
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

## 参数说明

### 文本生成参数
- `text`: 要生成的文本内容
- `temperature`: 温度参数，控制生成的随机性 (范围: 0.1-2.0, 默认: 1.0)
- `top_p`: 控制词汇选择范围 (范围: 0.1-1.0, 默认: 0.95)
- `max_new_tokens`: 控制音频长度 (范围: 128-4096, 默认: 1024)

### 语音克隆
- `ref_audio`: 参考音频文件名或路径，用于语音克隆
- 支持格式: .wav, .mp3, .flac, .m4a, .ogg

### 场景提示
- `scene_prompt`: 场景提示文件路径，用于控制生成风格

## 文件结构

```
.
├── run_generation.py        # 交互式生成脚本
├── higgs_audio_api.py       # HTTP API服务
├── higgs_audio_api.yaml     # API接口文档
├── README.md                # 使用说明
├── higgs-audio/             # Higgs Audio 官方库
│   ├── examples/            # 官方示例
│   ├── conda_env/           # Conda环境
│   └── ...                  # 其他文件
├── model/                   # 模型文件目录
│   └── higgs-v2-base/       # Higgs Audio v2 模型
├── generated_audio/         # 生成的音频文件（自动生成）
└── temp/                    # 临时文件（自动生成）
```

## 自动路径检测

脚本会自动检测以下路径：
- 模型路径: `./model/higgs-v2-base/`
- Tokenizer路径: HuggingFace缓存目录
- Conda环境路径: `./higgs-audio/conda_env/`

## 示例

```bash
# 生成英文音频
python run_generation.py
# 然后在交互中输入: "Hello, welcome to Higgs Audio generation!"

# 生成中文音频
python run_generation.py
# 然后在交互中输入: "你好，欢迎使用Higgs音频生成工具！"

# 启动API服务
python higgs_audio_api.py --port 5902
```

生成的音频文件会保存在 `generated_audio/` 目录下。