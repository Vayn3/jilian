# 武汉话方言对话系统# 级联式武汉话实时语音对话系统



基于豆包ASR + 通义千问LLM + 讯飞TTS的语音对话系统，支持ROS1音频输出。基于 **豆包ASR** → **千问LLM** → **讯飞TTS（武汉话）** 的流式语音对话系统。



## 快速启动## 近期更新



### 普通模式- 新增 **双播放模式**：`AudioConfig.output_mode` 支持 `pyaudio`（本地声卡）和 `ros1`（ROS 扬声器话题发布）。

```bash- 增加 **UDP 动作控制**：ASR/LLM 关键词触发 UDP 指令（默认 5557 端口）并在播放结束自动发送麦克风接管指令（默认 5558 端口）。

conda activate jilian- 支持 **关键词检测** 配置：关键词与正则在 `audio_constants.py` 中集中管理，可开关 `SystemConfig.enable_keyword_detection`。

python main.py- 新增 **ROS 扬声器发布器**：`ros_audio.py` 通过 `audio_msgs/AudioData` 或 `std_msgs/ByteMultiArray` 发布，缺省回落到本地播放。

```

## 系统架构

### ROS1模式（推荐）

```bash```

./run_with_ros.sh┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐

```│   麦克风    │───>│  豆包ASR    │───>│  千问LLM    │───>│  讯飞TTS    │

│  音频采集   │    │  流式识别   │    │  流式生成   │    │  武汉话合成  │

> 注意：ROS1模式需要确保已安装 `pyyaml` 和 `rospkg` 包│  +回声消除  │    │             │    │  +RAG接口   │    │             │

└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

## 系统架构       │                  │                  │                  │

       v                  v                  v                  v

```  [PCM音频流]      [实时文本片段]      [逐句输出]        [PCM音频流]

┌─────────────┐    48kHz     ┌─────────────┐    16kHz     ┌─────────────┐       │                  │                  │                  │

│  麦克风      │ ──────────▶ │  重采样      │ ──────────▶ │  豆包 ASR   │       └──────> asyncio.Queue 级联 <────────┴────> 实时播放 <───┘

│ (48kHz)     │             │  (scipy)    │             │ (武汉话)     │```

└─────────────┘             └─────────────┘             └─────────────┘

                                                              │## 文件结构

                                                              ▼

┌─────────────┐    24kHz     ┌─────────────┐    16kHz     ┌─────────────┐```

│  ROS Topic  │ ◀────────── │  格式转换    │ ◀────────── │  讯飞 TTS   │jilian/

│ (f32le)     │             │  (int16→f32)│             │             │├── main.py           # 主入口

└─────────────┘             └─────────────┘             └─────────────┘├── config.py         # 全局配置（含 output_mode、UDP、ROS 配置）

        │                                                     ▲├── asr_client.py     # 豆包ASR客户端（关键词→UDP 动作）

        │                                                     │├── llm_client.py     # 千问LLM客户端 + RAG接口

        ▼                                                     │├── tts_client.py     # 讯飞TTS客户端（关键词→UDP 动作）

┌─────────────┐             ┌─────────────┐             ┌─────────────┐├── audio_manager.py  # 音频采集/播放/回声消除/ROS 发布

│  下游设备    │             │  关键词检测   │ ◀────────── │  通义千问    │├── audio_constants.py# 采样/音量/关键词配置

│             │             │  (动作触发)   │             │  LLM        │├── ros_audio.py      # ROS 扬声器发布器

└─────────────┘             └─────────────┘             └─────────────┘├── conversation.py   # 对话管理/状态机/打断处理

```└── requirements.txt  # 依赖

```

## 文件结构

## 安装

| 文件 | 功能 |

|------|------|```bash

| `main.py` | 主程序入口，VAD检测，音频重采样 |cd jilian

| `config.py` | 全局配置（ASR/LLM/TTS/Audio） |pip install -r requirements.txt

| `audio_manager.py` | 音频采集、播放、格式转换、UDP控制 |```

| `asr_client.py` | 豆包ASR WebSocket客户端 |

| `llm_client.py` | 通义千问流式对话客户端 |### PyAudio 安装（Windows）

| `tts_client.py` | 讯飞TTS WebSocket客户端 + 关键词检测 |

| `conversation.py` | 对话状态机管理 |```bash

| `ros_audio.py` | ROS1音频发布器 |pip install pipwin

| `run_with_ros.sh` | ROS1环境启动脚本 |pipwin install pyaudio



## 关键配置# 或下载对应 whl 后安装

pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl

### 音频配置 (config.py)```



```python## 使用

@dataclass

class AudioConfig:```bash

    sample_rate: int = 48000          # 麦克风采集采样率（硬件限制）python main.py

    asr_sample_rate: int = 16000      # ASR所需采样率```

    ros1_output_sample_rate: int = 24000  # ROS输出采样率

    ros1_output_format: str = "f32le"     # ROS输出格式常用参数：

```

```bash

### 音频格式转换流程# 列出音频设备

python main.py --list-devices

1. **采集**: 麦克风 48kHz int16

2. **ASR输入**: 重采样到 16kHz int16（scipy.signal.resample）# 指定输入/输出设备

3. **TTS输出**: 16kHz int16python main.py --input-device 1 --output-device 2

4. **ROS输出**: 转换到 24kHz float32（AudioFormatConverter）

# 选择更准的模型（延迟更高）

## 关键词检测python main.py --model qwen-plus



系统支持从LLM回复中检测关键词并触发动作指令。# 调整 VAD

python main.py --vad-threshold 600 --silence-ms 300

### 动作关键词

# 禁用回声消除/打断

| 关键词 | 动作ID | UDP端口 | 说明 |python main.py --no-aec --no-barge-in

|--------|--------|---------|------|

| 你好 | 1 | 5557 | 打招呼动作 |# 启用调试日志

| 再见 | 2 | 5557 | 告别动作 |python main.py --debug

| 谢谢 | 3 | 5557 | 感谢动作 |```

| 对不起 | 4 | 5557 | 道歉动作 |

| 表演 | 5 | 5557 | 表演动作 |## 关键配置

| 跳舞 | 5 | 5557 | 跳舞动作 |

| 自我介绍 | 6 | 5557 | 自我介绍动作 |- `AudioConfig.output_mode`: `pyaudio`（默认）或 `ros1`。选择 `ros1` 时需已启动 ROS1 并有 `audio_msgs/AudioData` 消息，未找到 ROS 会自动回退到 `pyaudio`。

| 早上好 | 7 | 5557 | 早安动作 |- `AudioConfig.ros_topic_name` / `ros_node_name` / `ros_queue_size`: ROS 发布主题与队列。

| 晚安 | 8 | 5557 | 晚安动作 |- `SystemConfig.udp_host` / `voice_action_port`(默认 5557) / `mic_port`(默认 5558): UDP 指令目标。

- `SystemConfig.enable_keyword_detection`: 控制 ASR/LLM 关键词触发。

### 控制台输出示例- 关键词配置：见 `audio_constants.py`，可调整触发短语。



```## UDP 动作与麦克风切换

[语音助手启动] 使用ROS1模式

[用户] 你好啊机器人- ASR/LLM 识别到关键词会向 `voice_action_port` 发送对应动作（如 wave/nod/shake/woshou/end/good、left/right/photo/end）。

>>> [动作] 检测到关键词 '你好', 发送动作ID: 1- 音频播放完成时会向 `mic_port` 发送 `send_microphone`，ASR 结束会发送 `release_microphone`，用于麦克风占用切换。

[机器人] 你好呀！很高兴见到你，有什么我可以帮助你的吗？- 如需关闭，设置 `enable_keyword_detection=False` 或修改关键词表。

[用户] 给我跳个舞吧

>>> [动作] 检测到关键词 '跳舞', 发送动作ID: 5## ROS 扬声器模式

[机器人] 好的，我来给你跳一段舞蹈吧！

```- 设置 `output_mode=ros1` 后，播放将发布到 `AudioConfig.ros_topic_name`。

- 需要 ROS1 环境与 `rospy`，主题类型优先使用 `audio_msgs/AudioData`，缺失时使用 `std_msgs/ByteMultiArray`。

## UDP协议- 若 ROS 不可用，播放自动回落到本地 `pyaudio`。



### 动作指令（端口5557）## RAG 集成（可选）

- 格式: JSON `{"action_id": <int>}`

- 目标: `127.0.0.1:5557````python

from llm_client import RAGInterface, Document

### 麦克风控制（端口5558）

- 禁用麦克风: `{"cmd": "disable_mic"}`class MyRAG(RAGInterface):

- 启用麦克风: `{"cmd": "enable_mic"}`    async def retrieve(self, query: str, top_k: int = 5):

        pass

## 依赖安装

    async def build_context(self, documents):

```bash        return "\n".join([doc.content for doc in documents])

pip install -r requirements.txt

# 使用

# ROS1模式额外依赖# system = VoiceDialogSystem()

pip install pyyaml rospkg# system.set_rag(MyRAG())

``````



## 环境变量## 常见问题



ROS1模式需要设置PYTHONPATH:- **设备不支持参数**：先 `--list-devices`，再指定正确索引。

```bash- **回声严重**：保持回声消除开启或使用耳机。

export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages- **ROS 无法播放**：确认已启动 ROS1，已安装 `rospy`，并存在目标主题类型。

```

## 参考链接

或直接使用启动脚本 `./run_with_ros.sh`

- 豆包流式语音识别：https://www.volcengine.com/docs/6561/1354869?lang=zh

## 故障排除- 通义千问大模型：https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29619931.J_C-NDPSQ8SFKWB4aef8i6I.1.74cd10d7kfmjeA&tab=api#/api/?type=model&url=2712576

- 讯飞在线语音合成：https://www.xfyun.cn/doc/tts/online_tts/API.html

### ASR未识别到有效文本
- 检查麦克风采样率是否为48000Hz
- 确认scipy已正确安装用于重采样

### 未检测到ROS1
- 确保已安装 `pyyaml` 和 `rospkg`
- 使用 `./run_with_ros.sh` 启动

### 音频播放有噪音/杂音
- 检查下游设备期望的音频格式
- 确认 `ros1_output_sample_rate` 和 `ros1_output_format` 配置正确
