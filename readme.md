# 级联式武汉话实时语音对话系统

基于 **豆包ASR** → **千问LLM** → **讯飞TTS（武汉话）** 的流式语音对话系统。

## 近期更新

- 新增 **启动欢迎语**：系统启动时自动播放配置的欢迎语，可在 `config.py` 中设置 `SystemConfig.welcome_message`。
- 优化 **ROS 音频发布**：解决首帧丢失问题，发布者创建后自动预热（等待订阅者连接+发送静音帧）。
- 优化 **命令行日志输出**：精简日志，关键信息更清晰，关键词检测会详细打印匹配词和动作。
- 新增 **双播放模式**：`AudioConfig.output_mode` 支持 `pyaudio`（本地声卡）和 `ros1`（ROS 扬声器话题发布）。
- 增加 **UDP 动作控制**：ASR/LLM 关键词触发 UDP 指令（默认 5557 端口）并在播放结束自动发送麦克风接管指令（默认 5558 端口）。
- 支持 **关键词检测** 配置：关键词与正则在 `audio_constants.py` 中集中管理，可开关 `SystemConfig.enable_keyword_detection`。

## 系统架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   麦克风    │───>│  豆包ASR    │───>│  千问LLM    │───>│  讯飞TTS    │
│  音频采集   │    │  流式识别   │    │  流式生成   │    │  武汉话合成  │
│  +回声消除  │    │             │    │  +RAG接口   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
  [PCM音频流]      [实时文本片段]      [逐句输出]        [PCM音频流]
       │                  │                  │                  │
       └──────> asyncio.Queue 级联 <────────┴────> 实时播放 <───┘
```

## 启动流程

1. 系统初始化各模块（音频采集、ASR、LLM、TTS）
2. **播放欢迎语**（如配置了 `welcome_message`）
3. 进入对话循环，等待用户语音输入

## 文件结构

```
jilian/
├── main.py           # 主入口
├── config.py         # 全局配置（含欢迎语、output_mode、UDP、ROS 配置）
├── asr_client.py     # 豆包ASR客户端（关键词→UDP 动作）
├── llm_client.py     # 千问LLM客户端 + RAG接口
├── tts_client.py     # 讯飞TTS客户端（关键词→UDP 动作）
├── audio_manager.py  # 音频采集/播放/回声消除/ROS 发布
├── audio_constants.py# 采样/音量/关键词配置
├── ros_audio.py      # ROS 扬声器发布器（含预热机制）
├── conversation.py   # 对话管理/状态机/打断处理
└── requirements.txt  # 依赖
```

## 安装

```bash
cd jilian
pip install -r requirements.txt
```

### PyAudio 安装（Windows）

```bash
pip install pipwin
pipwin install pyaudio

# 或下载对应 whl 后安装
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

## 使用

```bash
python main.py
```

常用参数：

```bash
# 列出音频设备
python main.py --list-devices

# 指定输入/输出设备
python main.py --input-device 1 --output-device 2

# 选择更准的模型（延迟更高）
python main.py --model qwen-plus

# 调整 VAD
python main.py --vad-threshold 600 --silence-ms 300

# 禁用回声消除/打断
python main.py --no-aec --no-barge-in

# 启用调试日志
python main.py --debug
```

## 关键配置

### 启动欢迎语

在 `config.py` 中设置 `SystemConfig.welcome_message`：

```python
# 设置欢迎语（设为空字符串则不播放）
welcome_message: str = "大家好，我是华科机器人小科，很高兴见到你！"
```

### 其他配置

- `AudioConfig.output_mode`: `pyaudio`（默认）或 `ros1`。选择 `ros1` 时需已启动 ROS1 并有 `audio_msgs/AudioData` 消息，未找到 ROS 会自动回退到 `pyaudio`。
- `AudioConfig.ros1_topic` / `ros1_node_name` / `ros1_queue_size`: ROS 发布主题与队列。
- `SystemConfig.voice_udp_host` / `voice_udp_port`(默认 5557) / `mic_udp_port`(默认 5558): UDP 指令目标。
- `SystemConfig.enable_keyword_detection`: 控制 ASR/LLM 关键词触发。
- 关键词配置：见 `audio_constants.py`，可调整触发短语。

## UDP 动作与麦克风切换

- ASR/LLM 识别到关键词会向 `voice_udp_port` 发送对应动作（如 wave/nod/shake/woshou/end/good、left/right/photo/end）。
- 音频播放完成时会向 `mic_udp_port` 发送 `send_microphone`，ASR 结束会发送 `release_microphone`，用于麦克风占用切换。
- 如需关闭，设置 `enable_keyword_detection=False` 或修改关键词表。

### 日志输出示例

```text
[ASR] 用户: 你好啊
[ASR-KWS] 匹配: '你好' → 动作: wave | UDP已发送
[UDP:5557] 动作指令: wave
[LLM] 机器人: 你好！很高兴见到你！
[LLM-KWS] 匹配: '很高兴' → 动作: good | UDP已发送
[UDP:5557] 动作指令: good
[UDP:5558] MIC指令: send_microphone
```

## ROS 扬声器模式

- 设置 `output_mode=ros1` 后，播放将发布到 `AudioConfig.ros_topic_name`。
- 需要 ROS1 环境与 `rospy`，主题类型优先使用 `audio_msgs/AudioData`，缺失时使用 `std_msgs/ByteMultiArray`。
- 若 ROS 不可用，播放自动回落到本地 `pyaudio`。

## RAG 集成（可选）

```python
from llm_client import RAGInterface, Document

class MyRAG(RAGInterface):
    async def retrieve(self, query: str, top_k: int = 5):
        pass

    async def build_context(self, documents):
        return "\n".join([doc.content for doc in documents])

# 使用
# system = VoiceDialogSystem()
# system.set_rag(MyRAG())
```

## 常见问题

- **设备不支持参数**：先 `--list-devices`，再指定正确索引。
- **回声严重**：保持回声消除开启或使用耳机。
- **ROS 无法播放**：确认已启动 ROS1，已安装 `rospy`，并存在目标主题类型。
- **ROS 音频首帧丢失**：已通过预热机制解决，发布者创建后会自动等待订阅者连接并发送静音帧。

## 参考链接

- [豆包流式语音识别](https://www.volcengine.com/docs/6561/1354869?lang=zh)
- [通义千问大模型](https://bailian.console.aliyun.com/)
- [讯飞在线语音合成](https://www.xfyun.cn/doc/tts/online_tts/API.html)
