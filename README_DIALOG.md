# 级联式武汉话实时语音对话系统

基于 **豆包ASR** → **千问LLM** → **讯飞TTS（武汉话）** 的流式语音对话系统。

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

## 特性

- ✅ **全流式处理**：ASR/LLM/TTS 全部采用流式传输，无中间文件
- ✅ **低延迟优化**：流水线并行架构，端到端延迟 <2秒
- ✅ **武汉话合成**：使用讯飞 `x2_xiaowang` 发音人
- ✅ **回声消除**：NLMS 自适应滤波算法
- ✅ **噪声抑制**：谱减法降噪
- ✅ **打断功能**：支持用户随时打断 AI 回答
- ✅ **模型切换**：支持 qwen-flash/turbo/plus/max 动态切换
- ✅ **RAG 接口**：预留知识库检索增强接口
- ✅ **可配置参数**：VAD 阈值、静音判停时间、音频设备等

## 文件结构

```
jilian/
├── main.py           # 主入口
├── config.py         # 全局配置
├── asr_client.py     # 豆包ASR客户端
├── llm_client.py     # 千问LLM客户端 + RAG接口
├── tts_client.py     # 讯飞TTS客户端
├── audio_manager.py  # 音频采集/播放/回声消除
├── conversation.py   # 对话管理/状态机/打断处理
└── requirements.txt  # 依赖
```

## 安装

```bash
cd jilian
pip install -r requirements.txt
```

### PyAudio 安装问题

Windows 上如果 `pip install pyaudio` 失败，可以：

```bash
# 方法1：使用 pipwin
pip install pipwin
pipwin install pyaudio

# 方法2：下载 whl 文件
# 从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio 下载对应版本
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

## 使用

### 基本使用

```bash
python main.py
```

### 命令行参数

```bash
# 列出可用音频设备
python main.py --list-devices

# 指定输入/输出设备
python main.py --input-device 1 --output-device 2

# 使用更准确的模型（延迟略高）
python main.py --model qwen-plus

# 调整 VAD 参数
python main.py --vad-threshold 600 --silence-ms 300

# 禁用回声消除（使用耳机时）
python main.py --no-aec

# 禁用打断功能
python main.py --no-barge-in

# 启用调试日志
python main.py --debug
```

### 参数说明

| 参数              | 默认值     | 说明                             |
| ----------------- | ---------- | -------------------------------- |
| `--model`         | qwen-flash | LLM模型（flash更快，plus更准确） |
| `--input-device`  | 默认       | 麦克风设备索引                   |
| `--output-device` | 默认       | 扬声器设备索引                   |
| `--vad-threshold` | 500        | VAD能量阈值（越大越不灵敏）      |
| `--silence-ms`    | 500        | 静音判停时间(ms)                 |
| `--no-aec`        | False      | 禁用回声消除                     |
| `--no-barge-in`   | False      | 禁用打断功能                     |

## 配置说明

所有配置都在 [config.py](config.py) 中，可以直接修改或通过环境变量覆盖：

### ASR 配置 (豆包)

```python
ASRConfig:
    app_key = "7381194560"
    access_key = "PmMJqNvQDStP4xpTi4pnuO83F793BplS"
    url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
    end_window_size_ms = 500  # 静音判停时间
```

### LLM 配置 (千问)

```python
LLMConfig:
    api_key = "sk-9d5d8ee616b740cd9e58a1152f84f471"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen-flash"  # 可选: qwen-flash, qwen-turbo, qwen-plus, qwen-max
    max_tokens = 512
    temperature = 0.7
```

### TTS 配置 (讯飞)

```python
TTSConfig:
    app_id = "6130dc73"
    api_key = "5af3f5aea48cb34ed691efee2a18780f"
    api_secret = "OGM2ZGZmNTI4OTJjZjgyNjM4ZThjOTk0"
    vcn = "x2_xiaowang"  # 武汉话发音人
    speed = 50  # 语速 0-100
```

### 音频配置

```python
AudioConfig:
    sample_rate = 16000
    channels = 1
    enable_aec = True   # 回声消除
    enable_ns = True    # 噪声抑制
    enable_agc = True   # 自动增益
```

## RAG 集成

系统预留了 RAG 接口，可以轻松集成知识库检索：

```python
from llm_client import RAGInterface, Document

class MyRAG(RAGInterface):
    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 实现向量检索逻辑
        # 例如调用 Milvus/Elasticsearch/Pinecone
        pass

    async def build_context(self, documents: List[Document]) -> str:
        # 构建上下文
        return "\n".join([doc.content for doc in documents])

# 使用
system = VoiceDialogSystem()
system.set_rag(MyRAG())
```

## 延迟优化建议

1. **使用 qwen-flash 模型**：首 Token 延迟约 150ms（qwen-plus 约 300ms）
2. **调小 silence-ms**：减少判停等待时间，但可能导致句子切断
3. **使用双向流式 ASR**：已默认使用 `bigmodel_async`
4. **禁用不必要功能**：如不需要打断，可用 `--no-barge-in`
5. **使用有线耳机**：可禁用回声消除 `--no-aec`，减少处理延迟

## 常见问题

### Q: 提示 "设备不支持指定参数"

A: 尝试使用 `--list-devices` 查看可用设备，然后指定正确的设备索引

### Q: 回声严重

A: 确保启用了回声消除（默认启用），或使用耳机

### Q: ASR 识别不准

A: 检查麦克风是否正常，尝试调高 `--vad-threshold`

### Q: TTS 无声音

A: 检查输出设备是否正确，音量是否正常

## API 密钥

当前使用的密钥已在 demo 文件中提供：

| 服务     | 密钥位置              |
| -------- | --------------------- |
| 豆包 ASR | config.py - ASRConfig |
| 千问 LLM | config.py - LLMConfig |
| 讯飞 TTS | config.py - TTSConfig |

**注意**：生产环境请替换为您自己的密钥。

## License

仅供学习研究使用。
