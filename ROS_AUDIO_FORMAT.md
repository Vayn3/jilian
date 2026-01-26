# ROS音频格式配置说明

## 问题描述

下位机订阅ROS `/audio` 话题后，如果使用与上位机不同的音频格式播放，会导致音频乱码。

## 音频格式参数

### TTS输出格式（默认）
- **采样率**: 16000 Hz (16kHz)
- **位深度**: 16 bit (s16le, 即 signed 16-bit little-endian)
- **声道数**: 1 (单声道)

### 下位机播放格式（需配置）
根据您的下位机配置进行设置：
- **采样率**: 可能是 16kHz, 24kHz, 或其他
- **位深度**: 可能是 s16le (int16) 或 f32le (float32)
- **声道数**: 通常是 1 (单声道)

## 配置方法

在 `config.py` 中的 `AudioConfig` 类添加了两个新参数：

```python
# ROS1 音频格式转换配置（用于匹配下位机播放格式）
ros1_output_sample_rate: int = 24000  # 下位机期望的采样率
ros1_output_format: str = "f32le"     # 下位机期望的格式
```

### 参数说明

1. **ros1_output_sample_rate**: 下位机播放时使用的采样率
   - 如果下位机使用 24kHz 播放，设置为 `24000`
   - 如果下位机使用 16kHz 播放，设置为 `16000`（无需转换）
   - 如果下位机使用 48kHz 播放，设置为 `48000`

2. **ros1_output_format**: 下位机播放时使用的音频格式
   - `"s16le"`: 16位整数格式（signed 16-bit little-endian）- 无需转换
   - `"f32le"`: 32位浮点格式（float32 little-endian）- 需要转换

## 当前配置（针对24kHz + f32le）

```python
@dataclass
class AudioConfig:
    # ... 其他配置 ...
    
    # 输出模式
    output_mode: str = "ros1"
    
    # ROS1 扬声器发布配置
    ros1_topic: str = "/audio"
    ros1_node_name: str = "speaker_publisher"
    ros1_queue_size: int = 10
    ros1_latch: bool = False
    
    # ROS1 音频格式转换配置
    ros1_output_sample_rate: int = 24000  # 匹配下位机24kHz
    ros1_output_format: str = "f32le"     # 匹配下位机float32格式
```

## 格式转换流程

系统会自动在发送到ROS话题前进行格式转换：

```
TTS输出 (16kHz, int16)
    ↓
采样率转换 (16kHz → 24kHz)  [如果 ros1_output_sample_rate != 16000]
    ↓
格式转换 (int16 → float32)   [如果 ros1_output_format == "f32le"]
    ↓
发布到ROS话题 (/audio)
    ↓
下位机订阅并播放 (24kHz, float32)
```

## 验证配置

### 1. 检查下位机音频参数

在下位机上查看播放参数：
```bash
# 查看aplay或播放器配置
aplay -l
# 或查看您的播放程序参数
```

### 2. 监听ROS话题

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /audio
```

### 3. 测试音频

说话触发TTS，检查下位机是否正常播放，无乱码、无杂音。

## 常见配置组合

### 配置1: 16kHz + int16 (无需转换)
```python
ros1_output_sample_rate: int = 16000
ros1_output_format: str = "s16le"
```

### 配置2: 24kHz + int16
```python
ros1_output_sample_rate: int = 24000
ros1_output_format: str = "s16le"
```

### 配置3: 24kHz + float32 (当前配置)
```python
ros1_output_sample_rate: int = 24000
ros1_output_format: str = "f32le"
```

### 配置4: 48kHz + float32
```python
ros1_output_sample_rate: int = 48000
ros1_output_format: str = "f32le"
```

## 性能影响

- **无转换** (16kHz→16kHz, int16→int16): 无额外开销
- **仅采样率转换**: 轻微CPU开销
- **仅格式转换**: 非常小的CPU开销
- **两者都转换**: 轻微CPU开销（对现代CPU影响很小）

## 故障排除

### 问题: 音频仍然乱码
**解决方案**:
1. 确认下位机的实际播放参数
2. 检查 `config.py` 中的 `ros1_output_sample_rate` 和 `ros1_output_format` 是否匹配
3. 查看系统日志中的格式转换信息：
   ```
   音频重采样: 16000Hz -> 24000Hz
   音频格式转换: int16 -> float32
   ```

### 问题: 播放速度异常（太快或太慢）
**原因**: 采样率不匹配
**解决方案**: 调整 `ros1_output_sample_rate` 参数

### 问题: 音频有杂音/爆音
**原因**: 数据格式不匹配
**解决方案**: 检查并调整 `ros1_output_format` 参数

## 日志查看

启动系统时会输出格式转换日志：
```
2026-01-26 18:19:03,351 - audio_manager - DEBUG - 音频重采样: 16000Hz -> 24000Hz
2026-01-26 18:19:03,351 - audio_manager - DEBUG - 音频格式转换: int16 -> float32
```

如果看不到这些日志，可以在 `config.py` 中调整日志级别：
```python
log_level: str = "DEBUG"  # 改为DEBUG以查看详细日志
```
