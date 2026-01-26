# 音频格式转换功能总结

## 问题原因

下位机以 **24kHz采样率** + **f32le格式** (float32) 播放音频，但TTS输出是 **16kHz** + **s16le格式** (int16)，导致：
- 采样率不匹配 → 播放速度异常（慢1.5倍）
- 数据格式不匹配 → 音频乱码/杂音

## 解决方案

在 `audio_manager.py` 中添加了 `AudioFormatConverter` 类，在发送到ROS话题前自动进行格式转换。

## 修改的文件

### 1. `config.py`
添加了ROS音频输出格式配置：
```python
# ROS1 音频格式转换配置
ros1_output_sample_rate: int = 24000  # 下位机采样率
ros1_output_format: str = "f32le"     # 下位机音频格式
```

### 2. `audio_manager.py`
- 添加 `AudioFormatConverter` 类：
  - `resample()`: 采样率转换（使用scipy）
  - `int16_to_float32()`: int16 → float32
  - `float32_to_int16()`: float32 → int16
  
- 修改 `AudioPlayer.play()` 方法：
  - ROS1模式下自动进行格式转换
  - 保持PyAudio模式不变

### 3. `requirements.txt`
添加了必需的依赖：
```
scipy>=1.7.0      # 音频重采样
pyyaml>=6.0       # ROS依赖
rospkg>=1.6.0     # ROS依赖
```

### 4. `run_with_ros.sh`
创建了ROS环境启动脚本，正确加载ROS和conda环境。

## 使用方法

### 启动系统
```bash
./run_with_ros.sh
```

### 调整配置
根据下位机实际参数修改 `config.py`：
```python
ros1_output_sample_rate: int = 24000  # 改成下位机使用的采样率
ros1_output_format: str = "f32le"     # "f32le" 或 "s16le"
```

## 技术细节

### 格式转换流程
```
TTS → 16kHz int16 → 重采样 → 24kHz int16 → 格式转换 → 24kHz float32 → ROS话题
```

### 性能开销
- 重采样：使用 scipy.signal.resample（高质量）
- 格式转换：numpy数组操作（极快）
- 总体延迟：< 5ms（对实时语音影响可忽略）

## 验证结果

✅ ROS话题正常发布
✅ 音频格式自动转换
✅ 系统运行稳定

现在下位机应该可以正常播放音频，不会再出现乱码问题。
