# -*- coding: utf-8 -*-
"""
音频管理模块 - 麦克风采集、音频播放、回声消除
支持全双工音频处理，实现同时录音和播放
支持 PyAudio 本地播放 / ROS1 话题发布双模式
支持 UDP 动作关键词发送和 MIC 指令控制
"""

import asyncio
import audioop
import json
import logging
import socket
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import pyaudio
from scipy import signal as scipy_signal

from config import AudioConfig, get_config
from ros_audio import Ros1SpeakerStream, is_ros_available

logger = logging.getLogger(__name__)


# ================== 音频格式转换工具 ==================
class AudioFormatConverter:
    """音频格式转换器，支持采样率和数据格式转换"""

    @staticmethod
    def resample(
        audio_data: bytes,
        from_rate: int,
        to_rate: int,
        sample_width: int = 2,
    ) -> bytes:
        """
        重采样音频

        Args:
            audio_data: 输入音频数据 (int16)
            from_rate: 原采样率
            to_rate: 目标采样率
            sample_width: 采样宽度（字节）

        Returns:
            重采样后的音频数据 (int16)
        """
        if from_rate == to_rate:
            return audio_data

        # 转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # 计算重采样比例
        resample_ratio = to_rate / from_rate
        new_length = int(len(audio_array) * resample_ratio)

        # 使用scipy进行重采样
        resampled_array = scipy_signal.resample(audio_array, new_length)

        # 转换回int16
        return resampled_array.astype(np.int16).tobytes()

    @staticmethod
    def int16_to_float32(audio_data: bytes) -> bytes:
        """
        将int16音频转换为float32格式

        Args:
            audio_data: int16格式的音频数据

        Returns:
            float32格式的音频数据
        """
        # 转换为int16数组
        int16_array = np.frombuffer(audio_data, dtype=np.int16)

        # 归一化到 [-1.0, 1.0]
        float32_array = int16_array.astype(np.float32) / 32768.0

        return float32_array.tobytes()

    @staticmethod
    def float32_to_int16(audio_data: bytes) -> bytes:
        """
        将float32音频转换为int16格式

        Args:
            audio_data: float32格式的音频数据

        Returns:
            int16格式的音频数据
        """
        # 转换为float32数组
        float32_array = np.frombuffer(audio_data, dtype=np.float32)

        # 反归一化并限幅
        int16_array = np.clip(float32_array * 32768.0, -32768, 32767).astype(np.int16)

        return int16_array.tobytes()


# ================== 简单回声消除器 ==================
class SimpleAEC:
    """
    简单自适应回声消除器
    基于NLMS（归一化最小均方）算法
    """

    def __init__(
        self,
        filter_length: int = 2048,
        step_size: float = 0.1,
        sample_rate: int = 16000,
    ):
        """
        Args:
            filter_length: 滤波器长度（采样点数）
            step_size: 自适应步长（0.01-1.0）
            sample_rate: 采样率
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.sample_rate = sample_rate

        # 自适应滤波器系数
        self.weights = np.zeros(filter_length, dtype=np.float32)

        # 参考信号缓冲区（播放的音频）
        self.ref_buffer = deque(maxlen=filter_length)
        for _ in range(filter_length):
            self.ref_buffer.append(0.0)

        # 平滑因子
        self.eps = 1e-6

    def update_reference(self, audio_data: bytes) -> None:
        """
        更新参考信号（播放的音频）

        Args:
            audio_data: PCM音频数据
        """
        # 转换为float数组
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        for sample in samples:
            self.ref_buffer.append(sample)

    def process(self, mic_data: bytes) -> bytes:
        """
        处理麦克风信号，消除回声

        Args:
            mic_data: 麦克风PCM数据

        Returns:
            消除回声后的PCM数据
        """
        # 转换为float数组
        mic_samples = (
            np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        output_samples = np.zeros_like(mic_samples)
        ref_array = np.array(self.ref_buffer, dtype=np.float32)

        for i, mic_sample in enumerate(mic_samples):
            # 获取参考信号向量
            ref_vector = ref_array[-self.filter_length :]

            # 估计回声
            echo_estimate = np.dot(self.weights, ref_vector)

            # 误差（消除回声后的信号）
            error = mic_sample - echo_estimate
            output_samples[i] = error

            # NLMS更新滤波器系数
            norm = np.dot(ref_vector, ref_vector) + self.eps
            self.weights += (self.step_size / norm) * error * ref_vector

            # 更新参考缓冲区
            self.ref_buffer.append(0.0)  # 如果没有新的播放数据

        # 转换回int16
        output_samples = np.clip(output_samples * 32768.0, -32768, 32767).astype(
            np.int16
        )
        return output_samples.tobytes()

    def reset(self) -> None:
        """重置滤波器"""
        self.weights.fill(0)
        self.ref_buffer.clear()
        for _ in range(self.filter_length):
            self.ref_buffer.append(0.0)


# ================== 噪声抑制器 ==================
class SimpleNoiseSupressor:
    """
    简单噪声抑制器
    基于谱减法
    """

    def __init__(self, sample_rate: int = 16000, frame_size: int = 512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_estimate = None
        self.noise_frames = 0
        self.noise_estimation_frames = 10  # 初始几帧用于估计噪声

    def process(self, audio_data: bytes) -> bytes:
        """
        处理音频，抑制噪声

        Args:
            audio_data: PCM音频数据

        Returns:
            抑制噪声后的PCM数据
        """
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # FFT
        spectrum = np.fft.rfft(samples)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # 噪声估计（使用前几帧）
        if self.noise_frames < self.noise_estimation_frames:
            if self.noise_estimate is None:
                self.noise_estimate = magnitude.copy()
            else:
                self.noise_estimate = (
                    self.noise_estimate * self.noise_frames + magnitude
                ) / (self.noise_frames + 1)
            self.noise_frames += 1
            return audio_data

        # 谱减法
        if self.noise_estimate is not None:
            # 确保长度匹配
            min_len = min(len(magnitude), len(self.noise_estimate))
            magnitude_clean = np.maximum(
                magnitude[:min_len] - 1.5 * self.noise_estimate[:min_len], 0
            )

            # 重建信号
            if len(magnitude) > min_len:
                magnitude_clean = np.concatenate([magnitude_clean, magnitude[min_len:]])

            spectrum_clean = magnitude_clean * np.exp(1j * phase)
            samples_clean = np.fft.irfft(spectrum_clean, len(samples))
            samples_clean = np.clip(samples_clean, -32768, 32767).astype(np.int16)
            return samples_clean.tobytes()

        return audio_data

    def reset(self) -> None:
        """重置噪声估计"""
        self.noise_estimate = None
        self.noise_frames = 0


# ================== 音频采集器 ==================
class AudioCapture:
    """
    麦克风音频采集器
    支持回声消除和噪声抑制
    """

    def __init__(
        self,
        output_queue: asyncio.Queue,
        config: Optional[AudioConfig] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.output_queue = output_queue
        self.config = config or get_config().audio
        self.loop = loop or asyncio.get_event_loop()

        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 音频处理组件
        self.aec: Optional[SimpleAEC] = None
        self.ns: Optional[SimpleNoiseSupressor] = None

        if self.config.enable_aec:
            self.aec = SimpleAEC(
                filter_length=self.config.aec_filter_length,
                sample_rate=self.config.sample_rate,
            )

        if self.config.enable_ns:
            self.ns = SimpleNoiseSupressor(
                sample_rate=self.config.sample_rate,
            )

    def _get_device_info(self) -> dict:
        """获取设备信息"""
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        device_count = self.pa.get_device_count()
        devices = []

        for i in range(device_count):
            info = self.pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "channels": info["maxInputChannels"],
                    }
                )

        return devices

    def list_devices(self) -> List[dict]:
        """列出可用的输入设备"""
        return self._get_device_info()

    def _check_device_compatibility(self, device_index: Optional[int]) -> bool:
        """检查设备兼容性"""
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        try:
            if device_index is not None:
                info = self.pa.get_device_info_by_index(device_index)
            else:
                info = self.pa.get_default_input_device_info()

            # 检查采样率支持
            supported = self.pa.is_format_supported(
                self.config.sample_rate,
                input_device=device_index or info["index"],
                input_channels=self.config.channels,
                input_format=pyaudio.paInt16,
            )
            return supported
        except Exception as e:
            logger.warning(f"设备兼容性检查失败: {e}")
            return False

    def update_playback_reference(self, audio_data: bytes) -> None:
        """
        更新播放参考信号（用于回声消除）

        Args:
            audio_data: 正在播放的音频数据
        """
        if self.aec:
            self.aec.update_reference(audio_data)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        if status:
            logger.warning(f"音频状态: {status}")

        if not self._running:
            return (None, pyaudio.paComplete)

        processed_data = in_data

        # 回声消除
        if self.aec and self.config.enable_aec:
            processed_data = self.aec.process(processed_data)

        # 噪声抑制
        if self.ns and self.config.enable_ns:
            processed_data = self.ns.process(processed_data)

        # 放入队列
        try:
            self.loop.call_soon_threadsafe(
                lambda: self.output_queue.put_nowait(processed_data)
            )
        except Exception as e:
            logger.warning(f"放入队列失败: {e}")

        return (None, pyaudio.paContinue)

    def start(self) -> None:
        """启动采集"""
        if self._running:
            return

        self._running = True

        if not self.pa:
            self.pa = pyaudio.PyAudio()

        # 检查设备兼容性
        if not self._check_device_compatibility(self.config.input_device_index):
            logger.warning("设备不支持指定参数，尝试使用默认设置")

        # 计算帧大小
        frames_per_buffer = int(
            self.config.sample_rate * self.config.input_chunk_ms / 1000
        )

        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._audio_callback,
            )

            self.stream.start_stream()
            logger.info(
                f"音频采集已启动 (采样率: {self.config.sample_rate}, 通道: {self.config.channels})"
            )

        except Exception as e:
            logger.error(f"启动音频采集失败: {e}")
            self._running = False
            raise

    def stop(self) -> None:
        """停止采集"""
        self._running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.pa:
            self.pa.terminate()
            self.pa = None

        logger.info("音频采集已停止")

    def reset_aec(self) -> None:
        """重置回声消除器"""
        if self.aec:
            self.aec.reset()

    def reset_ns(self) -> None:
        """重置噪声抑制器"""
        if self.ns:
            self.ns.reset()


# ================== UDP 动作控制器 ==================
class UDPActionController:
    """
    UDP 动作控制器
    发送语音关键词和 MIC 指令到下位机
    """

    def __init__(self):
        self._config = get_config()

        # 语音关键词 UDP socket
        self._voice_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # MIC 指令 UDP socket
        self._mic_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit_voice_keyword(self, keyword: str) -> None:
        """
        发送语音关键词到下位机

        Args:
            keyword: 关键词标签（如 wave/nod/shake/woshou/end 等）
        """
        if not self._config.enable_keyword_detection:
            return

        try:
            now = time.time()
            msg = json.dumps(
                {"type": "voice_keyword", "keyword": keyword, "timestamp": now}
            )
            self._voice_socket.sendto(
                msg.encode("utf-8"),
                (self._config.voice_udp_host, self._config.voice_udp_port),
            )
            logger.info(f"[KWS] 发送语音关键词：{keyword}")
        except Exception as e:
            logger.error(f"[KWS] 发送UDP失败: {e}")

    def send_mic_command(self, command: str) -> None:
        """
        发送 MIC 指令到下位机

        Args:
            command: 指令名称（"send_microphone" 或 "release_microphone"）
        """
        try:
            now = time.time()
            msg = json.dumps(
                {"type": "mic_command", "command": command, "timestamp": now}
            )
            self._mic_socket.sendto(
                msg.encode("utf-8"),
                (self._config.mic_udp_host, self._config.mic_udp_port),
            )
            logger.info(f"[MIC-UDP:{self._config.mic_udp_port}] 发送指令：{command}")
        except Exception as e:
            logger.error(f"[MIC-UDP] 发送失败: {e}")

    def close(self) -> None:
        """关闭 UDP sockets"""
        try:
            self._voice_socket.close()
        except Exception:
            pass
        try:
            self._mic_socket.close()
        except Exception:
            pass


# 全局 UDP 控制器实例
_udp_controller: Optional[UDPActionController] = None


def get_udp_controller() -> UDPActionController:
    """获取全局 UDP 控制器"""
    global _udp_controller
    if _udp_controller is None:
        _udp_controller = UDPActionController()
    return _udp_controller


# ================== 音频播放器 ==================
class AudioPlayer:
    """
    实时音频播放器
    支持 PyAudio 本地播放和 ROS1 话题发布双模式
    支持打断和回声消除参考信号更新
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        aec_callback: Optional[Callable[[bytes], None]] = None,
    ):
        self.config = config or get_config().audio
        self.aec_callback = aec_callback  # 回调用于更新AEC参考信号

        # 输出模式
        self._output_mode = (self.config.output_mode or "pyaudio").lower()

        # 音频格式转换器
        self.format_converter = AudioFormatConverter()

        # PyAudio 相关
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream = None  # 可以是 PyAudio Stream 或 Ros1SpeakerStream

        self._is_playing = False
        self._interrupted = threading.Event()
        self._lock = threading.Lock()

        # UDP 控制器
        self._udp_controller = get_udp_controller()

    def _check_output_device(self, device_index: Optional[int]) -> bool:
        """检查输出设备（仅 PyAudio 模式）"""
        if self._output_mode != "pyaudio":
            return True

        if not self.pa:
            self.pa = pyaudio.PyAudio()

        try:
            if device_index is not None:
                info = self.pa.get_device_info_by_index(device_index)
            else:
                info = self.pa.get_default_output_device_info()

            return info["maxOutputChannels"] > 0
        except Exception:
            return False

    def list_devices(self) -> List[dict]:
        """列出可用的输出设备"""
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "channels": info["maxOutputChannels"],
                    }
                )
        return devices

    def _ensure_stream(self) -> None:
        """确保播放流已打开"""
        if self.stream is not None:
            return

        if self._output_mode == "ros1":
            # ROS1 模式
            if not is_ros_available():
                logger.warning("未检测到 ROS1，回退到 PyAudio 模式")
                self._output_mode = "pyaudio"
            else:
                self.stream = Ros1SpeakerStream(
                    topic=self.config.ros1_topic,
                    node_name=self.config.ros1_node_name,
                    queue_size=self.config.ros1_queue_size,
                    latched=self.config.ros1_latch,
                )
                logger.info(f"使用 ROS1 模式发布音频到 {self.config.ros1_topic}")
                return

        # PyAudio 模式
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        if not self._check_output_device(self.config.output_device_index):
            logger.warning("输出设备不可用，使用默认设备")
            self.config.output_device_index = None

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            output_device_index=self.config.output_device_index,
            frames_per_buffer=self.config.output_buffer_size,
        )
        logger.info("使用 PyAudio 模式本地播放")

    def play(self, audio_data: bytes) -> bool:
        """
        播放音频数据

        Args:
            audio_data: PCM音频数据 (int16, TTS采样率)

        Returns:
            是否成功播放（未被打断）
        """
        if self._interrupted.is_set():
            return False

        with self._lock:
            self._ensure_stream()
            self._is_playing = True

            try:
                # 更新AEC参考信号（使用原始数据）
                if self.aec_callback:
                    self.aec_callback(audio_data)

                # ROS1模式需要格式转换
                output_data = audio_data
                if self._output_mode == "ros1":
                    # 1. 采样率转换（如果需要）
                    tts_rate = get_config().tts.sample_rate
                    target_rate = self.config.ros1_output_sample_rate

                    if tts_rate != target_rate:
                        output_data = self.format_converter.resample(
                            output_data,
                            from_rate=tts_rate,
                            to_rate=target_rate,
                            sample_width=2,
                        )
                        logger.debug(f"音频重采样: {tts_rate}Hz -> {target_rate}Hz")

                    # 2. 格式转换（int16 -> float32，如果需要）
                    if self.config.ros1_output_format == "f32le":
                        output_data = self.format_converter.int16_to_float32(
                            output_data
                        )
                        logger.debug("音频格式转换: int16 -> float32")

                # 写入音频数据
                self.stream.write(output_data)
                return True

            except Exception as e:
                logger.error(f"播放失败: {e}")
                return False
            finally:
                self._is_playing = False

    def on_playback_complete(self) -> None:
        """播放完成回调，发送递麦克风信号"""
        self._udp_controller.send_mic_command("send_microphone")

    def interrupt(self) -> None:
        """打断播放"""
        self._interrupted.set()
        logger.info("播放已打断")

    def resume(self) -> None:
        """恢复播放"""
        self._interrupted.clear()

    @property
    def is_playing(self) -> bool:
        """是否正在播放"""
        return self._is_playing

    def close(self) -> None:
        """关闭播放器"""
        if self.stream:
            if hasattr(self.stream, "stop_stream"):
                self.stream.stop_stream()
            if hasattr(self.stream, "close"):
                self.stream.close()
            self.stream = None

        if self.pa:
            self.pa.terminate()
            self.pa = None


# ================== 实时音频播放会话 ==================
class RealtimeAudioPlaySession:
    """
    实时音频播放会话
    从队列读取音频数据并播放
    """

    def __init__(
        self,
        input_queue: asyncio.Queue,
        config: Optional[AudioConfig] = None,
        aec_callback: Optional[Callable[[bytes], None]] = None,
    ):
        self.input_queue = input_queue
        self.player = AudioPlayer(config, aec_callback)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """启动播放会话"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("音频播放会话已启动")

    async def stop(self) -> None:
        """停止播放会话"""
        self._running = False
        self.player.interrupt()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.player.close()
        logger.info("音频播放会话已停止")

    def interrupt(self) -> None:
        """打断播放"""
        self.player.interrupt()

    def resume(self) -> None:
        """恢复播放"""
        self.player.resume()

    async def _run(self) -> None:
        """运行播放循环"""
        while self._running:
            try:
                # 等待音频数据
                audio_data = await self.input_queue.get()

                if audio_data is None:  # 一轮结束标记
                    # 播放完成，发送递麦克风信号
                    self.player.on_playback_complete()
                    continue

                # 在线程池中播放（避免阻塞事件循环）
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.player.play, audio_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"播放出错: {e}")


# ================== VAD (语音活动检测) ==================
class SimpleVAD:
    """
    简单语音活动检测
    基于能量阈值
    """

    def __init__(
        self,
        threshold: int = 500,
        sample_width: int = 2,
    ):
        self.threshold = threshold
        self.sample_width = sample_width

    def is_speech(self, audio_data: bytes) -> bool:
        """
        检测是否有语音

        Args:
            audio_data: PCM音频数据

        Returns:
            是否检测到语音
        """
        if not audio_data:
            return False

        rms = audioop.rms(audio_data, self.sample_width)
        return rms > self.threshold

    def set_threshold(self, threshold: int) -> None:
        """设置阈值"""
        self.threshold = threshold
