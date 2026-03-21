#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 流式接收端示例
- 订阅音频 topic（默认 /audio）并实时播放
- 订阅 LLM 文本流 topic（默认 /dialog/llm_stream）并打印

说明：
1) 为兼容你当前发送端，音频消息类型支持两种：
   - audio_common_msgs/AudioData（优先）
   - std_msgs/ByteMultiArray（兜底）
2) 文本流消息使用 std_msgs/String，内容为 JSON：
   {"seq":int,"utterance_id":int,"is_final":bool,"text":str,"timestamp":float}
"""

import argparse
import json
import logging
import threading

import numpy as np
import rospy
from std_msgs.msg import ByteMultiArray, String

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    HAS_PYAUDIO = False

try:
    from audio_common_msgs.msg import AudioData

    HAS_AUDIO_DATA = True
except ImportError:
    AudioData = None
    HAS_AUDIO_DATA = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioSink:
    def __init__(
        self,
        sample_rate: int,
        sample_format: str,
        channels: int,
        output_device_index: int = None,
        enable_playback: bool = True,
    ):
        self.source_rate = sample_rate
        self.sample_format = sample_format.lower()
        self.channels = channels
        self.output_device_index = output_device_index
        self.enable_playback = enable_playback

        if self.enable_playback and not HAS_PYAUDIO:
            raise OSError("未安装 PyAudio，无法本地播放；可使用 --no-playback")

        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.device_rate = self.source_rate

        if self.enable_playback:
            self._open_output_stream_with_fallback()

    def _open_output_stream_with_fallback(self) -> None:
        """打开输出流：优先源采样率，不支持则自动降级到设备可用采样率"""
        preferred_rates = [
            self.source_rate,
            48000,
            44100,
            32000,
            24000,
            22050,
            16000,
            11025,
            8000,
        ]
        # 去重并保持顺序
        rates = []
        for r in preferred_rates:
            if r not in rates:
                rates.append(r)

        last_error = None
        for rate in rates:
            try:
                self.pa.is_format_supported(
                    rate,
                    output_device=self.output_device_index,
                    output_channels=self.channels,
                    output_format=pyaudio.paInt16,
                )
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=rate,
                    output=True,
                    output_device_index=self.output_device_index,
                    frames_per_buffer=1024,
                )
                self.device_rate = rate
                if self.device_rate != self.source_rate:
                    logger.warning(
                        "输出设备不支持 %sHz，已自动切换为 %sHz 并启用重采样",
                        self.source_rate,
                        self.device_rate,
                    )
                else:
                    logger.info("输出流已打开: %sHz", self.device_rate)
                return
            except (ValueError, OSError) as e:
                last_error = e
                continue

        raise OSError(f"无法打开任何可用音频输出采样率: {last_error}")

    @staticmethod
    def list_output_devices() -> None:
        """列出可用输出设备"""
        if not HAS_PYAUDIO:
            print("未安装 PyAudio，无法列出输出设备")
            return

        pa = pyaudio.PyAudio()
        try:
            print("\n=== 输出设备 ===")
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxOutputChannels", 0) > 0:
                    print(
                        f"[{i}] {info['name']} | default_rate={int(info['defaultSampleRate'])} | channels={int(info['maxOutputChannels'])}"
                    )
        finally:
            pa.terminate()

    def _to_int16(self, audio_bytes: bytes) -> bytes:
        """输入可能是 f32le 或 s16le，统一转 int16 送给 PyAudio"""
        if self.sample_format == "f32le":
            float_data = np.frombuffer(audio_bytes, dtype=np.float32)
            int_data = np.clip(float_data * 32768.0, -32768, 32767).astype(np.int16)
            return int_data.tobytes()
        return audio_bytes

    def _resample_if_needed(self, int16_bytes: bytes) -> bytes:
        if self.source_rate == self.device_rate:
            return int16_bytes
        data = np.frombuffer(int16_bytes, dtype=np.int16)
        if data.size == 0:
            return int16_bytes

        target_len = max(1, int(round(data.size * self.device_rate / self.source_rate)))
        x_old = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        resampled = np.interp(x_new, x_old, data.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    def play(self, audio_bytes: bytes) -> None:
        if not audio_bytes or not self.enable_playback or self.stream is None:
            return
        out = self._to_int16(audio_bytes)
        out = self._resample_if_needed(out)
        self.stream.write(out)

    def close(self) -> None:
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            self.pa.terminate()


class StreamSubscriber:
    def __init__(
        self,
        audio_topic: str,
        text_topic: str,
        sample_rate: int,
        sample_format: str,
        channels: int,
        audio_msg_type: str,
        output_device_index: int = None,
        enable_playback: bool = True,
    ):
        self.audio_topic = audio_topic
        self.text_topic = text_topic
        self.sample_rate = sample_rate
        self.sample_format = sample_format.lower()
        self.channels = channels
        try:
            self.sink = AudioSink(
                sample_rate,
                sample_format,
                channels,
                output_device_index=output_device_index,
                enable_playback=enable_playback,
            )
        except OSError as e:
            self.sink = None
            logger.error("音频播放初始化失败，切换为仅文本订阅模式: %s", e)
        self.audio_bytes_total = 0
        self.audio_chunks = 0
        self.audio_msg_type = audio_msg_type
        self._stats_lock = threading.Lock()
        self._stats_stop_event = threading.Event()
        self._stats_thread = None

    def _bytes_per_sample(self) -> int:
        return 4 if self.sample_format == "f32le" else 2

    def _print_stats(self) -> None:
        with self._stats_lock:
            chunks = self.audio_chunks
            total_bytes = self.audio_bytes_total

        bps = (
            self._bytes_per_sample() * max(1, self.channels) * max(1, self.sample_rate)
        )
        est_seconds = (total_bytes / bps) if bps > 0 else 0.0
        print(
            f"[AUDIO-STATS] chunks={chunks} bytes={total_bytes} est_duration={est_seconds:.2f}s",
            flush=True,
        )

    def _stats_loop(self) -> None:
        while not self._stats_stop_event.wait(1.0):
            self._print_stats()

    def on_audio_byte_multi(self, msg: ByteMultiArray) -> None:
        data = bytes(msg.data)
        with self._stats_lock:
            self.audio_chunks += 1
            self.audio_bytes_total += len(data)
        if self.sink:
            self.sink.play(data)

    def on_audio_audio_data(self, msg) -> None:
        data = bytes(msg.data)
        with self._stats_lock:
            self.audio_chunks += 1
            self.audio_bytes_total += len(data)
        if self.sink:
            self.sink.play(data)

    def on_text(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            seq = payload.get("seq")
            utterance_id = payload.get("utterance_id")
            is_final = payload.get("is_final")
            text = payload.get("text", "")
            tag = "FINAL" if is_final else "CHUNK"
            print(f"[LLM-{tag}] u={utterance_id} seq={seq}: {text}")
        except json.JSONDecodeError:
            print(f"[LLM-RAW] {msg.data}")

    def start(self) -> None:
        rospy.Subscriber(self.text_topic, String, self.on_text, queue_size=200)
        self._stats_thread = threading.Thread(
            target=self._stats_loop,
            name="audio-stats-printer",
            daemon=True,
        )
        self._stats_thread.start()

        if self.audio_msg_type == "audio_data":
            if not HAS_AUDIO_DATA:
                raise RuntimeError(
                    "当前环境缺少 audio_common_msgs，请改用 --audio-msg-type byte_multi"
                )
            rospy.Subscriber(
                self.audio_topic, AudioData, self.on_audio_audio_data, queue_size=200
            )
            logger.info("订阅音频: %s (audio_common_msgs/AudioData)", self.audio_topic)
        else:
            rospy.Subscriber(
                self.audio_topic,
                ByteMultiArray,
                self.on_audio_byte_multi,
                queue_size=200,
            )
            logger.info("订阅音频: %s (std_msgs/ByteMultiArray)", self.audio_topic)

        logger.info("订阅文本: %s (std_msgs/String JSON)", self.text_topic)

    def close(self) -> None:
        self._stats_stop_event.set()
        if self._stats_thread and self._stats_thread.is_alive():
            self._stats_thread.join(timeout=1.5)
        if self.sink:
            self.sink.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS 流式接收端示例")
    parser.add_argument("--audio-topic", default="/audio")
    parser.add_argument("--text-topic", default="/dialog/llm_stream")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--sample-format", choices=["s16le", "f32le"], default="f32le")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=None, help="输出设备索引")
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="仅订阅并打印文本/音频统计，不做本地播放",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出输出设备并退出",
    )
    parser.add_argument(
        "--audio-msg-type",
        choices=["audio_data", "byte_multi"],
        default="audio_data" if HAS_AUDIO_DATA else "byte_multi",
        help="需与发送端消息类型一致",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioSink.list_output_devices()
        return

    if not HAS_PYAUDIO and not args.no_playback:
        logger.warning("检测到未安装 PyAudio，自动切换为 --no-playback 模式")
        args.no_playback = True

    rospy.init_node("ros_stream_subscriber_demo", anonymous=True)
    sub = StreamSubscriber(
        audio_topic=args.audio_topic,
        text_topic=args.text_topic,
        sample_rate=args.sample_rate,
        sample_format=args.sample_format,
        channels=args.channels,
        audio_msg_type=args.audio_msg_type,
        output_device_index=args.output_device,
        enable_playback=not args.no_playback,
    )
    sub.start()

    logger.info("接收端已启动，按 Ctrl+C 退出")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()


if __name__ == "__main__":
    main()
