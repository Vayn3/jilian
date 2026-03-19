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

import pyaudio
import rospy
from std_msgs.msg import ByteMultiArray, String

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
    def __init__(self, sample_rate: int, sample_format: str, channels: int):
        self.sample_rate = sample_rate
        self.sample_format = sample_format.lower()
        self.channels = channels

        self.pa = pyaudio.PyAudio()
        if self.sample_format == "f32le":
            pa_format = pyaudio.paFloat32
        else:
            pa_format = pyaudio.paInt16

        self.stream = self.pa.open(
            format=pa_format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024,
        )

    def play(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        self.stream.write(audio_bytes)

    def close(self) -> None:
        try:
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
    ):
        self.audio_topic = audio_topic
        self.text_topic = text_topic
        self.sink = AudioSink(sample_rate, sample_format, channels)
        self.audio_bytes_total = 0
        self.audio_chunks = 0
        self.audio_msg_type = audio_msg_type

    def on_audio_byte_multi(self, msg: ByteMultiArray) -> None:
        data = bytes(msg.data)
        self.audio_chunks += 1
        self.audio_bytes_total += len(data)
        self.sink.play(data)

    def on_audio_audio_data(self, msg) -> None:
        data = bytes(msg.data)
        self.audio_chunks += 1
        self.audio_bytes_total += len(data)
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
        self.sink.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS 流式接收端示例")
    parser.add_argument("--audio-topic", default="/audio")
    parser.add_argument("--text-topic", default="/dialog/llm_stream")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--sample-format", choices=["s16le", "f32le"], default="f32le")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument(
        "--audio-msg-type",
        choices=["audio_data", "byte_multi"],
        default="audio_data" if HAS_AUDIO_DATA else "byte_multi",
        help="需与发送端消息类型一致",
    )
    args = parser.parse_args()

    rospy.init_node("ros_stream_subscriber_demo", anonymous=True)
    sub = StreamSubscriber(
        audio_topic=args.audio_topic,
        text_topic=args.text_topic,
        sample_rate=args.sample_rate,
        sample_format=args.sample_format,
        channels=args.channels,
        audio_msg_type=args.audio_msg_type,
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
