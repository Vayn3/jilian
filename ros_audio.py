# -*- coding: utf-8 -*-
"""
ROS1 音频发布模块
支持将音频数据发布到 ROS 话题，供下位机播放
"""

import logging
import sys
from typing import Optional  # noqa: F401

logger = logging.getLogger(__name__)

# ---------- ROS 相关导入 ----------
try:
    import rospy
    from std_msgs.msg import Bool, ByteMultiArray

    try:
        from audio_common_msgs.msg import AudioData as RosAudioData

        _HAS_AUDIO_DATA_MSG: bool = True
    except Exception:
        RosAudioData = None
        _HAS_AUDIO_DATA_MSG = False
    _HAS_ROS1: bool = True
except Exception:
    rospy = None
    Bool = None
    ByteMultiArray = None
    RosAudioData = None
    _HAS_ROS1 = False
    _HAS_AUDIO_DATA_MSG = False


# ROS 发布者预热配置
ROS_WARMUP_DELAY_SEC = 0.3  # 发布者创建后等待订阅者连接的时间
ROS_WARMUP_SILENCE_FRAMES = 5  # 预热静音帧数
ROS_WARMUP_FRAME_SIZE = 1920  # 每帧静音数据大小（字节）


class Ros1SpeakerStream:
    """
    ROS1 扬声器音频发布流
    将 PCM 音频数据发布到 ROS 话题，供下位机订阅并播放

    支持两种消息类型：
    - audio_common_msgs/AudioData（优先）
    - std_msgs/ByteMultiArray（兜底）

    特性：
    - 创建发布者后自动预热（等待订阅者连接+发送静音帧），避免首帧丢失
    """

    def __init__(
        self,
        topic: str = "/robot/speaker/audio",
        node_name: str = "speaker_publisher",
        queue_size: int = 10,
        latched: bool = False,
        warmup: bool = True,
    ):
        """
        初始化 ROS1 扬声器发布流

        Args:
            topic: ROS 话题名称
            node_name: ROS 节点名称（如尚未初始化则自动创建）
            queue_size: 发布队列大小
            latched: 是否使用 latched 模式
            warmup: 是否进行预热（发送静音帧等待订阅者）
        """
        if not _HAS_ROS1:
            raise RuntimeError("未检测到 ROS1 (rospy)。请在 ROS1 环境运行。")

        # 如果 ROS 节点尚未初始化，则自动初始化
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
            # ROS 初始化会重置 logging，这里强制恢复到控制台输出
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                force=True,
            )
            logging.getLogger("rosout").setLevel(logging.WARNING)

        self.topic = topic
        self._use_audio_msg = _HAS_AUDIO_DATA_MSG
        self._closed = False
        self._warmed_up = False

        # 根据可用的消息类型创建发布者
        if self._use_audio_msg:
            self._pub = rospy.Publisher(
                topic, RosAudioData, queue_size=queue_size, latch=latched
            )
            logger.info(f"[ROS] 使用 audio_common_msgs/AudioData 发布到 {topic}")
        else:
            self._pub = rospy.Publisher(
                topic, ByteMultiArray, queue_size=queue_size, latch=latched
            )
            logger.info(f"[ROS] 使用 std_msgs/ByteMultiArray 发布到 {topic}")

        # 预热：等待订阅者连接并发送静音帧
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """
        预热发布者：等待订阅者连接并发送静音数据
        解决 ROS 订阅者首次接收丢帧问题
        """
        import time

        logger.info(f"[ROS] 预热中，等待订阅者连接...")

        # 等待一段时间让订阅者有机会连接
        time.sleep(ROS_WARMUP_DELAY_SEC)

        # 发送静音帧预热通道
        silence_data = bytes(ROS_WARMUP_FRAME_SIZE)  # 全零静音数据
        for i in range(ROS_WARMUP_SILENCE_FRAMES):
            self._publish_raw(silence_data)
            time.sleep(0.02)  # 20ms 间隔

        self._warmed_up = True
        logger.info(f"[ROS] 预热完成，已发送 {ROS_WARMUP_SILENCE_FRAMES} 帧静音数据")

    def _publish_raw(self, audio_bytes: bytes) -> None:
        """内部发布方法"""
        if self._use_audio_msg:
            msg = RosAudioData()
            msg.data = list(audio_bytes)
        else:
            msg = ByteMultiArray()
            msg.data = list(audio_bytes)
        self._pub.publish(msg)

    def write(self, audio_bytes: bytes) -> None:
        """
        发布音频数据到 ROS 话题

        Args:
            audio_bytes: PCM 音频数据
        """
        if self._closed:
            return

        self._publish_raw(audio_bytes)

    def stop_stream(self) -> None:
        """停止流（兼容 PyAudio 接口）"""
        # 兼容 PyAudio Stream 接口

    def close(self) -> None:
        """关闭发布者"""
        self._closed = True
        logger.info("[ROS] 音频发布流已关闭")


def is_ros_available() -> bool:
    """检查 ROS1 是否可用"""
    return _HAS_ROS1


def is_audio_msg_available() -> bool:
    """检查 audio_common_msgs 是否可用"""
    return _HAS_AUDIO_DATA_MSG
