# -*- coding: utf-8 -*-
"""
ROS1 音频发布模块
支持将音频数据发布到 ROS 话题，供下位机播放
"""

from typing import Optional

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


class Ros1SpeakerStream:
    """
    ROS1 扬声器音频发布流
    将 PCM 音频数据发布到 ROS 话题，供下位机订阅并播放
    
    支持两种消息类型：
    - audio_common_msgs/AudioData（优先）
    - std_msgs/ByteMultiArray（兜底）
    """
    
    def __init__(
        self,
        topic: str = "/robot/speaker/audio",
        node_name: str = "speaker_publisher",
        queue_size: int = 10,
        latched: bool = False,
    ):
        """
        初始化 ROS1 扬声器发布流
        
        Args:
            topic: ROS 话题名称
            node_name: ROS 节点名称（如尚未初始化则自动创建）
            queue_size: 发布队列大小
            latched: 是否使用 latched 模式
        """
        if not _HAS_ROS1:
            raise RuntimeError("未检测到 ROS1 (rospy)。请在 ROS1 环境运行。")
        
        # 如果 ROS 节点尚未初始化，则自动初始化
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        
        self.topic = topic
        self._use_audio_msg = _HAS_AUDIO_DATA_MSG
        
        # 根据可用的消息类型创建发布者
        if self._use_audio_msg:
            self._pub = rospy.Publisher(
                topic, RosAudioData, queue_size=queue_size, latch=latched
            )
            print(f"[ROS-AUDIO] 使用 audio_common_msgs/AudioData 发布到 {topic}")
        else:
            self._pub = rospy.Publisher(
                topic, ByteMultiArray, queue_size=queue_size, latch=latched
            )
            print(f"[ROS-AUDIO] 使用 std_msgs/ByteMultiArray 发布到 {topic}")
        
        self._closed = False
    
    def write(self, audio_bytes: bytes) -> None:
        """
        发布音频数据到 ROS 话题
        
        Args:
            audio_bytes: PCM 音频数据
        """
        if self._closed:
            return
        
        if self._use_audio_msg:
            msg = RosAudioData()
            msg.data = list(audio_bytes)
        else:
            msg = ByteMultiArray()
            msg.data = list(audio_bytes)
        
        self._pub.publish(msg)
    
    def stop_stream(self) -> None:
        """停止流（兼容 PyAudio 接口）"""
        pass
    
    def close(self) -> None:
        """关闭发布者"""
        self._closed = True


def is_ros_available() -> bool:
    """检查 ROS1 是否可用"""
    return _HAS_ROS1


def is_audio_msg_available() -> bool:
    """检查 audio_common_msgs 是否可用"""
    return _HAS_AUDIO_DATA_MSG
