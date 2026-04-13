import logging
import sys
import time
from typing import Optional

from config import AudioConfig, get_config
from duplex_audio import build_stop_message, pack_audio_frame

logger = logging.getLogger(__name__)

try:
    import rospy
    from std_msgs.msg import ByteMultiArray, String

    try:
        from audio_common_msgs.msg import AudioData as RosAudioData

        _HAS_AUDIO_DATA_MSG: bool = True
    except Exception:
        RosAudioData = None
        _HAS_AUDIO_DATA_MSG = False
    _HAS_ROS1: bool = True
except Exception:
    rospy = None
    ByteMultiArray = None
    String = None
    RosAudioData = None
    _HAS_ROS1 = False
    _HAS_AUDIO_DATA_MSG = False


ROS_WARMUP_DELAY_SEC = 0.3
ROS_WARMUP_SILENCE_FRAMES = 5
ROS_WARMUP_FRAME_SIZE = 1920


class Ros1SpeakerStream:
    """ROS1 speaker publisher with optional duplex control channel."""

    def __init__(
        self,
        topic: str = "/robot/speaker/audio",
        control_topic: Optional[str] = None,
        node_name: str = "speaker_publisher",
        queue_size: int = 10,
        latched: bool = False,
        warmup: bool = True,
        duplex_mode: str = "half",
    ):
        if not _HAS_ROS1:
            raise RuntimeError("ROS1 (rospy) is required for Ros1SpeakerStream")

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                force=True,
            )
            logging.getLogger("rosout").setLevel(logging.WARNING)

        self.topic = topic
        self.control_topic = control_topic or get_config().audio.ros1_control_topic
        self.duplex_mode = (duplex_mode or "half").lower()
        self._use_audio_msg = _HAS_AUDIO_DATA_MSG
        self._closed = False
        self._last_utterance_id: Optional[int] = None

        if self._use_audio_msg:
            self._pub = rospy.Publisher(
                topic, RosAudioData, queue_size=queue_size, latch=latched
            )
            logger.info("[ROS] publishing AudioData to %s", topic)
        else:
            self._pub = rospy.Publisher(
                topic, ByteMultiArray, queue_size=queue_size, latch=latched
            )
            logger.info("[ROS] publishing ByteMultiArray to %s", topic)

        self._control_pub = rospy.Publisher(
            self.control_topic,
            String,
            queue_size=max(1, min(queue_size, 10)),
        )

        if warmup:
            self._warmup()

    def _publish_raw(self, audio_bytes: bytes) -> None:
        if self._use_audio_msg:
            msg = RosAudioData()
            msg.data = list(audio_bytes)
        else:
            msg = ByteMultiArray()
            msg.data = list(audio_bytes)
        self._pub.publish(msg)

    def _warmup(self) -> None:
        logger.info("[ROS] warming up speaker publisher, waiting for subscribers...")
        time.sleep(ROS_WARMUP_DELAY_SEC)

        silence_data = bytes(ROS_WARMUP_FRAME_SIZE)
        for _ in range(ROS_WARMUP_SILENCE_FRAMES):
            self._publish_raw(silence_data)
            time.sleep(0.02)

    def write(
        self,
        audio_bytes: bytes,
        utterance_id: Optional[int] = None,
        chunk_seq: int = 0,
        is_last: bool = False,
    ) -> None:
        if self._closed:
            return

        payload = audio_bytes
        if self.duplex_mode == "full" and utterance_id is not None:
            payload = pack_audio_frame(
                payload,
                utterance_id=utterance_id,
                chunk_seq=chunk_seq,
                is_last=is_last,
            )
            self._last_utterance_id = utterance_id

        self._publish_raw(payload)

    def send_stop(self, utterance_id: int, reason: str = "barge_in") -> None:
        if self._closed or String is None:
            return

        msg = String()
        msg.data = build_stop_message(utterance_id=utterance_id, reason=reason)
        self._control_pub.publish(msg)
        logger.info("[ROS] sent stop control for utterance %s (%s)", utterance_id, reason)

    def stop_stream(self) -> None:
        """Compatibility hook for PyAudio-like interface."""

    def close(self) -> None:
        if self.duplex_mode == "full" and self._last_utterance_id is not None:
            self.send_stop(self._last_utterance_id, reason="stream_close")
        self._closed = True
        logger.info("[ROS] speaker publisher closed")


def is_ros_available() -> bool:
    return _HAS_ROS1


def is_audio_msg_available() -> bool:
    return _HAS_AUDIO_DATA_MSG
