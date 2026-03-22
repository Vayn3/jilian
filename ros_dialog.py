# -*- coding: utf-8 -*-
"""
ROS1 对话文本流发布模块
发布 LLM 流式文本（chunk + 句终）到 ROS topic，供其他设备订阅。
"""

import json
import logging
import sys
import time

logger = logging.getLogger(__name__)

try:
    import rospy
    from std_msgs.msg import String

    _HAS_ROS1: bool = True
except ImportError:
    rospy = None
    String = None
    _HAS_ROS1 = False


class Ros1DialogTextPublisher:
    """ROS1 文本流发布器（std_msgs/String, JSON payload）"""

    def __init__(
        self,
        topic: str = "/hri/dialog/llm_stream",
        node_name: str = "dialog_text_publisher",
        queue_size: int = 50,
    ):
        if not _HAS_ROS1:
            raise RuntimeError("未检测到 ROS1 (rospy)。请在 ROS1 环境运行。")

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
        self._pub = rospy.Publisher(topic, String, queue_size=queue_size)
        self._closed = False
        self._seq = 0
        self._utterance_id = 0

        logger.info("[ROS] 文本流发布启用: %s", topic)

    def start_new_utterance(self) -> None:
        """开始新一轮机器人回复（可选调用）"""
        self._utterance_id += 1

    def publish_text(self, text: str, is_final: bool) -> None:
        """发布文本事件。

        数据格式（JSON 字符串）:
        {
          "seq": int,
          "utterance_id": int,
          "is_final": bool,
          "text": str,
          "timestamp": float
        }
        """
        if self._closed:
            return

        payload = {
            "seq": self._seq,
            "utterance_id": self._utterance_id,
            "is_final": bool(is_final),
            "text": text,
            "timestamp": time.time(),
        }
        self._seq += 1

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub.publish(msg)

    def close(self) -> None:
        self._closed = True
        logger.info("[ROS] 文本流发布器已关闭")


def is_ros_available() -> bool:
    """检查 ROS1 是否可用"""
    return _HAS_ROS1
