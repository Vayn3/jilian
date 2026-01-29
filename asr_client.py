# -*- coding: utf-8 -*-
"""
ASR客户端模块 - 豆包流式语音识别
基于双向流式优化版API，支持实时识别和VAD
支持 ASR 关键词检测并发送 UDP 动作指令
"""

import asyncio
import gzip
import json
import logging
import struct
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import aiohttp

from audio_constants import ASR_KWS_PATTERNS
from audio_manager import get_udp_controller
from config import ASRConfig, get_config

logger = logging.getLogger(__name__)


# ================== 协议常量 ==================
class ProtocolVersion:
    V1 = 0b0001


class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ACK = 0b1011
    SERVER_ERROR_RESPONSE = 0b1111


class MessageTypeSpecificFlags:
    NO_SEQUENCE = 0b0000
    POS_SEQUENCE = 0b0001
    NEG_SEQUENCE = 0b0010
    NEG_WITH_SEQUENCE = 0b0011
    HAS_EVENT = 0b0100


class SerializationType:
    NO_SERIALIZATION = 0b0000
    JSON = 0b0001


class CompressionType:
    NO_COMPRESSION = 0b0000
    GZIP = 0b0001


# ================== 响应结构 ==================
class ASRResponse:
    """ASR识别响应"""

    def __init__(self):
        self.code: int = 0
        self.event: int = 0
        self.is_last_package: bool = False
        self.payload_sequence: int = 0
        self.payload_size: int = 0
        self.payload_msg: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "event": self.event,
            "is_last_package": self.is_last_package,
            "payload_sequence": self.payload_sequence,
            "payload_size": self.payload_size,
            "payload_msg": self.payload_msg,
        }

    def get_text(self) -> str:
        """获取识别文本"""
        if not self.payload_msg:
            return ""
        result = self.payload_msg.get("result", {})
        return result.get("text", "") or ""

    def get_utterances(self) -> List[Dict]:
        """获取分句列表"""
        if not self.payload_msg:
            return []
        result = self.payload_msg.get("result", {})
        return result.get("utterances", [])

    def get_definite_text(self) -> Optional[str]:
        """获取已确定的完整句子文本"""
        utterances = self.get_utterances()
        for utt in utterances:
            if utt.get("definite"):
                return utt.get("text", "")
        return None


# ================== 请求头封装 ==================
class ASRRequestHeader:
    def __init__(self):
        self.message_type = MessageType.CLIENT_FULL_REQUEST
        self.message_type_specific_flags = MessageTypeSpecificFlags.POS_SEQUENCE
        self.serialization_type = SerializationType.JSON
        self.compression_type = CompressionType.GZIP
        self.reserved_data = bytes([0x00])

    def with_message_type(self, message_type: int) -> "ASRRequestHeader":
        self.message_type = message_type
        return self

    def with_message_type_specific_flags(self, flags: int) -> "ASRRequestHeader":
        self.message_type_specific_flags = flags
        return self

    def to_bytes(self) -> bytes:
        header = bytearray()
        header.append((ProtocolVersion.V1 << 4) | 0b0001)
        header.append((self.message_type << 4) | self.message_type_specific_flags)
        header.append((self.serialization_type << 4) | self.compression_type)
        header.extend(self.reserved_data)
        return bytes(header)


# ================== 响应解析器 ==================
class ResponseParser:
    @staticmethod
    def parse_response(msg: bytes) -> ASRResponse:
        resp = ASRResponse()

        if len(msg) < 4:
            resp.code = -1
            return resp

        header_size_words = msg[0] & 0x0F
        message_type = msg[1] >> 4
        flags = msg[1] & 0x0F
        serialization_method = msg[2] >> 4
        compression = msg[2] & 0x0F

        payload = msg[header_size_words * 4 :]

        # flags解析
        if flags & 0x01:  # POS_SEQUENCE
            if len(payload) >= 4:
                resp.payload_sequence = struct.unpack(">i", payload[:4])[0]
                payload = payload[4:]

        if flags & 0x02:  # NEG_SEQUENCE
            resp.is_last_package = True

        if flags & 0x04:  # HAS_EVENT
            if len(payload) >= 4:
                resp.event = struct.unpack(">i", payload[:4])[0]
                payload = payload[4:]

        # message_type
        if message_type in (MessageType.SERVER_FULL_RESPONSE, MessageType.SERVER_ACK):
            if len(payload) >= 4:
                resp.payload_size = struct.unpack(">I", payload[:4])[0]
                payload = payload[4:]
        elif message_type == MessageType.SERVER_ERROR_RESPONSE:
            # 错误响应包含 code(int32) + payload_size(uint32)
            if len(payload) >= 8:
                resp.code = struct.unpack(">i", payload[:4])[0]
                resp.payload_size = struct.unpack(">I", payload[4:8])[0]
                payload = payload[8:]

        if not payload:
            return resp

        # 解压
        if compression == CompressionType.GZIP:
            try:
                payload = gzip.decompress(payload)
            except Exception as e:
                logger.warning(f"解压失败: {e}")
                return resp

        # JSON解析
        if serialization_method == SerializationType.JSON:
            try:
                resp.payload_msg = json.loads(payload.decode("utf-8"))
            except UnicodeDecodeError as e:
                logger.warning(f"JSON解码失败(非UTF-8编码): {e}, 长度: {len(payload)}")
                # 尝试其他编码
                try:
                    resp.payload_msg = json.loads(payload.decode("gbk"))
                except Exception:
                    pass
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败: {e}, 长度: {len(payload)}")
            except Exception as e:
                logger.warning(f"JSON解析异常: {e}")

        return resp


# ================== ASR客户端 ==================
class ASRClient:
    """流式ASR客户端"""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or get_config().asr
        self.seq = 1
        self.session: Optional[aiohttp.ClientSession] = None
        self.conn: Optional[aiohttp.ClientWebSocketResponse] = None
        self._is_connected = False

    async def __aenter__(self) -> "ASRClient":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """关闭连接"""
        if self.conn and not self.conn.closed:
            await self.conn.close()
        if self.session and not self.session.closed:
            await self.session.close()
        self._is_connected = False

    def _build_auth_headers(self) -> Dict[str, str]:
        """构建认证头"""
        return {
            "X-Api-Resource-Id": "volc.bigasr.sauc.duration",
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Access-Key": self.config.access_key,
            "X-Api-App-Key": self.config.app_key,
        }

    def _build_full_request(self, seq: int) -> bytes:
        """构建初始化请求"""
        header = ASRRequestHeader().with_message_type(MessageType.CLIENT_FULL_REQUEST)

        payload = {
            "user": {"uid": "voice_dialog_user"},
            "audio": {
                "format": "pcm",
                "sample_rate": self.config.sample_rate,
                "bits": self.config.bits,
                "channel": self.config.channels,
                "codec": "raw",
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": self.config.enable_itn,
                "enable_punc": self.config.enable_punc,
                "enable_ddc": self.config.enable_ddc,
                "show_utterances": self.config.show_utterances,
                "enable_nonstream": False,
            },
        }

        payload_bytes = json.dumps(payload).encode("utf-8")
        compressed = gzip.compress(payload_bytes)

        buf = bytearray()
        buf.extend(header.to_bytes())
        buf.extend(struct.pack(">i", seq))
        buf.extend(struct.pack(">I", len(compressed)))
        buf.extend(compressed)
        return bytes(buf)

    def _build_audio_request(
        self, seq: int, audio_data: bytes, is_last: bool = False
    ) -> bytes:
        """构建音频请求"""
        header = ASRRequestHeader().with_message_type(
            MessageType.CLIENT_AUDIO_ONLY_REQUEST
        )

        if is_last:
            header.with_message_type_specific_flags(
                MessageTypeSpecificFlags.NEG_WITH_SEQUENCE
            )
            seq = -abs(seq)
        else:
            header.with_message_type_specific_flags(
                MessageTypeSpecificFlags.POS_SEQUENCE
            )

        buf = bytearray()
        buf.extend(header.to_bytes())
        buf.extend(struct.pack(">i", seq))

        compressed = gzip.compress(audio_data)
        buf.extend(struct.pack(">I", len(compressed)))
        buf.extend(compressed)
        return bytes(buf)

    async def connect(self) -> None:
        """建立WebSocket连接"""
        if self._is_connected:
            return

        headers = self._build_auth_headers()
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            self.conn = await self.session.ws_connect(
                self.config.url, headers=headers, heartbeat=30.0
            )
            self._is_connected = True
            logger.debug(f"ASR WebSocket已连接")
        except Exception as e:
            logger.error(f"ASR WebSocket连接失败: {e}")
            raise

    async def initialize(self) -> ASRResponse:
        """发送初始化请求"""
        if not self._is_connected:
            await self.connect()

        self.seq = 1
        req_bytes = self._build_full_request(self.seq)
        await self.conn.send_bytes(req_bytes)
        self.seq += 1

        # 等待初始化响应
        msg = await self.conn.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            resp = ResponseParser.parse_response(msg.data)
            logger.debug("ASR初始化成功")
            return resp
        else:
            logger.warning(f"ASR初始化响应异常: {msg.type}")
            return ASRResponse()

    async def send_audio(self, audio_data: bytes, is_last: bool = False) -> None:
        """发送音频数据"""
        if not self._is_connected:
            raise RuntimeError("WebSocket未连接")

        req = self._build_audio_request(self.seq, audio_data, is_last)
        await self.conn.send_bytes(req)
        self.seq += 1

    async def receive_responses(self) -> AsyncGenerator[ASRResponse, None]:
        """接收识别响应（流式）"""
        if not self._is_connected:
            raise RuntimeError("WebSocket未连接")

        try:
            async for msg in self.conn:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    resp = ResponseParser.parse_response(msg.data)
                    yield resp
                    if resp.is_last_package or resp.code != 0:
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.error(f"接收ASR响应出错: {e}")
            raise

    async def recognize_stream(
        self,
        audio_queue: asyncio.Queue,
        result_callback: Optional[Callable[[ASRResponse], None]] = None,
        definite_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式识别

        Args:
            audio_queue: 音频数据队列（每个元素为PCM bytes）
            result_callback: 每次收到响应的回调
            definite_callback: 收到确定句子的回调

        Yields:
            识别出的确定文本
        """
        await self.initialize()

        # 启动发送任务
        send_task = asyncio.create_task(self._send_audio_from_queue(audio_queue))

        try:
            async for resp in self.receive_responses():
                if result_callback:
                    result_callback(resp)

                # 检查是否有确定的句子
                definite_text = resp.get_definite_text()
                if definite_text:
                    if definite_callback:
                        definite_callback(definite_text)
                    yield definite_text

                if resp.is_last_package:
                    # 最后一包，返回完整文本
                    final_text = resp.get_text()
                    if final_text and final_text != definite_text:
                        if definite_callback:
                            definite_callback(final_text)
                        yield final_text
                    break
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    async def _send_audio_from_queue(self, audio_queue: asyncio.Queue) -> None:
        """从队列发送音频"""
        try:
            while True:
                data = await audio_queue.get()
                if data is None:  # None表示结束
                    await self.send_audio(b"", is_last=True)
                    break
                await self.send_audio(data, is_last=False)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"发送音频出错: {e}")


# ================== ASR 关键词检测器 ==================
class ASRKeywordDetector:
    """
    ASR 关键词检测器
    检测用户语音识别结果中的动作关键词并发送 UDP 指令
    """

    def __init__(self):
        self._udp_controller = get_udp_controller()
        self._config = get_config()

    def detect_and_emit(self, text: str) -> None:
        """
        检测 ASR 识别文本中的关键词并发送 UDP 指令

        Args:
            text: ASR 识别出的文本
        """
        if not text:
            return

        # 检查是否启用关键词检测
        if not self._config.enable_keyword_detection:
            return

        # 遍历 ASR_KWS_PATTERNS，检测关键短语
        for keyword, patterns in ASR_KWS_PATTERNS.items():
            matched_pattern = next((p for p in patterns if p in text), None)
            if matched_pattern:
                self._udp_controller.emit_voice_keyword(keyword)
                logger.info(
                    f"[ASR-KWS] 匹配: '{matched_pattern}' → 动作: {keyword} | UDP已发送"
                )
                # 找到一个就触发，可以 break 或继续检测其他
                break


# 全局 ASR 关键词检测器
_asr_kw_detector: Optional[ASRKeywordDetector] = None


def get_asr_keyword_detector() -> ASRKeywordDetector:
    """获取全局 ASR 关键词检测器"""
    global _asr_kw_detector
    if _asr_kw_detector is None:
        _asr_kw_detector = ASRKeywordDetector()
    return _asr_kw_detector


# ================== 实时ASR会话 ==================
class RealtimeASRSession:
    """
    实时ASR会话管理器
    支持从音频队列持续识别，并将结果推送到输出队列
    支持 ASR 关键词检测并发送 UDP 动作指令
    """

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        config: Optional[ASRConfig] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config or get_config().asr
        self.client: Optional[ASRClient] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # ASR 关键词检测器
        self._kw_detector = get_asr_keyword_detector()

        # UDP 控制器（用于发送收麦克风信号）
        self._udp_controller = get_udp_controller()

    async def start(self) -> None:
        """启动ASR会话"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.debug("ASR会话已启动")

    async def stop(self) -> None:
        """停止ASR会话"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.client:
            await self.client.close()
        logger.debug("ASR会话已停止")

    def _on_asr_result(self, text: str) -> None:
        """ASR 识别结果回调，进行关键词检测"""
        logger.debug(f"ASR识别中间结果: {text}")
        # 检测关键词并发送 UDP
        self._kw_detector.detect_and_emit(text)

    async def _run(self) -> None:
        """运行识别循环"""
        async with ASRClient(self.config) as client:
            self.client = client

            while self._running:
                try:
                    async for text in client.recognize_stream(
                        self.input_queue, definite_callback=self._on_asr_result
                    ):
                        if text:
                            # 发送收麦克风信号（用户说完话了）
                            self._udp_controller.send_mic_command("release_microphone")

                            await self.output_queue.put(text)
                            logger.info(f"[ASR] 用户: {text}")
                except Exception as e:
                    logger.error(f"ASR识别出错: {e}")
                    if self._running:
                        await asyncio.sleep(1)  # 重试前等待
                        await client.close()
                        await client.connect()
