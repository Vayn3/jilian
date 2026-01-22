# -*- coding: utf-8 -*-
"""
TTS客户端模块 - 讯飞语音合成（武汉话）
基于WebSocket流式合成，支持实时音频输出
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from typing import AsyncGenerator, Callable, Optional, Any
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websockets

from config import get_config, TTSConfig

logger = logging.getLogger(__name__)


class TTSClient:
    """讯飞TTS流式客户端"""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self._ws = None
    
    def _create_url(self) -> str:
        """生成带鉴权的WebSocket URL"""
        url = self.config.url
        
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        # 拼接签名字符串
        signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/tts HTTP/1.1"
        
        # HMAC-SHA256签名
        signature_sha = hmac.new(
            self.config.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
        
        # 构建authorization
        authorization_origin = (
            f'api_key="{self.config.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature_sha_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        # 拼接URL
        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        
        return f"{url}?{urlencode(params)}"
    
    def _build_request(self, text: str) -> str:
        """构建TTS请求"""
        # 文本编码
        text_base64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        
        request = {
            "common": {
                "app_id": self.config.app_id
            },
            "business": {
                "aue": self.config.aue,
                "auf": self.config.auf,
                "vcn": self.config.vcn,  # 武汉话发音人
                "tte": self.config.tte,
                "speed": self.config.speed,
                "volume": self.config.volume,
                "pitch": self.config.pitch,
            },
            "data": {
                "status": 2,  # 一次性发送全部文本
                "text": text_base64
            }
        }
        
        return json.dumps(request)
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        流式合成语音
        
        Args:
            text: 要合成的文本
        
        Yields:
            PCM音频数据块
        """
        if not text or not text.strip():
            return
        
        url = self._create_url()
        
        try:
            async with websockets.connect(url, ssl=True) as ws:
                # 发送合成请求
                request = self._build_request(text)
                await ws.send(request)
                logger.info(f"TTS发送文本: {text[:50]}...")
                
                # 接收音频数据
                while True:
                    try:
                        response = await ws.recv()
                        result = json.loads(response)
                        
                        code = result.get("code", -1)
                        if code != 0:
                            error_msg = result.get("message", "未知错误")
                            logger.error(f"TTS合成错误: {code} - {error_msg}")
                            break
                        
                        data = result.get("data", {})
                        audio_base64 = data.get("audio")
                        status = data.get("status", 0)
                        
                        if audio_base64:
                            audio_bytes = base64.b64decode(audio_base64)
                            yield audio_bytes
                        
                        if status == 2:  # 合成结束
                            logger.info("TTS合成完成")
                            break
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("TTS WebSocket连接已关闭")
                        break
                        
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            raise
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        合成语音并返回完整音频数据
        
        Args:
            text: 要合成的文本
        
        Returns:
            完整PCM音频数据
        """
        audio_chunks = []
        async for chunk in self.synthesize(text):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)


# ================== 连接池管理 ==================
class TTSConnectionPool:
    """
    TTS连接池
    预建立连接以降低延迟
    """
    
    def __init__(self, pool_size: int = 2, config: Optional[TTSConfig] = None):
        self.pool_size = pool_size
        self.config = config or get_config().tts
        self._available: asyncio.Queue = asyncio.Queue()
        self._all_connections: list = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化连接池"""
        if self._initialized:
            return
        
        # 由于讯飞TTS是短连接，这里主要是预热客户端
        for _ in range(self.pool_size):
            client = TTSClient(self.config)
            await self._available.put(client)
            self._all_connections.append(client)
        
        self._initialized = True
        logger.info(f"TTS连接池已初始化，大小: {self.pool_size}")
    
    async def acquire(self) -> TTSClient:
        """获取一个TTS客户端"""
        if not self._initialized:
            await self.initialize()
        return await self._available.get()
    
    async def release(self, client: TTSClient) -> None:
        """释放TTS客户端"""
        await self._available.put(client)
    
    async def close(self) -> None:
        """关闭连接池"""
        self._all_connections.clear()
        while not self._available.empty():
            await self._available.get()
        self._initialized = False


# ================== 实时TTS会话 ==================
class RealtimeTTSSession:
    """
    实时TTS会话管理器
    从输入队列读取文本，流式合成音频，推送到输出队列
    """
    
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        config: Optional[TTSConfig] = None,
        on_tts_start: Optional[Callable[[str], Any]] = None,
        on_tts_end: Optional[Callable[[], Any]] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client = TTSClient(config)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._interrupted = asyncio.Event()
        self._on_tts_start = on_tts_start
        self._on_tts_end = on_tts_end
    
    async def start(self) -> None:
        """启动TTS会话"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("TTS会话已启动")
    
    async def stop(self) -> None:
        """停止TTS会话"""
        self._running = False
        self._interrupted.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TTS会话已停止")
    
    def interrupt(self) -> None:
        """打断当前合成"""
        self._interrupted.set()
        logger.info("TTS合成被打断")
    
    def resume(self) -> None:
        """恢复合成"""
        self._interrupted.clear()
    
    async def _run(self) -> None:
        """运行处理循环"""
        while self._running:
            try:
                # 等待文本输入
                text = await self.input_queue.get()
                
                if text is None:  # 一轮对话结束标记
                    # 发送结束标记到音频队列
                    await self.output_queue.put(None)
                    continue
                
                if self._interrupted.is_set():
                    continue
                
                logger.info(f"TTS开始合成: {text[:30]}...")
                # 通知开始（用于状态机切换到SPEAKING）
                if self._on_tts_start:
                    try:
                        res = self._on_tts_start(text)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        logger.warning(f"TTS开始回调出错: {e}")
                
                # 流式合成
                async for audio_chunk in self.client.synthesize(text):
                    if self._interrupted.is_set():
                        logger.info("TTS合成被打断，停止输出")
                        break
                    
                    await self.output_queue.put(audio_chunk)
                
                # 合成结束，通知状态机
                if self._on_tts_end:
                    try:
                        res = self._on_tts_end()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        logger.warning(f"TTS结束回调出错: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS处理出错: {e}")


# ================== 文本预处理 ==================
class TextPreprocessor:
    """
    TTS文本预处理器
    处理不适合语音合成的特殊字符
    """
    
    # 需要替换的字符映射
    REPLACEMENTS = {
        '\n': '，',
        '\r': '',
        '\t': ' ',
        '...': '，',
        '……': '，',
        '~': '',
        '～': '',
        '*': '',
        '#': '',
        '@': '',
        '&': '和',
        '%': '百分之',
    }
    
    @classmethod
    def process(cls, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
        
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # 替换特殊字符
        for old, new in cls.REPLACEMENTS.items():
            text = text.replace(old, new)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        # 移除不可打印字符
        text = ''.join(c for c in text if c.isprintable() or c in ' \n')
        
        return text.strip()
    
    @classmethod
    def split_long_text(cls, text: str, max_length: int = 500) -> list:
        """
        切分过长文本
        讯飞TTS单次最大支持约2000汉字
        
        Args:
            text: 原始文本
            max_length: 最大长度
        
        Returns:
            切分后的文本列表
        """
        if len(text) <= max_length:
            return [text]
        
        result = []
        current = ""
        
        # 按句子切分
        import re
        sentences = re.split(r'([。！？.!?])', text)
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
            
            if len(current) + len(sentence) <= max_length:
                current += sentence
            else:
                if current:
                    result.append(current)
                current = sentence
        
        # 处理最后一个
        if i + 2 < len(sentences):
            current += sentences[-1]
        
        if current:
            result.append(current)
        
        return result
