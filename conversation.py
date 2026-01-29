# -*- coding: utf-8 -*-
"""
对话管理模块 - 状态机、打断处理、多轮对话
实现完整的语音对话流程控制
"""

import asyncio
import audioop
import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from config import SystemConfig, get_config

logger = logging.getLogger(__name__)


class DialogState(Enum):
    """对话状态"""

    IDLE = auto()  # 空闲，等待用户说话
    LISTENING = auto()  # 正在听取用户输入
    PROCESSING = auto()  # 正在处理（ASR识别中）
    THINKING = auto()  # LLM思考中
    SPEAKING = auto()  # TTS播放中
    INTERRUPTED = auto()  # 被打断


class DialogEvent(Enum):
    """对话事件"""

    VOICE_START = auto()  # 检测到用户开始说话
    VOICE_END = auto()  # 用户说话结束
    ASR_RESULT = auto()  # ASR识别结果
    LLM_START = auto()  # LLM开始生成
    LLM_SENTENCE = auto()  # LLM生成一句话
    LLM_END = auto()  # LLM生成结束
    TTS_START = auto()  # TTS开始播放
    TTS_END = auto()  # TTS播放结束
    BARGE_IN = auto()  # 用户打断
    ERROR = auto()  # 发生错误
    RESET = auto()  # 重置对话


class DialogManager:
    """
    对话管理器
    管理对话状态、处理打断、协调各模块
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        self.state = DialogState.IDLE
        self._state_lock = asyncio.Lock()

        # 事件回调
        self._callbacks: Dict[DialogEvent, List[Callable]] = {
            event: [] for event in DialogEvent
        }

        # 打断检测
        self._is_speaking = False
        self._barge_in_enabled = self.config.enable_barge_in

        # 统计信息
        self._stats = {
            "total_turns": 0,
            "total_interrupts": 0,
            "avg_response_time": 0.0,
            "last_turn_time": 0.0,
        }

        # 当前轮次计时
        self._turn_start_time: Optional[float] = None

    @property
    def current_state(self) -> DialogState:
        """当前状态"""
        return self.state

    @property
    def is_speaking(self) -> bool:
        """是否正在说话"""
        return self._is_speaking

    def register_callback(
        self,
        event: DialogEvent,
        callback: Callable,
    ) -> None:
        """
        注册事件回调

        Args:
            event: 事件类型
            callback: 回调函数
        """
        self._callbacks[event].append(callback)

    def unregister_callback(
        self,
        event: DialogEvent,
        callback: Callable,
    ) -> None:
        """注销事件回调"""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    async def _trigger_event(
        self,
        event: DialogEvent,
        data: Any = None,
    ) -> None:
        """触发事件"""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"事件回调出错 [{event}]: {e}")

    async def transition_to(self, new_state: DialogState) -> bool:
        """
        状态转换

        Args:
            new_state: 目标状态

        Returns:
            是否转换成功
        """
        async with self._state_lock:
            old_state = self.state

            # 验证状态转换是否合法
            if not self._is_valid_transition(old_state, new_state):
                logger.warning(f"非法状态转换: {old_state} -> {new_state}")
                return False

            self.state = new_state
            logger.debug(f"[状态] {old_state.name} → {new_state.name}")

            # 特殊处理
            if new_state == DialogState.SPEAKING:
                self._is_speaking = True
            elif old_state == DialogState.SPEAKING:
                self._is_speaking = False

            if new_state == DialogState.LISTENING:
                self._turn_start_time = time.time()

            return True

    def _is_valid_transition(
        self,
        from_state: DialogState,
        to_state: DialogState,
    ) -> bool:
        """检查状态转换是否合法"""
        # 允许的状态转换
        valid_transitions = {
            DialogState.IDLE: {DialogState.LISTENING, DialogState.IDLE},
            DialogState.LISTENING: {DialogState.PROCESSING, DialogState.IDLE},
            DialogState.PROCESSING: {
                DialogState.THINKING,
                DialogState.IDLE,
                DialogState.LISTENING,
            },
            DialogState.THINKING: {
                DialogState.SPEAKING,
                DialogState.INTERRUPTED,
                DialogState.IDLE,
            },
            DialogState.SPEAKING: {
                DialogState.IDLE,
                DialogState.INTERRUPTED,
                DialogState.LISTENING,
            },
            DialogState.INTERRUPTED: {DialogState.LISTENING, DialogState.IDLE},
        }

        return to_state in valid_transitions.get(from_state, set())

    async def handle_voice_start(self) -> None:
        """处理用户开始说话"""
        if self.state == DialogState.IDLE:
            await self.transition_to(DialogState.LISTENING)
            await self._trigger_event(DialogEvent.VOICE_START)
        elif self.state == DialogState.SPEAKING and self._barge_in_enabled:
            # 打断
            await self.handle_barge_in()

    async def handle_voice_end(self) -> None:
        """处理用户说话结束"""
        if self.state == DialogState.LISTENING:
            await self.transition_to(DialogState.PROCESSING)
            await self._trigger_event(DialogEvent.VOICE_END)

    async def handle_asr_result(self, text: str) -> None:
        """处理ASR识别结果"""
        if self.state == DialogState.PROCESSING:
            await self.transition_to(DialogState.THINKING)
            await self._trigger_event(DialogEvent.ASR_RESULT, text)

    async def handle_llm_sentence(self, sentence: str) -> None:
        """处理LLM生成的句子"""
        if self.state == DialogState.THINKING:
            await self.transition_to(DialogState.SPEAKING)
        await self._trigger_event(DialogEvent.LLM_SENTENCE, sentence)

    async def handle_llm_end(self) -> None:
        """处理LLM生成结束"""
        await self._trigger_event(DialogEvent.LLM_END)

    async def handle_tts_end(self) -> None:
        """处理TTS播放结束"""
        if self.state == DialogState.SPEAKING:
            await self.transition_to(DialogState.IDLE)

            # 更新统计
            if self._turn_start_time:
                turn_time = time.time() - self._turn_start_time
                self._stats["last_turn_time"] = turn_time
                self._stats["total_turns"] += 1

                # 更新平均响应时间
                n = self._stats["total_turns"]
                avg = self._stats["avg_response_time"]
                self._stats["avg_response_time"] = (avg * (n - 1) + turn_time) / n

                self._turn_start_time = None

            await self._trigger_event(DialogEvent.TTS_END)

    async def handle_barge_in(self) -> None:
        """处理用户打断"""
        if self.state in (DialogState.THINKING, DialogState.SPEAKING):
            await self.transition_to(DialogState.INTERRUPTED)
            self._stats["total_interrupts"] += 1
            await self._trigger_event(DialogEvent.BARGE_IN)

            # 短暂延迟后转为监听
            await asyncio.sleep(0.1)
            await self.transition_to(DialogState.LISTENING)

    async def reset(self) -> None:
        """重置对话状态"""
        async with self._state_lock:
            self.state = DialogState.IDLE
            self._is_speaking = False
            self._turn_start_time = None

        await self._trigger_event(DialogEvent.RESET)
        logger.debug("对话已重置")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()

    def set_barge_in_enabled(self, enabled: bool) -> None:
        """设置是否启用打断"""
        self._barge_in_enabled = enabled


# ================== 打断检测器 ==================
class BargeInDetector:
    """
    打断检测器
    在TTS播放时检测用户是否要打断
    """

    def __init__(
        self,
        threshold: int = 600,
        min_duration_ms: int = 100,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ):
        self.threshold = threshold
        self.min_duration_ms = min_duration_ms
        self.sample_rate = sample_rate
        self.sample_width = sample_width

        self._consecutive_frames = 0
        self._frames_needed = int(
            min_duration_ms * sample_rate / 1000 / 320
        )  # 假设每帧320采样

    def detect(self, audio_data: bytes) -> bool:
        """
        检测是否有打断意图

        Args:
            audio_data: 麦克风音频数据

        Returns:
            是否检测到打断
        """
        if not audio_data:
            self._consecutive_frames = 0
            return False

        rms = audioop.rms(audio_data, self.sample_width)

        if rms > self.threshold:
            self._consecutive_frames += 1
            if self._consecutive_frames >= self._frames_needed:
                self._consecutive_frames = 0
                return True
        else:
            self._consecutive_frames = 0

        return False

    def reset(self) -> None:
        """重置检测器"""
        self._consecutive_frames = 0

    def set_threshold(self, threshold: int) -> None:
        """设置阈值"""
        self.threshold = threshold


# ================== 对话会话 ==================
class DialogSession:
    """
    完整对话会话
    整合所有模块，实现端到端语音对话
    """

    def __init__(
        self,
        audio_queue: asyncio.Queue,  # 麦克风输入
        asr_queue: asyncio.Queue,  # ASR输出/LLM输入
        llm_queue: asyncio.Queue,  # LLM输出/TTS输入
        tts_queue: asyncio.Queue,  # TTS输出/播放输入
        config: Optional[SystemConfig] = None,
    ):
        self.audio_queue = audio_queue
        self.asr_queue = asr_queue
        self.llm_queue = llm_queue
        self.tts_queue = tts_queue
        self.config = config or get_config()

        self.dialog_manager = DialogManager(self.config)
        self.barge_in_detector = BargeInDetector(
            threshold=self.config.barge_in_threshold,
        )

        self._running = False
        self._tasks: List[asyncio.Task] = []

        # 模块引用（由外部注入）
        self.asr_session = None
        self.llm_session = None
        self.tts_session = None
        self.audio_player = None
        self.audio_capture = None

    def set_modules(
        self,
        asr_session=None,
        llm_session=None,
        tts_session=None,
        audio_player=None,
        audio_capture=None,
    ) -> None:
        """设置模块引用"""
        self.asr_session = asr_session
        self.llm_session = llm_session
        self.tts_session = tts_session
        self.audio_player = audio_player
        self.audio_capture = audio_capture

        # 注册打断回调
        if self.tts_session:
            self.dialog_manager.register_callback(
                DialogEvent.BARGE_IN, lambda e, d: self.tts_session.interrupt()
            )

        if self.audio_player:
            self.dialog_manager.register_callback(
                DialogEvent.BARGE_IN, lambda e, d: self.audio_player.interrupt()
            )

    async def start(self) -> None:
        """启动对话会话"""
        if self._running:
            return

        self._running = True

        # 启动打断检测任务
        if self.config.enable_barge_in:
            task = asyncio.create_task(self._barge_in_monitor())
            self._tasks.append(task)

        logger.debug("对话会话已启动")

    async def stop(self) -> None:
        """停止对话会话"""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        await self.dialog_manager.reset()

        logger.debug("对话会话已停止")

    async def _barge_in_monitor(self) -> None:
        """打断监控任务"""
        while self._running:
            try:
                if self.dialog_manager.is_speaking:
                    # 从音频队列复制一份检测（不消费原始数据）
                    # 这里简化处理，实际应该使用独立的监控流
                    pass

                await asyncio.sleep(0.05)  # 50ms检测间隔

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"打断监控出错: {e}")

    def switch_model(self, model_name: str) -> bool:
        """切换LLM模型"""
        if self.llm_session:
            return self.llm_session.switch_model(model_name)
        return False

    def clear_history(self) -> None:
        """清空对话历史"""
        if self.llm_session:
            self.llm_session.clear_history()

    def get_stats(self) -> Dict[str, Any]:
        """获取对话统计"""
        return self.dialog_manager.get_stats()


# ================== 实用函数 ==================
def create_dialog_queues() -> Dict[str, asyncio.Queue]:
    """创建对话所需的队列"""
    return {
        "audio": asyncio.Queue(),  # 麦克风 -> ASR
        "asr": asyncio.Queue(),  # ASR -> LLM
        "llm": asyncio.Queue(),  # LLM -> TTS
        "tts": asyncio.Queue(),  # TTS -> 播放
    }
