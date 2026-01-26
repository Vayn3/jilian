# -*- coding: utf-8 -*-
"""
级联式武汉话实时语音对话系统 - 主入口
豆包ASR → 千问LLM → 讯飞TTS（武汉话）
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from asr_client import ASRClient, ASRResponse
from audio_manager import AudioCapture, AudioPlayer, RealtimeAudioPlaySession, SimpleVAD
from config import SystemConfig, get_config
from conversation import DialogEvent, DialogManager, DialogState, create_dialog_queues
from llm_client import LLMClient, RAGInterface, RealtimeLLMSession, SimpleRAG
from tts_client import RealtimeTTSSession, TextPreprocessor, TTSClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VoiceDialogSystem:
    """
    武汉话实时语音对话系统
    整合ASR、LLM、TTS三个模块，实现流式语音对话
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()

        # 创建队列
        self.queues = create_dialog_queues()
        self.audio_queue = self.queues["audio"]  # 麦克风 -> VAD/ASR
        self.asr_queue = self.queues["asr"]  # ASR -> LLM
        self.llm_queue = self.queues["llm"]  # LLM -> TTS
        self.tts_queue = self.queues["tts"]  # TTS -> 播放

        # 对话管理
        self.dialog_manager = DialogManager(self.config)
        self.vad = SimpleVAD(threshold=self.config.asr.vad_silence_threshold)

        # 组件（延迟初始化）
        self.audio_capture: Optional[AudioCapture] = None
        self.audio_player: Optional[RealtimeAudioPlaySession] = None
        self.asr_client: Optional[ASRClient] = None
        self.llm_session: Optional[RealtimeLLMSession] = None
        self.tts_session: Optional[RealtimeTTSSession] = None

        # RAG（可选）
        self.rag: Optional[RAGInterface] = None

        # 运行状态
        self._running = False
        self._tasks = []

        # 当前对话状态
        self._current_text = ""
        self._is_listening = False
        self._silence_frames = 0

    def set_rag(self, rag: RAGInterface) -> None:
        """设置RAG实现"""
        self.rag = rag
        if self.llm_session:
            self.llm_session.set_rag(rag)
        self.config.rag.enabled = True
        logger.info("RAG已启用")

    def switch_model(self, model_name: str) -> bool:
        """
        切换LLM模型

        Args:
            model_name: 模型名称 (qwen-flash, qwen-turbo, qwen-plus, qwen-max)

        Returns:
            是否切换成功
        """
        if self.config.llm.switch_model(model_name):
            if self.llm_session:
                self.llm_session.client.config.model = model_name
            logger.info(f"已切换到模型: {model_name}")
            return True
        logger.warning(f"不支持的模型: {model_name}")
        return False

    def _init_components(self) -> None:
        """初始化组件"""
        loop = asyncio.get_running_loop()

        # 音频采集（带回声消除）
        self.audio_capture = AudioCapture(
            output_queue=self.audio_queue,
            config=self.config.audio,
            loop=loop,
        )

        # 音频播放（提供AEC参考信号回调）
        self.audio_player = RealtimeAudioPlaySession(
            input_queue=self.tts_queue,
            config=self.config.audio,
            aec_callback=(
                self.audio_capture.update_playback_reference
                if self.audio_capture
                else None
            ),
        )

        # ASR客户端
        self.asr_client = ASRClient(self.config.asr)

        # LLM会话
        self.llm_session = RealtimeLLMSession(
            input_queue=self.asr_queue,
            output_queue=self.llm_queue,
            config=self.config.llm,
            rag=self.rag,
        )

        # TTS会话
        self.tts_session = RealtimeTTSSession(
            input_queue=self.llm_queue,
            output_queue=self.tts_queue,
            config=self.config.tts,
            on_tts_start=self._on_tts_start,
            on_tts_end=self._on_tts_end,
        )

        # 注册打断回调
        self.dialog_manager.register_callback(DialogEvent.BARGE_IN, self._on_barge_in)

    async def _on_barge_in(self, event: DialogEvent, data) -> None:
        """打断回调"""
        logger.info("检测到用户打断")
        if self.tts_session:
            self.tts_session.interrupt()
        if self.audio_player:
            self.audio_player.interrupt()

    async def _on_tts_start(self, text: str) -> None:
        """TTS开始时通知状态机进入SPEAKING"""
        await self.dialog_manager.handle_llm_sentence(text)

    async def _on_tts_end(self) -> None:
        """TTS结束时回到IDLE"""
        await self.dialog_manager.handle_tts_end()

    async def start(self) -> None:
        """启动系统"""
        if self._running:
            return

        logger.info("=" * 50)
        logger.info("武汉话实时语音对话系统启动中...")
        logger.info("=" * 50)

        self._running = True
        self._init_components()

        # 启动音频采集
        self.audio_capture.start()

        # 启动各个会话
        await self.llm_session.start()
        await self.tts_session.start()
        await self.audio_player.start()

        # 启动主处理循环
        self._tasks = [
            asyncio.create_task(self._vad_loop()),
            asyncio.create_task(self._asr_loop()),
        ]

        logger.info("-" * 50)
        logger.info(f"模型: {self.config.llm.model} | TTS: {self.config.tts.vcn}")
        logger.info(
            f"输出模式: {self.config.audio.output_mode} | 关键词检测: {'开' if self.config.enable_keyword_detection else '关'}"
        )
        logger.info("-" * 50)
        logger.info("✓ 系统就绪，请开始说话...")

    async def stop(self) -> None:
        """停止系统"""
        if not self._running:
            return

        logger.info("正在停止系统...")
        self._running = False

        # 取消任务
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # 停止组件
        if self.audio_capture:
            self.audio_capture.stop()
        if self.audio_player:
            await self.audio_player.stop()
        if self.llm_session:
            await self.llm_session.stop()
        if self.tts_session:
            await self.tts_session.stop()
        if self.asr_client:
            await self.asr_client.close()

        logger.info("系统已停止")

    async def _vad_loop(self) -> None:
        """VAD检测循环"""
        max_silence_frames = int(
            self.config.asr.max_silence_ms / self.config.audio.input_chunk_ms
        )

        while self._running:
            try:
                # 获取音频帧
                audio_frame = await self.audio_queue.get()

                if not audio_frame:
                    continue

                # VAD检测
                is_speech = self.vad.is_speech(audio_frame)

                # 检查是否需要打断
                if self.dialog_manager.is_speaking and is_speech:
                    if self.config.enable_barge_in:
                        await self.dialog_manager.handle_barge_in()
                        self.tts_session.resume()
                        self.audio_player.resume()

                # 状态处理
                if self.dialog_manager.current_state == DialogState.IDLE:
                    if is_speech:
                        # 开始说话
                        await self.dialog_manager.handle_voice_start()
                        self._is_listening = True
                        self._silence_frames = 0
                        self._current_audio_buffer = [audio_frame]

                elif self.dialog_manager.current_state == DialogState.LISTENING:
                    if is_speech:
                        self._silence_frames = 0
                        self._current_audio_buffer.append(audio_frame)
                    else:
                        self._silence_frames += 1
                        self._current_audio_buffer.append(audio_frame)

                        # 检查是否说话结束
                        if self._silence_frames >= max_silence_frames:
                            await self.dialog_manager.handle_voice_end()
                            self._is_listening = False

                            # 将音频发送到ASR
                            audio_data = b"".join(self._current_audio_buffer)
                            await self._process_audio(audio_data)
                            self._current_audio_buffer = []

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"VAD循环出错: {e}")

    async def _process_audio(self, audio_data: bytes) -> None:
        """处理录制的音频"""
        if not audio_data:
            await self.dialog_manager.reset()
            return

        # 重采样：从麦克风采样率转换到ASR期望的采样率
        if self.config.audio.sample_rate != self.config.asr.sample_rate:
            try:
                # 转换为numpy数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # 计算重采样比例
                resample_ratio = (
                    self.config.asr.sample_rate / self.config.audio.sample_rate
                )
                new_length = int(len(audio_array) * resample_ratio)

                # 使用scipy进行重采样
                resampled_array = scipy_signal.resample(audio_array, new_length)

                # 转换回int16并转为bytes
                audio_data = resampled_array.astype(np.int16).tobytes()
            except Exception as e:
                logger.error(f"重采样失败: {e}")
                await self.dialog_manager.reset()
                return

        try:
            async with ASRClient(self.config.asr) as client:
                await client.initialize()

                # 分片发送音频（使用ASR采样率计算）
                segment_size = int(
                    self.config.asr.sample_rate
                    * self.config.audio.sample_width
                    * self.config.asr.segment_duration_ms
                    / 1000
                )

                segments = [
                    audio_data[i : i + segment_size]
                    for i in range(0, len(audio_data), segment_size)
                ]

                for i, segment in enumerate(segments):
                    is_last = i == len(segments) - 1
                    await client.send_audio(segment, is_last)

                # 接收识别结果
                final_text = ""
                async for resp in client.receive_responses():
                    if resp.code != 0:
                        err_detail = None
                        if resp.payload_msg:
                            err_detail = (
                                resp.payload_msg.get("message")
                                or resp.payload_msg.get("msg")
                                or resp.payload_msg
                            )
                        if err_detail:
                            logger.error(f"ASR错误: {resp.code}, 详情: {err_detail}")
                        else:
                            logger.error(f"ASR错误: {resp.code}")
                        break

                    text = resp.get_text()

                    if resp.is_last_package:
                        final_text = resp.get_text()
                        break

                if final_text:
                    logger.info(f"[用户] {final_text}")
                    await self.dialog_manager.handle_asr_result(final_text)

                    # 发送到LLM
                    await self.asr_queue.put(final_text)
                else:
                    await self.dialog_manager.reset()

        except Exception as e:
            logger.error(f"ASR处理出错: {e}")
            await self.dialog_manager.reset()

    async def _asr_loop(self) -> None:
        """ASR结果处理循环（监控LLM输出触发状态更新）"""
        while self._running:
            try:
                await asyncio.sleep(0.1)

                # 检查TTS队列是否有结束标记
                if self.dialog_manager.current_state == DialogState.SPEAKING:
                    # 简单检测播放是否结束
                    if (
                        self.tts_queue.empty()
                        and not self.audio_player.player.is_playing
                    ):
                        await self.dialog_manager.handle_tts_end()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR循环出错: {e}")

    def list_audio_devices(self) -> None:
        """列出可用音频设备"""
        import pyaudio

        pa = pyaudio.PyAudio()

        print("\n=== 输入设备 ===")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(
                    f"  [{i}] {info['name']} (采样率: {int(info['defaultSampleRate'])})"
                )

        print("\n=== 输出设备 ===")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                print(
                    f"  [{i}] {info['name']} (采样率: {int(info['defaultSampleRate'])})"
                )

        pa.terminate()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.dialog_manager.get_stats()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="武汉话实时语音对话系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 使用默认配置启动
  python main.py --model qwen-plus  # 使用qwen-plus模型
  python main.py --list-devices     # 列出音频设备
  python main.py --input-device 1   # 指定输入设备
        """,
    )

    # 模型选择
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen-flash",
        choices=["qwen-flash", "qwen-turbo", "qwen-plus", "qwen-max"],
        help="LLM模型 (默认: qwen-flash，更快；qwen-plus更准确)",
    )

    # 音频设备
    parser.add_argument(
        "--list-devices", "-l", action="store_true", help="列出可用音频设备"
    )
    parser.add_argument(
        "--input-device", "-i", type=int, default=None, help="输入设备索引"
    )
    parser.add_argument(
        "--output-device", "-o", type=int, default=None, help="输出设备索引"
    )

    # VAD参数
    parser.add_argument(
        "--vad-threshold", type=int, default=500, help="VAD能量阈值 (默认: 500)"
    )
    parser.add_argument(
        "--silence-ms", type=int, default=500, help="静音判停时间(ms) (默认: 500)"
    )

    # 功能开关
    parser.add_argument("--no-aec", action="store_true", help="禁用回声消除")
    parser.add_argument("--no-barge-in", action="store_true", help="禁用打断功能")

    # 日志级别
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 列出设备
    if args.list_devices:
        system = VoiceDialogSystem()
        system.list_audio_devices()
        return

    # 更新配置
    config = get_config()
    config.llm.model = args.model
    config.audio.input_device_index = args.input_device
    config.audio.output_device_index = args.output_device
    config.asr.vad_silence_threshold = args.vad_threshold
    config.asr.max_silence_ms = args.silence_ms
    config.audio.enable_aec = not args.no_aec
    config.enable_barge_in = not args.no_barge_in

    # 创建系统
    system = VoiceDialogSystem(config)

    # 设置信号处理
    loop = asyncio.get_running_loop()

    async def shutdown():
        logger.info("收到退出信号...")
        await system.stop()

    # Windows不支持add_signal_handler，使用try-except
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    except NotImplementedError:
        # Windows fallback
        pass

    try:
        await system.start()

        # 保持运行
        while system._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
