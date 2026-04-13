# -*- coding: utf-8 -*-
"""级联式实时语音对话系统主入口。"""

import argparse
import asyncio
import logging
import signal
import sys
from collections import deque
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from asr_client import ASRClient, get_asr_keyword_detector
from audio_manager import AudioCapture, SimpleVAD
from config import SystemConfig, get_config
from conversation import (
    BargeInDetector,
    DialogEvent,
    DialogManager,
    DialogState,
    create_dialog_queues,
)
from dialog_logger import DialogFileLogger
from dialog_messages import LLMSentence, TTSChunk, UserTurn, WELCOME_UTTERANCE_ID
from duplex_sessions import RealtimeAudioPlaySession, RealtimeLLMSession, RealtimeTTSSession
from llm_client import RAGInterface
from ros_dialog import Ros1DialogTextPublisher
from ros_dialog import is_ros_available as is_ros_dialog_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("rosout").setLevel(logging.WARNING)
logging.getLogger("rospy").setLevel(logging.WARNING)
logging.getLogger("rospy.internal").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class VoiceDialogSystem:
    """整合 ASR/LLM/TTS 的实时对话系统。"""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()

        self.queues = create_dialog_queues()
        self.audio_queue = self.queues["audio"]
        self.asr_queue = self.queues["asr"]
        self.llm_queue = self.queues["llm"]
        self.tts_queue = self.queues["tts"]

        self.dialog_manager = DialogManager(self.config)
        self.vad = SimpleVAD(threshold=self.config.asr.vad_silence_threshold)
        self.barge_in_detector = BargeInDetector(
            threshold=self.config.barge_in_threshold,
            min_duration_ms=self.config.barge_in_min_duration_ms,
            sample_rate=self.config.audio.sample_rate,
            sample_width=self.config.audio.sample_width,
            frame_duration_ms=self.config.audio.input_chunk_ms,
        )
        self._asr_kw_detector = get_asr_keyword_detector()

        self.audio_capture: Optional[AudioCapture] = None
        self.audio_player: Optional[RealtimeAudioPlaySession] = None
        self.llm_session: Optional[RealtimeLLMSession] = None
        self.tts_session: Optional[RealtimeTTSSession] = None
        self.dialog_logger: Optional[DialogFileLogger] = None
        self.dialog_text_pub: Optional[Ros1DialogTextPublisher] = None

        self.rag: Optional[RAGInterface] = None

        self._running = False
        self._tasks = []
        self._background_tasks = set()

        self._is_listening = False
        self._silence_frames = 0
        self._current_audio_buffer = []
        self._preroll_frames = deque(
            maxlen=max(
                1,
                int(
                    self.config.barge_in_preroll_ms
                    / max(1, self.config.audio.input_chunk_ms)
                ),
            )
        )
        self._next_utterance_id = 1
        self._active_robot_utterance_id: Optional[int] = None

    def set_rag(self, rag: RAGInterface) -> None:
        self.rag = rag
        if self.llm_session:
            self.llm_session.set_rag(rag)
        self.config.rag.enabled = True
        logger.info("RAG enabled")

    def switch_model(self, model_name: str) -> bool:
        if self.config.llm.switch_model(model_name):
            if self.llm_session:
                self.llm_session.switch_model(model_name)
            logger.info("Switched model to %s", model_name)
            return True
        logger.warning("Unsupported model: %s", model_name)
        return False

    def _allocate_utterance_id(self) -> int:
        utterance_id = self._next_utterance_id
        self._next_utterance_id += 1
        return utterance_id

    def _is_utterance_active(self, utterance_id: int) -> bool:
        return (
            self._active_robot_utterance_id is not None
            and int(utterance_id) == int(self._active_robot_utterance_id)
        )

    def _track_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _drain_async_queue(self, queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def interrupt_current_utterance(self, reason: str = "barge_in") -> None:
        utterance_id = self._active_robot_utterance_id
        if utterance_id is None:
            return

        logger.info(
            "Interrupt current reply: utterance_id=%s reason=%s",
            utterance_id,
            reason,
        )
        self._active_robot_utterance_id = None
        self.barge_in_detector.reset()

        if self.llm_session:
            await self.llm_session.interrupt(utterance_id)
        if self.tts_session:
            await self.tts_session.interrupt(utterance_id)
        if self.audio_player:
            self.audio_player.interrupt(utterance_id=utterance_id, reason=reason)

        self._drain_async_queue(self.llm_queue)
        self._drain_async_queue(self.tts_queue)

    def _init_components(self) -> None:
        loop = asyncio.get_running_loop()

        self.dialog_logger = DialogFileLogger(file_path="dialog.txt")

        if (
            self.config.audio.output_mode.lower() == "ros1"
            and is_ros_dialog_available()
        ):
            try:
                self.dialog_text_pub = Ros1DialogTextPublisher(
                    topic=self.config.audio.ros1_llm_text_topic,
                    node_name=self.config.audio.ros1_node_name,
                    queue_size=self.config.audio.ros1_llm_text_queue_size,
                )
            except Exception as e:
                self.dialog_text_pub = None
                logger.warning("Failed to init ROS text publisher: %s", e)

        self.audio_capture = AudioCapture(
            output_queue=self.audio_queue,
            config=self.config.audio,
            loop=loop,
        )

        self.audio_player = RealtimeAudioPlaySession(
            input_queue=self.tts_queue,
            config=self.config.audio,
            aec_callback=(
                self.audio_capture.update_playback_reference
                if self.audio_capture
                else None
            ),
            on_playback_end=self._on_playback_end,
            is_utterance_active=self._is_utterance_active,
        )

        self.llm_session = RealtimeLLMSession(
            input_queue=self.asr_queue,
            output_queue=self.llm_queue,
            config=self.config.llm,
            rag=self.rag,
            on_stream_text=self._on_llm_stream_text,
            is_utterance_active=self._is_utterance_active,
        )

        self.tts_session = RealtimeTTSSession(
            input_queue=self.llm_queue,
            output_queue=self.tts_queue,
            config=self.config.tts,
            on_tts_start=self._on_tts_start,
            is_utterance_active=self._is_utterance_active,
        )

        self.dialog_manager.register_callback(DialogEvent.BARGE_IN, self._on_barge_in)

    async def _on_barge_in(self, event: DialogEvent, data) -> None:
        logger.info("Detected user barge-in")
        await self.interrupt_current_utterance(reason="barge_in")

    async def _on_tts_start(self, sentence: LLMSentence) -> None:
        if sentence.text:
            await self.dialog_manager.handle_llm_sentence(sentence.text)

    async def _on_playback_end(self, utterance_id: int) -> None:
        if not self._is_utterance_active(utterance_id):
            return

        self._active_robot_utterance_id = None
        if self.dialog_manager.current_state == DialogState.SPEAKING:
            await self.dialog_manager.handle_tts_end()
        elif self.dialog_manager.current_state == DialogState.THINKING:
            await self.dialog_manager.reset()

    async def _on_llm_stream_text(
        self,
        utterance_id: int,
        text: str,
        is_final: bool,
    ) -> None:
        if self.dialog_text_pub:
            self.dialog_text_pub.publish_text(text, is_final, utterance_id)

        if is_final and text and self.dialog_logger:
            await self.dialog_logger.log_line("BOT", text)

    async def start(self) -> None:
        if self._running:
            return

        logger.info("=" * 50)
        logger.info("Starting realtime voice dialog system...")
        logger.info("=" * 50)

        self._running = True
        self._init_components()

        if self.dialog_logger:
            await self.dialog_logger.start()

        self.audio_capture.start()
        await self.llm_session.start()
        await self.tts_session.start()
        await self.audio_player.start()

        self._tasks = [
            asyncio.create_task(self._vad_loop()),
            asyncio.create_task(self._heartbeat()),
        ]

        logger.info(
            "Model=%s | TTS=%s | duplex=%s | input=%s | output=%s",
            self.config.llm.model,
            self.config.tts.vcn,
            self.config.duplex_mode,
            self.config.audio.input_mode,
            self.config.audio.output_mode,
        )

        if self.config.welcome_message:
            welcome_task = asyncio.create_task(self._play_welcome_message())
            self._track_background_task(welcome_task)
            await asyncio.sleep(0.05)

        logger.info("System ready, please start speaking.")

    async def _play_welcome_message(self) -> None:
        welcome_text = (self.config.welcome_message or "").strip()
        if not welcome_text or not self.tts_session:
            return

        logger.info("[Welcome] Synthesizing startup prompt")
        self._active_robot_utterance_id = WELCOME_UTTERANCE_ID

        try:
            chunk_seq = 0
            async for audio_chunk in self.tts_session.client.synthesize(welcome_text):
                if not self._is_utterance_active(WELCOME_UTTERANCE_ID):
                    break
                await self.tts_queue.put(
                    TTSChunk(
                        utterance_id=WELCOME_UTTERANCE_ID,
                        chunk_seq=chunk_seq,
                        audio_bytes=audio_chunk,
                        is_last=False,
                    )
                )
                chunk_seq += 1

            if self._is_utterance_active(WELCOME_UTTERANCE_ID):
                await self.tts_queue.put(
                    TTSChunk(
                        utterance_id=WELCOME_UTTERANCE_ID,
                        chunk_seq=chunk_seq,
                        audio_bytes=b"",
                        is_last=True,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[Welcome] Playback failed: %s", e)
            if self._is_utterance_active(WELCOME_UTTERANCE_ID):
                self._active_robot_utterance_id = None

    async def _heartbeat(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(5)
                logger.info("System heartbeat")
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        if not self._running:
            return

        logger.info("Stopping system...")
        self._running = False

        await self.interrupt_current_utterance(reason="shutdown")

        for task in self._tasks:
            task.cancel()
        for task in list(self._background_tasks):
            task.cancel()

        for task in self._tasks + list(self._background_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("Background task stopped with error: %s", e)

        if self.audio_capture:
            self.audio_capture.stop()
        if self.audio_player:
            await self.audio_player.stop()
        if self.llm_session:
            await self.llm_session.stop()
        if self.tts_session:
            await self.tts_session.stop()
        if self.dialog_logger:
            await self.dialog_logger.stop()
        if self.dialog_text_pub:
            self.dialog_text_pub.close()

        logger.info("System stopped")

    async def _vad_loop(self) -> None:
        max_silence_frames = max(
            1,
            int(self.config.asr.max_silence_ms / self.config.audio.input_chunk_ms),
        )
        full_duplex_enabled = (
            (self.config.duplex_mode or "half").lower() == "full"
            and self.config.enable_barge_in
        )

        while self._running:
            try:
                audio_frame = await self.audio_queue.get()
                if not audio_frame:
                    continue

                self._preroll_frames.append(audio_frame)
                is_speech = self.vad.is_speech(audio_frame)
                state = self.dialog_manager.current_state

                if full_duplex_enabled and state in (
                    DialogState.THINKING,
                    DialogState.SPEAKING,
                ):
                    playback_floor = (
                        self.audio_capture.get_playback_reference_level()
                        if self.audio_capture
                        else 0.0
                    )
                    if is_speech and self.barge_in_detector.detect(
                        audio_frame,
                        playback_leak_floor=playback_floor,
                        echo_ratio=self.config.barge_in_echo_ratio,
                    ):
                        await self.dialog_manager.handle_barge_in()
                        self._is_listening = True
                        self._silence_frames = 0
                        self._current_audio_buffer = list(self._preroll_frames)
                        self.barge_in_detector.reset()
                        continue

                    if not is_speech:
                        self.barge_in_detector.reset()
                    continue

                self.barge_in_detector.reset()

                if state == DialogState.IDLE:
                    if is_speech:
                        await self.dialog_manager.handle_voice_start()
                        self._is_listening = True
                        self._silence_frames = 0
                        self._current_audio_buffer = list(self._preroll_frames)
                        continue

                if self.dialog_manager.current_state != DialogState.LISTENING:
                    continue

                self._current_audio_buffer.append(audio_frame)
                if is_speech:
                    self._silence_frames = 0
                else:
                    self._silence_frames += 1

                if self._silence_frames < max_silence_frames:
                    continue

                await self.dialog_manager.handle_voice_end()
                self._is_listening = False
                self._silence_frames = 0

                audio_data = b"".join(self._current_audio_buffer)
                self._current_audio_buffer = []

                process_task = asyncio.create_task(self._process_audio(audio_data))
                self._track_background_task(process_task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("VAD loop error: %s", e)

    async def _process_audio(self, audio_data: bytes) -> None:
        if not audio_data:
            await self.dialog_manager.reset()
            return

        if self.config.audio.sample_rate != self.config.asr.sample_rate:
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                resample_ratio = (
                    self.config.asr.sample_rate / self.config.audio.sample_rate
                )
                new_length = max(1, int(len(audio_array) * resample_ratio))
                audio_data = scipy_signal.resample(audio_array, new_length).astype(
                    np.int16
                ).tobytes()
            except Exception as e:
                logger.error("Failed to resample audio for ASR: %s", e)
                await self.dialog_manager.reset()
                return

        try:
            async with ASRClient(self.config.asr) as client:
                await client.initialize()

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
                    await client.send_audio(segment, i == len(segments) - 1)

                final_text = ""
                async for resp in client.receive_responses():
                    if resp.code != 0:
                        detail = None
                        if resp.payload_msg:
                            detail = (
                                resp.payload_msg.get("message")
                                or resp.payload_msg.get("msg")
                                or resp.payload_msg
                            )
                        logger.error("ASR error %s: %s", resp.code, detail or "unknown")
                        break

                    if resp.is_last_package:
                        final_text = resp.get_text()
                        break

                if not final_text:
                    logger.debug("ASR produced no final text")
                    await self.dialog_manager.reset()
                    return

                logger.info("[ASR] User: %s", final_text)
                utterance_id = self._allocate_utterance_id()
                self._active_robot_utterance_id = utterance_id
                self.barge_in_detector.reset()

                await self.dialog_manager.handle_asr_result(final_text)

                if self.dialog_logger:
                    await self.dialog_logger.log_line("USER", final_text)

                self._asr_kw_detector.detect_and_emit(final_text)
                await self.asr_queue.put(UserTurn(utterance_id, final_text))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("ASR processing error: %s", e)
            await self.dialog_manager.reset()

    def list_audio_devices(self) -> None:
        import pyaudio

        pa = pyaudio.PyAudio()

        print("\n=== 输入设备 ===")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} (采样率: {int(info['defaultSampleRate'])})")

        print("\n=== 输出设备 ===")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                print(f"  [{i}] {info['name']} (采样率: {int(info['defaultSampleRate'])})")

        pa.terminate()

    def get_stats(self) -> dict:
        return self.dialog_manager.get_stats()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime cascaded speech dialog system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --model qwen-plus
  python main.py --duplex-mode full
  python main.py --list-devices
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen-flash",
        choices=["qwen-flash", "qwen-turbo", "qwen-plus", "qwen-max"],
        help="LLM model",
    )
    parser.add_argument(
        "--list-devices",
        "-l",
        action="store_true",
        help="List available audio devices",
    )
    parser.add_argument(
        "--input-device",
        "-i",
        type=int,
        default=None,
        help="Input device index",
    )
    parser.add_argument(
        "--output-device",
        "-o",
        type=int,
        default=None,
        help="Output device index",
    )
    parser.add_argument(
        "--vad-threshold",
        type=int,
        default=500,
        help="VAD energy threshold",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=500,
        help="Silence duration that ends a user turn",
    )
    parser.add_argument(
        "--duplex-mode",
        choices=["half", "full"],
        default="half",
        help="Dialog duplex mode",
    )
    parser.add_argument("--no-aec", action="store_true", help="Disable AEC")
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Compatibility alias: force half-duplex mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_devices:
        system = VoiceDialogSystem()
        system.list_audio_devices()
        return

    config = get_config()
    config.llm.model = args.model
    config.audio.input_device_index = args.input_device
    config.audio.output_device_index = args.output_device
    config.asr.vad_silence_threshold = args.vad_threshold
    config.asr.max_silence_ms = args.silence_ms
    config.audio.enable_aec = not args.no_aec

    if args.no_barge_in:
        config.duplex_mode = "half"
        config.enable_barge_in = False
    else:
        config.duplex_mode = args.duplex_mode
        config.enable_barge_in = config.duplex_mode == "full"

    system = VoiceDialogSystem(config)
    loop = asyncio.get_running_loop()

    async def shutdown():
        logger.info("Shutdown signal received")
        await system.stop()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    except NotImplementedError:
        pass

    try:
        await system.start()
        while system._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
