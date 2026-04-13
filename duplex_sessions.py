import asyncio
import logging
from typing import Any, Callable, Optional, Set

from audio_constants import LLM_KWS_PATTERNS
from audio_manager import AudioPlayer, get_udp_controller
from config import LLMConfig, TTSConfig, get_config
from dialog_messages import LLMSentence, TTSChunk, UserTurn
from llm_client import LLMClient, RAGInterface, SentenceSplitter
from tts_client import TTSClient

logger = logging.getLogger(__name__)


def _drain_async_queue(
    queue: asyncio.Queue,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> int:
    removed = 0
    kept = []

    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if predicate is None or predicate(item):
            removed += 1
            continue
        kept.append(item)

    for item in kept:
        queue.put_nowait(item)

    return removed


class RealtimeLLMSession:
    """Interruptible utterance-scoped LLM streaming session."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        config: Optional[LLMConfig] = None,
        rag: Optional[RAGInterface] = None,
        on_stream_text: Optional[Callable[[int, str, bool], Any]] = None,
        is_utterance_active: Optional[Callable[[int], bool]] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client = LLMClient(config, rag)
        self.splitter = SentenceSplitter()
        self._on_stream_text = on_stream_text
        self._is_utterance_active = is_utterance_active or (lambda utterance_id: True)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_task: Optional[asyncio.Task] = None
        self._current_utterance_id: Optional[int] = None

    def _can_emit(self, utterance_id: int) -> bool:
        return self._is_utterance_active(int(utterance_id))

    async def _emit_stream_text(
        self,
        utterance_id: int,
        text: str,
        is_final: bool,
    ) -> None:
        if not self._on_stream_text:
            return

        try:
            res = self._on_stream_text(utterance_id, text, is_final)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.warning("LLM stream callback failed: %s", e)

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        await self.interrupt()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def switch_model(self, model_name: str) -> bool:
        return self.client.switch_model(model_name)

    def set_rag(self, rag: RAGInterface) -> None:
        self.client.set_rag(rag)

    def clear_history(self) -> None:
        self.client.clear_history()

    async def interrupt(self, utterance_id: Optional[int] = None) -> None:
        if (
            self._current_task
            and not self._current_task.done()
            and (utterance_id is None or utterance_id == self._current_utterance_id)
        ):
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        self._current_task = None
        self._current_utterance_id = None
        self.splitter.reset()
        _drain_async_queue(
            self.output_queue,
            predicate=lambda item: utterance_id is None
            or isinstance(item, LLMSentence)
            and item.utterance_id == utterance_id,
        )

    async def _run_response(self, user_turn: UserTurn) -> None:
        use_rag = get_config().rag.enabled

        async for chunk in self.client.chat_stream(user_turn.text, use_rag):
            if not self._can_emit(user_turn.utterance_id):
                raise asyncio.CancelledError()

            await self._emit_stream_text(user_turn.utterance_id, chunk, False)

            sentences = self.splitter.feed(chunk)
            for sentence in sentences:
                if not self._can_emit(user_turn.utterance_id):
                    raise asyncio.CancelledError()
                await self.output_queue.put(
                    LLMSentence(user_turn.utterance_id, sentence, False)
                )
                await self._emit_stream_text(user_turn.utterance_id, sentence, True)
                logger.info("[LLM] 机器人: %s", sentence)

        remaining = self.splitter.flush()
        if remaining and self._can_emit(user_turn.utterance_id):
            await self.output_queue.put(LLMSentence(user_turn.utterance_id, remaining, False))
            await self._emit_stream_text(user_turn.utterance_id, remaining, True)
            logger.info("[LLM] 机器人: %s", remaining)

        if self._can_emit(user_turn.utterance_id):
            await self.output_queue.put(LLMSentence(user_turn.utterance_id, "", True))

    async def _run(self) -> None:
        while self._running:
            try:
                user_turn = await self.input_queue.get()
                if user_turn is None:
                    break

                if not isinstance(user_turn, UserTurn):
                    user_turn = UserTurn(-1, str(user_turn))

                if not self._can_emit(user_turn.utterance_id):
                    continue

                self.splitter.reset()
                self._current_utterance_id = user_turn.utterance_id
                self._current_task = asyncio.create_task(self._run_response(user_turn))
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    logger.debug("LLM response interrupted")
                finally:
                    self._current_task = None
                    self._current_utterance_id = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("LLM session error: %s", e)


class RealtimeTTSSession:
    """Interruptible utterance-scoped TTS session."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        config: Optional[TTSConfig] = None,
        on_tts_start: Optional[Callable[[LLMSentence], Any]] = None,
        is_utterance_active: Optional[Callable[[int], bool]] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client = TTSClient(config)
        self._on_tts_start = on_tts_start
        self._is_utterance_active = is_utterance_active or (lambda utterance_id: True)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_task: Optional[asyncio.Task] = None
        self._current_utterance_id: Optional[int] = None
        self._chunk_seq = {}

        self._llm_keyword_buffer = ""
        self._llm_buffer_max_len = 50
        self._llm_kws_fired: Set[str] = set()
        self._udp_controller = get_udp_controller()

    def _can_emit(self, utterance_id: int) -> bool:
        return self._is_utterance_active(int(utterance_id))

    def _reset_keywords(self) -> None:
        self._llm_keyword_buffer = ""
        self._llm_kws_fired.clear()

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        await self.interrupt()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def interrupt(self, utterance_id: Optional[int] = None) -> None:
        if (
            self._current_task
            and not self._current_task.done()
            and (utterance_id is None or utterance_id == self._current_utterance_id)
        ):
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        self._current_task = None
        self._current_utterance_id = None
        _drain_async_queue(
            self.output_queue,
            predicate=lambda item: utterance_id is None
            or isinstance(item, TTSChunk)
            and item.utterance_id == utterance_id,
        )
        self._reset_keywords()

    def _detect_llm_keywords(self, text: str) -> None:
        if not get_config().enable_keyword_detection:
            return

        self._llm_keyword_buffer += text
        if len(self._llm_keyword_buffer) > self._llm_buffer_max_len:
            self._llm_keyword_buffer = self._llm_keyword_buffer[-self._llm_buffer_max_len :]

        for keyword, patterns in LLM_KWS_PATTERNS.items():
            if keyword in self._llm_kws_fired:
                continue
            matched_pattern = next((p for p in patterns if p in self._llm_keyword_buffer), None)
            if not matched_pattern:
                continue

            self._udp_controller.emit_voice_keyword(keyword)
            logger.info("[LLM-KWS] match '%s' -> %s", matched_pattern, keyword)
            self._llm_kws_fired.add(keyword)
            if keyword == "end":
                self._llm_keyword_buffer = ""

    async def _synthesize_sentence(self, sentence: LLMSentence) -> None:
        next_seq = int(self._chunk_seq.get(sentence.utterance_id, 0))

        async for audio_chunk in self.client.synthesize(sentence.text):
            if not self._can_emit(sentence.utterance_id):
                raise asyncio.CancelledError()

            await self.output_queue.put(
                TTSChunk(
                    utterance_id=sentence.utterance_id,
                    chunk_seq=next_seq,
                    audio_bytes=audio_chunk,
                    is_last=False,
                )
            )
            next_seq += 1

        self._chunk_seq[sentence.utterance_id] = next_seq

    async def _run(self) -> None:
        while self._running:
            try:
                sentence = await self.input_queue.get()
                if sentence is None:
                    break

                if not isinstance(sentence, LLMSentence):
                    sentence = LLMSentence(-1, str(sentence), False)

                if not self._can_emit(sentence.utterance_id):
                    continue

                if sentence.is_final and not sentence.text:
                    await self.output_queue.put(
                        TTSChunk(
                            utterance_id=sentence.utterance_id,
                            chunk_seq=int(self._chunk_seq.get(sentence.utterance_id, 0)),
                            audio_bytes=b"",
                            is_last=True,
                        )
                    )
                    self._reset_keywords()
                    continue

                if not sentence.text:
                    continue

                self._detect_llm_keywords(sentence.text)
                if self._on_tts_start:
                    res = self._on_tts_start(sentence)
                    if asyncio.iscoroutine(res):
                        await res

                self._current_utterance_id = sentence.utterance_id
                self._current_task = asyncio.create_task(self._synthesize_sentence(sentence))
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    logger.debug("TTS sentence interrupted")
                finally:
                    self._current_task = None
                    self._current_utterance_id = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("TTS session error: %s", e)


class RealtimeAudioPlaySession:
    """Interruptible audio playback session."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        config=None,
        aec_callback: Optional[Callable[[bytes], None]] = None,
        on_playback_end: Optional[Callable[[int], Any]] = None,
        is_utterance_active: Optional[Callable[[int], bool]] = None,
    ):
        self.input_queue = input_queue
        self.player = AudioPlayer(config, aec_callback)
        self._on_playback_end = on_playback_end
        self._is_utterance_active = is_utterance_active or (lambda utterance_id: True)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_utterance_id: Optional[int] = None

    def _can_play(self, utterance_id: int) -> bool:
        return self._is_utterance_active(int(utterance_id))

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        self.interrupt(reason="stop")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.player.close()

    def interrupt(self, utterance_id: Optional[int] = None, reason: str = "barge_in") -> None:
        self.player.interrupt(utterance_id=utterance_id, reason=reason)
        if utterance_id is None or utterance_id == self._current_utterance_id:
            self._current_utterance_id = None
        _drain_async_queue(
            self.input_queue,
            predicate=lambda item: utterance_id is None
            or isinstance(item, TTSChunk)
            and item.utterance_id == utterance_id,
        )

    async def _run(self) -> None:
        while self._running:
            try:
                chunk = await self.input_queue.get()
                if chunk is None:
                    continue

                if not isinstance(chunk, TTSChunk):
                    continue

                if chunk.is_last and not chunk.audio_bytes:
                    if self._can_play(chunk.utterance_id):
                        self.player.on_playback_complete()
                        self._current_utterance_id = None
                        if self._on_playback_end:
                            res = self._on_playback_end(chunk.utterance_id)
                            if asyncio.iscoroutine(res):
                                await res
                    continue

                if not chunk.audio_bytes or not self._can_play(chunk.utterance_id):
                    continue

                if chunk.utterance_id != self._current_utterance_id:
                    self.player.resume()
                    self._current_utterance_id = chunk.utterance_id

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.player.play_chunk, chunk)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Playback session error: %s", e)
