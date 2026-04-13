# -*- coding: utf-8 -*-
"""Audio capture, playback, and duplex-related helpers."""

import asyncio
import audioop
import json
import logging
import socket
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import pyaudio
from scipy import signal as scipy_signal

from config import AudioConfig, get_config
from dialog_messages import TTSChunk
from ros_audio_stream import Ros1SpeakerStream, is_ros_available

try:
    import rospy
    from std_msgs.msg import ByteMultiArray

    try:
        from audio_common_msgs.msg import AudioData as RosAudioData

        _HAS_ROS_AUDIO_DATA = True
    except Exception:
        RosAudioData = None
        _HAS_ROS_AUDIO_DATA = False
    _HAS_ROS1 = True
except Exception:
    rospy = None
    ByteMultiArray = None
    RosAudioData = None
    _HAS_ROS1 = False
    _HAS_ROS_AUDIO_DATA = False

logger = logging.getLogger(__name__)


class AudioFormatConverter:
    @staticmethod
    def resample(
        audio_data: bytes,
        from_rate: int,
        to_rate: int,
        sample_width: int = 2,
    ) -> bytes:
        if from_rate == to_rate or not audio_data:
            return audio_data

        if sample_width != 2:
            raise ValueError("Only int16 resampling is supported")

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return audio_data

        resample_ratio = to_rate / from_rate
        new_length = max(1, int(len(audio_array) * resample_ratio))
        resampled_array = scipy_signal.resample(audio_array, new_length)
        return resampled_array.astype(np.int16).tobytes()

    @staticmethod
    def int16_to_float32(audio_data: bytes) -> bytes:
        int16_array = np.frombuffer(audio_data, dtype=np.int16)
        float32_array = int16_array.astype(np.float32) / 32768.0
        return float32_array.tobytes()

    @staticmethod
    def float32_to_int16(audio_data: bytes) -> bytes:
        float32_array = np.frombuffer(audio_data, dtype=np.float32)
        int16_array = np.clip(float32_array * 32768.0, -32768, 32767).astype(np.int16)
        return int16_array.tobytes()


class SimpleAEC:
    """A lightweight NLMS-style echo canceller."""

    def __init__(
        self,
        filter_length: int = 2048,
        step_size: float = 0.1,
        sample_rate: int = 16000,
    ):
        self.filter_length = max(32, int(filter_length))
        self.step_size = float(step_size)
        self.sample_rate = int(sample_rate)
        self.weights = np.zeros(self.filter_length, dtype=np.float32)
        self.ref_buffer = deque([0.0] * self.filter_length, maxlen=self.filter_length)
        self.eps = 1e-6

    def update_reference(self, audio_data: bytes) -> None:
        if not audio_data:
            return
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        for sample in samples:
            self.ref_buffer.append(float(sample))

    def process(self, mic_data: bytes) -> bytes:
        if not mic_data:
            return mic_data

        mic_samples = (
            np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        if len(mic_samples) == 0:
            return mic_data

        output_samples = np.zeros_like(mic_samples)
        ref_array = np.array(self.ref_buffer, dtype=np.float32)

        for i, mic_sample in enumerate(mic_samples):
            ref_vector = ref_array[-self.filter_length :]
            echo_estimate = float(np.dot(self.weights, ref_vector))
            error = float(mic_sample - echo_estimate)
            output_samples[i] = error

            norm = float(np.dot(ref_vector, ref_vector)) + self.eps
            self.weights += (self.step_size / norm) * error * ref_vector

            self.ref_buffer.append(0.0)
            ref_array = np.array(self.ref_buffer, dtype=np.float32)

        output_samples = np.clip(output_samples * 32768.0, -32768, 32767).astype(np.int16)
        return output_samples.tobytes()

    def reset(self) -> None:
        self.weights.fill(0.0)
        self.ref_buffer.clear()
        self.ref_buffer.extend([0.0] * self.filter_length)


class SimpleNoiseSupressor:
    """A small adaptive noise gate."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = int(sample_rate)
        self.noise_estimate = None
        self.noise_frames = 0
        self.noise_estimation_frames = 10

    def process(self, audio_data: bytes) -> bytes:
        if not audio_data:
            return audio_data

        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return audio_data

        rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-6)
        if self.noise_frames < self.noise_estimation_frames:
            if self.noise_estimate is None:
                self.noise_estimate = rms
            else:
                self.noise_estimate = (self.noise_estimate * self.noise_frames + rms) / (
                    self.noise_frames + 1
                )
            self.noise_frames += 1
            return audio_data

        noise_floor = float(self.noise_estimate or 0.0)
        if rms <= noise_floor * 1.2:
            samples *= 0.35
        elif rms <= noise_floor * 1.8:
            samples *= 0.7

        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    def reset(self) -> None:
        self.noise_estimate = None
        self.noise_frames = 0


class AudioCapture:
    """Microphone capture that supports PyAudio or ROS input."""

    def __init__(
        self,
        output_queue: asyncio.Queue,
        config: Optional[AudioConfig] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.output_queue = output_queue
        self.config = config or get_config().audio
        self.loop = loop or asyncio.get_event_loop()

        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self._running = False
        self._input_mode = (self.config.input_mode or "pyaudio").lower()

        self._ros_audio_sub = None
        self._ros_msg_type = ""
        self._ros_ratecv_state = None
        self._ros_input_remainder = b""
        self._playback_reference_rms = 0.0

        self.aec: Optional[SimpleAEC] = None
        self.ns: Optional[SimpleNoiseSupressor] = None

        if self.config.enable_aec:
            self.aec = SimpleAEC(
                filter_length=self.config.aec_filter_length,
                sample_rate=self.config.sample_rate,
            )
        if self.config.enable_ns:
            self.ns = SimpleNoiseSupressor(sample_rate=self.config.sample_rate)

    def _enqueue_audio_frame(self, audio_data: bytes) -> None:
        def _put():
            try:
                self.output_queue.put_nowait(audio_data)
            except Exception as e:
                logger.warning("Failed to enqueue audio frame: %s", e)

        try:
            self.loop.call_soon_threadsafe(_put)
        except Exception as e:
            logger.warning("Failed to schedule audio frame enqueue: %s", e)

    def _enqueue_fixed_frames(self, audio_data: bytes) -> None:
        frame_ms = max(1, int(self.config.ros1_audio_frame_ms))
        frame_bytes = int(
            self.config.sample_rate
            * self.config.sample_width
            * self.config.channels
            * frame_ms
            / 1000
        )
        if frame_bytes <= 0:
            self._enqueue_audio_frame(audio_data)
            return

        pending = self._ros_input_remainder + audio_data
        while len(pending) >= frame_bytes:
            self._enqueue_audio_frame(pending[:frame_bytes])
            pending = pending[frame_bytes:]
        self._ros_input_remainder = pending

    def _apply_input_processing(self, audio_data: bytes) -> bytes:
        processed_data = audio_data
        if self.aec and self.config.enable_aec:
            processed_data = self.aec.process(processed_data)
        if self.ns and self.config.enable_ns:
            processed_data = self.ns.process(processed_data)
        return processed_data

    def _normalize_ros_audio(self, audio_data: bytes) -> bytes:
        normalized = audio_data
        sample_width = self.config.sample_width
        in_channels = max(1, int(self.config.ros1_input_channels))
        out_channels = max(1, int(self.config.channels))
        in_rate = max(1, int(self.config.ros1_input_sample_rate))
        out_rate = max(1, int(self.config.sample_rate))

        if in_channels != out_channels:
            if in_channels == 2 and out_channels == 1:
                normalized = audioop.tomono(normalized, sample_width, 0.5, 0.5)
            elif in_channels == 1 and out_channels == 2:
                normalized = audioop.tostereo(normalized, sample_width, 1.0, 1.0)
            else:
                raise ValueError(f"Unsupported channel conversion: {in_channels}->{out_channels}")

        if in_rate != out_rate:
            normalized, self._ros_ratecv_state = audioop.ratecv(
                normalized,
                sample_width,
                out_channels,
                in_rate,
                out_rate,
                self._ros_ratecv_state,
            )

        return normalized

    def _ros_audio_callback(self, msg) -> None:
        if not self._running:
            return

        try:
            audio_data = bytes(msg.data)
            normalized = self._normalize_ros_audio(audio_data)
            processed_data = self._apply_input_processing(normalized)
            self._enqueue_fixed_frames(processed_data)
        except Exception as e:
            logger.warning("Failed to process ROS input audio: %s", e)

    def _start_ros_capture(self) -> None:
        if not _HAS_ROS1:
            raise RuntimeError(
                "ROS1 (rospy) is unavailable, use pyaudio input or install ROS1"
            )

        if not rospy.core.is_initialized():
            rospy.init_node(
                self.config.ros1_input_node_name,
                anonymous=True,
                disable_signals=True,
            )

        topic = self.config.ros1_input_topic
        queue_size = self.config.ros1_input_queue_size
        if _HAS_ROS_AUDIO_DATA:
            self._ros_audio_sub = rospy.Subscriber(
                topic,
                RosAudioData,
                self._ros_audio_callback,
                queue_size=queue_size,
                tcp_nodelay=self.config.ros1_input_tcp_nodelay,
            )
            self._ros_msg_type = "audio_common_msgs/AudioData"
        else:
            self._ros_audio_sub = rospy.Subscriber(
                topic,
                ByteMultiArray,
                self._ros_audio_callback,
                queue_size=queue_size,
                tcp_nodelay=self.config.ros1_input_tcp_nodelay,
            )
            self._ros_msg_type = "std_msgs/ByteMultiArray"

        self._ros_ratecv_state = None
        self._ros_input_remainder = b""
        logger.info("ROS input subscribed: %s (%s)", topic, self._ros_msg_type)

    def _stop_ros_capture(self) -> None:
        if self._ros_audio_sub:
            try:
                self._ros_audio_sub.unregister()
            except Exception:
                pass
            self._ros_audio_sub = None
        self._ros_ratecv_state = None
        self._ros_input_remainder = b""

    def _get_device_info(self) -> List[dict]:
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "channels": info["maxInputChannels"],
                    }
                )
        return devices

    def list_devices(self) -> List[dict]:
        return self._get_device_info()

    def _check_device_compatibility(self, device_index: Optional[int]) -> bool:
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        try:
            if device_index is not None:
                info = self.pa.get_device_info_by_index(device_index)
            else:
                info = self.pa.get_default_input_device_info()

            return bool(
                self.pa.is_format_supported(
                    self.config.sample_rate,
                    input_device=device_index or info["index"],
                    input_channels=self.config.channels,
                    input_format=pyaudio.paInt16,
                )
            )
        except Exception as e:
            logger.warning("Input device compatibility check failed: %s", e)
            return False

    def update_playback_reference(self, audio_data: bytes) -> None:
        if self.aec:
            self.aec.update_reference(audio_data)
        if audio_data:
            try:
                current_rms = float(audioop.rms(audio_data, self.config.sample_width))
                self._playback_reference_rms = self._playback_reference_rms * 0.8 + current_rms * 0.2
            except Exception:
                pass

    def get_playback_reference_level(self) -> float:
        return float(self._playback_reference_rms)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            logger.warning("PyAudio input status: %s", status)

        if not self._running:
            return (None, pyaudio.paComplete)

        processed_data = self._apply_input_processing(in_data)
        self._enqueue_audio_frame(processed_data)
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        if self._input_mode == "ros1":
            try:
                self._start_ros_capture()
                return
            except Exception:
                self._running = False
                raise

        if self._input_mode != "pyaudio":
            self._running = False
            raise ValueError(f"Unsupported input_mode: {self._input_mode}")

        if not self.pa:
            self.pa = pyaudio.PyAudio()

        if not self._check_device_compatibility(self.config.input_device_index):
            logger.warning("Input device does not support requested format, trying defaults")

        frames_per_buffer = int(self.config.sample_rate * self.config.input_chunk_ms / 1000)
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._audio_callback,
            )
            self.stream.start_stream()
            logger.debug("PyAudio input started at %s Hz", self.config.sample_rate)
        except Exception as e:
            self._running = False
            logger.error("Failed to start audio capture: %s", e)
            raise

    def stop(self) -> None:
        self._running = False

        if self._input_mode == "ros1":
            self._stop_ros_capture()

        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.pa:
            self.pa.terminate()
            self.pa = None

        self._playback_reference_rms = 0.0
        self._ros_ratecv_state = None
        self._ros_input_remainder = b""

    def reset_aec(self) -> None:
        if self.aec:
            self.aec.reset()

    def reset_ns(self) -> None:
        if self.ns:
            self.ns.reset()


class UDPActionController:
    def __init__(self):
        self._config = get_config()
        self._voice_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._mic_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit_voice_keyword(self, keyword: str) -> None:
        if not self._config.enable_keyword_detection:
            return

        try:
            msg = json.dumps(
                {"type": "voice_keyword", "keyword": keyword, "timestamp": time.time()}
            )
            self._voice_socket.sendto(
                msg.encode("utf-8"),
                (self._config.voice_udp_host, self._config.voice_udp_port),
            )
            logger.info("[UDP:%s] voice keyword: %s", self._config.voice_udp_port, keyword)
        except Exception as e:
            logger.error("Failed to emit voice keyword: %s", e)

    def send_mic_command(self, command: str) -> None:
        try:
            msg = json.dumps(
                {"type": "mic_command", "command": command, "timestamp": time.time()}
            )
            self._mic_socket.sendto(
                msg.encode("utf-8"),
                (self._config.mic_udp_host, self._config.mic_udp_port),
            )
            logger.info("[UDP:%s] mic command: %s", self._config.mic_udp_port, command)
        except Exception as e:
            logger.error("Failed to send mic command: %s", e)

    def close(self) -> None:
        try:
            self._voice_socket.close()
        except Exception:
            pass
        try:
            self._mic_socket.close()
        except Exception:
            pass


_udp_controller: Optional[UDPActionController] = None


def get_udp_controller() -> UDPActionController:
    global _udp_controller
    if _udp_controller is None:
        _udp_controller = UDPActionController()
    return _udp_controller


class AudioPlayer:
    """Realtime audio player for local output or ROS publishing."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        aec_callback: Optional[Callable[[bytes], None]] = None,
    ):
        self.config = config or get_config().audio
        self.aec_callback = aec_callback
        self._output_mode = (self.config.output_mode or "pyaudio").lower()
        self._duplex_mode = (get_config().duplex_mode or "half").lower()

        self.format_converter = AudioFormatConverter()
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream = None

        self._is_playing = False
        self._interrupted = threading.Event()
        self._lock = threading.Lock()
        self._current_utterance_id: Optional[int] = None
        self._ros_chunk_seq = 0
        self._udp_controller = get_udp_controller()

    def _check_output_device(self, device_index: Optional[int]) -> bool:
        if self._output_mode != "pyaudio":
            return True

        if not self.pa:
            self.pa = pyaudio.PyAudio()

        try:
            if device_index is not None:
                info = self.pa.get_device_info_by_index(device_index)
            else:
                info = self.pa.get_default_output_device_info()
            return info["maxOutputChannels"] > 0
        except Exception:
            return False

    def list_devices(self) -> List[dict]:
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "channels": info["maxOutputChannels"],
                    }
                )
        return devices

    def _ensure_stream(self) -> None:
        if self.stream is not None:
            return

        if self._output_mode == "ros1":
            if not is_ros_available():
                logger.warning("ROS1 output unavailable, falling back to PyAudio")
                self._output_mode = "pyaudio"
            else:
                self.stream = Ros1SpeakerStream(
                    topic=self.config.ros1_topic,
                    control_topic=self.config.ros1_control_topic,
                    node_name=self.config.ros1_node_name,
                    queue_size=self.config.ros1_queue_size,
                    latched=self.config.ros1_latch,
                    duplex_mode=self._duplex_mode,
                )
                logger.info("Using ROS1 output on %s", self.config.ros1_topic)
                return

        if not self.pa:
            self.pa = pyaudio.PyAudio()

        if not self._check_output_device(self.config.output_device_index):
            logger.warning("Selected output device is unavailable, using default output")
            self.config.output_device_index = None

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            output_device_index=self.config.output_device_index,
            frames_per_buffer=self.config.output_buffer_size,
        )
        logger.info("Using local PyAudio output")

    def _ros_sample_width(self) -> int:
        return 4 if self.config.ros1_output_format == "f32le" else self.config.sample_width

    def _convert_output_audio(self, audio_data: bytes) -> bytes:
        output_data = audio_data
        if self._output_mode != "ros1":
            return output_data

        tts_rate = get_config().tts.sample_rate
        target_rate = self.config.ros1_output_sample_rate
        if tts_rate != target_rate:
            output_data = self.format_converter.resample(
                output_data,
                from_rate=tts_rate,
                to_rate=target_rate,
                sample_width=2,
            )

        if self.config.ros1_output_format == "f32le":
            output_data = self.format_converter.int16_to_float32(output_data)

        return output_data

    def _split_for_ros(self, output_data: bytes) -> List[bytes]:
        if self._duplex_mode != "full":
            return [output_data]

        frame_bytes = int(
            self.config.ros1_output_sample_rate
            * self._ros_sample_width()
            * self.config.channels
            * max(1, int(self.config.ros1_audio_frame_ms))
            / 1000
        )
        if frame_bytes <= 0 or len(output_data) <= frame_bytes:
            return [output_data]

        return [output_data[i : i + frame_bytes] for i in range(0, len(output_data), frame_bytes)]

    def _hard_flush_locked(self) -> None:
        if self.stream is None:
            return

        try:
            if hasattr(self.stream, "stop_stream"):
                self.stream.stop_stream()
        except Exception:
            pass
        try:
            if hasattr(self.stream, "close"):
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None

    def play_chunk(self, chunk: TTSChunk) -> bool:
        if self._interrupted.is_set():
            return False

        with self._lock:
            if self._interrupted.is_set():
                return False

            self._ensure_stream()
            self._is_playing = True
            if self._current_utterance_id != chunk.utterance_id:
                self._ros_chunk_seq = 0
            self._current_utterance_id = chunk.utterance_id

            try:
                if self.aec_callback and chunk.audio_bytes:
                    self.aec_callback(chunk.audio_bytes)

                output_data = self._convert_output_audio(chunk.audio_bytes)
                if self._output_mode == "ros1":
                    frames = self._split_for_ros(output_data)
                    for idx, frame in enumerate(frames):
                        if self._interrupted.is_set():
                            return False
                        self.stream.write(
                            frame,
                            utterance_id=chunk.utterance_id,
                            chunk_seq=self._ros_chunk_seq,
                            is_last=chunk.is_last and idx == len(frames) - 1,
                        )
                        self._ros_chunk_seq += 1
                else:
                    self.stream.write(output_data)
                return True
            except Exception as e:
                logger.error("Playback failed: %s", e)
                return False
            finally:
                self._is_playing = False

    def play(self, audio_data: bytes) -> bool:
        if self._interrupted.is_set():
            return False

        with self._lock:
            if self._interrupted.is_set():
                return False

            self._ensure_stream()
            self._is_playing = True
            try:
                if self.aec_callback and audio_data:
                    self.aec_callback(audio_data)
                output_data = self._convert_output_audio(audio_data)
                self.stream.write(output_data)
                return True
            except Exception as e:
                logger.error("Playback failed: %s", e)
                return False
            finally:
                self._is_playing = False

    def on_playback_complete(self) -> None:
        if self._duplex_mode == "half":
            self._udp_controller.send_mic_command("send_microphone")

    def interrupt(
        self,
        utterance_id: Optional[int] = None,
        reason: str = "barge_in",
    ) -> None:
        self._interrupted.set()
        with self._lock:
            if self._output_mode == "ros1" and self.stream and utterance_id is not None:
                try:
                    self.stream.send_stop(utterance_id, reason=reason)
                except Exception:
                    pass
            self._hard_flush_locked()
            self._is_playing = False
        logger.debug("Playback interrupted")

    def resume(self) -> None:
        self._interrupted.clear()
        self._current_utterance_id = None
        self._ros_chunk_seq = 0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def close(self) -> None:
        with self._lock:
            self._hard_flush_locked()
        if self.pa:
            self.pa.terminate()
            self.pa = None


class RealtimeAudioPlaySession:
    """Legacy playback session that consumes raw audio bytes."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        config: Optional[AudioConfig] = None,
        aec_callback: Optional[Callable[[bytes], None]] = None,
    ):
        self.input_queue = input_queue
        self.player = AudioPlayer(config, aec_callback)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        self.player.interrupt(reason="stop")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.player.close()

    def interrupt(self) -> None:
        self.player.interrupt()

    def resume(self) -> None:
        self.player.resume()

    async def _run(self) -> None:
        while self._running:
            try:
                audio_data = await self.input_queue.get()
                if not audio_data:
                    continue
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.player.play, audio_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Playback session error: %s", e)


class SimpleVAD:
    def __init__(self, threshold: int = 500):
        self.threshold = int(threshold)

    def is_speech(self, audio_data: bytes) -> bool:
        if not audio_data:
            return False
        try:
            energy = audioop.rms(audio_data, 2)
        except Exception:
            return False
        return energy >= self.threshold
