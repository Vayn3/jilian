import json
import struct
import time
from dataclasses import dataclass
from typing import Optional, Union


AUDIO_FRAME_MAGIC = b"JLAU"
AUDIO_FRAME_VERSION = 1
AUDIO_FRAME_FLAG_LAST = 0x01
_AUDIO_FRAME_HEADER = struct.Struct(">4sBBIII")


@dataclass(frozen=True)
class DecodedAudioFrame:
    utterance_id: int
    chunk_seq: int
    is_last: bool
    payload: bytes


@dataclass(frozen=True)
class AudioControlMessage:
    cmd: str
    utterance_id: int
    reason: str
    timestamp: float


def pack_audio_frame(
    payload: bytes,
    utterance_id: int,
    chunk_seq: int,
    is_last: bool = False,
) -> bytes:
    flags = AUDIO_FRAME_FLAG_LAST if is_last else 0
    header = _AUDIO_FRAME_HEADER.pack(
        AUDIO_FRAME_MAGIC,
        AUDIO_FRAME_VERSION,
        flags,
        int(utterance_id),
        int(chunk_seq),
        len(payload),
    )
    return header + payload


def try_unpack_audio_frame(data: bytes) -> Optional[DecodedAudioFrame]:
    if len(data) < _AUDIO_FRAME_HEADER.size:
        return None

    magic, version, flags, utterance_id, chunk_seq, payload_len = (
        _AUDIO_FRAME_HEADER.unpack_from(data)
    )
    if magic != AUDIO_FRAME_MAGIC or version != AUDIO_FRAME_VERSION:
        return None

    payload = data[_AUDIO_FRAME_HEADER.size :]
    if payload_len != len(payload):
        return None

    return DecodedAudioFrame(
        utterance_id=utterance_id,
        chunk_seq=chunk_seq,
        is_last=bool(flags & AUDIO_FRAME_FLAG_LAST),
        payload=payload,
    )


def build_stop_message(utterance_id: int, reason: str) -> str:
    payload = {
        "cmd": "stop",
        "utterance_id": int(utterance_id),
        "reason": reason,
        "timestamp": time.time(),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_control_message(
    payload: Union[str, bytes, bytearray],
) -> Optional[AudioControlMessage]:
    try:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    cmd = str(data.get("cmd") or "").strip()
    if not cmd:
        return None

    try:
        utterance_id = int(data.get("utterance_id", -1))
    except Exception:
        return None

    return AudioControlMessage(
        cmd=cmd,
        utterance_id=utterance_id,
        reason=str(data.get("reason") or ""),
        timestamp=float(data.get("timestamp") or 0.0),
    )
