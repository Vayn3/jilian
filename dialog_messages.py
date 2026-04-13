from dataclasses import dataclass


WELCOME_UTTERANCE_ID = 0


@dataclass(frozen=True)
class UserTurn:
    utterance_id: int
    text: str


@dataclass(frozen=True)
class LLMSentence:
    utterance_id: int
    text: str
    is_final: bool = False


@dataclass(frozen=True)
class TTSChunk:
    utterance_id: int
    chunk_seq: int
    audio_bytes: bytes
    is_last: bool = False
