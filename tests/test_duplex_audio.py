import unittest

from duplex_audio import (
    AUDIO_FRAME_FLAG_LAST,
    build_stop_message,
    pack_audio_frame,
    parse_control_message,
    try_unpack_audio_frame,
)


class DuplexAudioTests(unittest.TestCase):
    def test_pack_and_unpack_audio_frame(self):
        payload = b"\x01\x02\x03\x04"
        packed = pack_audio_frame(
            payload,
            utterance_id=7,
            chunk_seq=3,
            is_last=True,
        )

        decoded = try_unpack_audio_frame(packed)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.utterance_id, 7)
        self.assertEqual(decoded.chunk_seq, 3)
        self.assertTrue(decoded.is_last)
        self.assertEqual(decoded.payload, payload)

    def test_legacy_payload_returns_none(self):
        self.assertIsNone(try_unpack_audio_frame(b"plain-pcm-bytes"))

    def test_stop_message_round_trip(self):
        raw = build_stop_message(utterance_id=11, reason="barge_in")
        parsed = parse_control_message(raw)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.cmd, "stop")
        self.assertEqual(parsed.utterance_id, 11)
        self.assertEqual(parsed.reason, "barge_in")


if __name__ == "__main__":
    unittest.main()
