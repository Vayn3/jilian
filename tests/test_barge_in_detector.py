import struct
import unittest

from conversation import BargeInDetector


def pcm_frame(level: int, samples: int = 320) -> bytes:
    level = max(-32768, min(32767, int(level)))
    return struct.pack("<" + "h" * samples, *([level] * samples))


class BargeInDetectorTests(unittest.TestCase):
    def test_static_threshold_with_min_duration(self):
        detector = BargeInDetector(
            threshold=500,
            min_duration_ms=120,
            frame_duration_ms=20,
        )

        for _ in range(5):
            self.assertFalse(detector.detect(pcm_frame(700)))
        self.assertTrue(detector.detect(pcm_frame(700)))

    def test_dynamic_echo_threshold_blocks_false_trigger(self):
        detector = BargeInDetector(
            threshold=300,
            min_duration_ms=100,
            frame_duration_ms=20,
        )

        for _ in range(8):
            self.assertFalse(
                detector.detect(
                    pcm_frame(650),
                    playback_leak_floor=400,
                    echo_ratio=1.8,
                )
            )

        for _ in range(4):
            self.assertFalse(
                detector.detect(
                    pcm_frame(900),
                    playback_leak_floor=400,
                    echo_ratio=1.8,
                )
            )
        self.assertTrue(
            detector.detect(
                pcm_frame(900),
                playback_leak_floor=400,
                echo_ratio=1.8,
            )
        )

    def test_reset_after_silence(self):
        detector = BargeInDetector(
            threshold=500,
            min_duration_ms=60,
            frame_duration_ms=20,
        )

        self.assertFalse(detector.detect(pcm_frame(800)))
        self.assertFalse(detector.detect(pcm_frame(0)))
        self.assertFalse(detector.detect(pcm_frame(800)))
        self.assertFalse(detector.detect(pcm_frame(800)))
        self.assertTrue(detector.detect(pcm_frame(800)))


if __name__ == "__main__":
    unittest.main()
