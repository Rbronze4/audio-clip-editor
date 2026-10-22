import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playback import PCMFeeder, fatal_audio_error


class FakeDocument:
    source = SimpleNamespace(channels=2)

    def chunks(self, start, end, chunk_frames):
        samples = np.arange(400, dtype='<i2').reshape(-1, 2)[start:end]
        for i in range(0, len(samples), 7):
            yield samples[i:i + 7]


class Output:
    def __init__(self):
        self.data = bytearray()
        self.limit = 9

    def write(self, data):
        count = min(len(data), self.limit)
        self.data.extend(data[:count])
        return count


class PlaybackTests(unittest.TestCase):
    def test_partial_writes_and_backpressure_preserve_exact_selection(self):
        feeder = PCMFeeder(FakeDocument(), 13, 123)
        output = Output()
        feeder.feed(output, 0)
        self.assertEqual(output.data, b'')
        output.limit = 0
        feeder.feed(output, 200)
        self.assertFalse(feeder.finished)
        output.limit = 9
        for _ in range(100):
            feeder.feed(output, 17)
            if feeder.finished:
                break
        expected = np.arange(400, dtype='<i2').reshape(-1, 2)[13:123].tobytes()
        self.assertEqual(output.data, expected)
        self.assertTrue(feeder.finished)
        feeder.feed(output, 200)
        self.assertEqual(output.data, expected)

    def test_underrun_is_recoverable_but_real_errors_are_fatal(self):
        for name in ('NoError', 'UnderrunError'):
            self.assertFalse(fatal_audio_error(SimpleNamespace(name=name)))
        for name in ('OpenError', 'IOError', 'FatalError'):
            self.assertTrue(fatal_audio_error(SimpleNamespace(name=name)))

    def test_write_failure_surfaces(self):
        output = SimpleNamespace(write=lambda data: -1)
        with self.assertRaises(OSError):
            PCMFeeder(FakeDocument(), 0, 10).feed(output, 40)

    def test_native_qt_no_error_is_not_fatal(self):
        try:
            from PySide6 import QtMultimedia
        except ImportError:
            self.skipTest('PySide6 is not installed')
        # Qt 6.11 exposes distinct legacy QAudio and current QtAudio enums.
        current = getattr(QtMultimedia, 'QtAudio', QtMultimedia.QAudio)
        self.assertFalse(fatal_audio_error(current.Error.NoError))
        self.assertFalse(fatal_audio_error(QtMultimedia.QAudio.Error.NoError))
