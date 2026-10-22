import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import find_ffmpeg
from separation import convert_for_model, restore_model_output


class ConversionTests(unittest.TestCase):
    def test_mono_stereo_roundtrip_preserves_gain_and_length(self):
        try:
            find_ffmpeg()
        except RuntimeError:
            self.skipTest('FFmpeg is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original = (np.sin(np.arange(44100) * 2 * np.pi * 440 / 44100) * 12000).astype('<i2')
            with wave.open(str(folder / 'input.wav'), 'wb') as wav:
                wav.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                wav.writeframes(original.tobytes())
            cancel = threading.Event()
            convert_for_model(folder / 'input.wav', folder / 'stereo.wav', 1, cancel)
            restore_model_output(folder / 'stereo.wav', folder / 'roundtrip.pcm',
                                 44100, 1, len(original), cancel)
            result = np.fromfile(folder / 'roundtrip.pcm', dtype='<i2')
            np.testing.assert_array_equal(result, original)
