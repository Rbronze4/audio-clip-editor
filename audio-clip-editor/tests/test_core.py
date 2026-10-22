import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Cancelled, Document, Source, export_wav


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.samples = np.arange(40000, dtype=np.int16).reshape(-1, 2)
        self.pcm = self.folder / 'source.pcm'
        self.pcm.write_bytes(self.samples.tobytes())
        self.source = Source('original.mp3', self.pcm, 44100, 2, len(self.samples), 0,
                             threading.Event())
        self.doc = Document(self.source)

    def tearDown(self):
        self.source.close()
        self.temp.cleanup()

    def materialize(self):
        chunks = list(self.doc.chunks())
        return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int16)

    def test_delete_join_and_undo_redo_preserve_source(self):
        original = self.pcm.read_bytes()
        self.doc.select((400, 1600))
        self.doc.seek(500)
        before = self.doc.state
        self.doc.edit(False)
        np.testing.assert_array_equal(self.materialize(),
                                      np.concatenate([self.samples[:400], self.samples[1600:]]))
        self.doc.undo()
        self.assertEqual(self.doc.state, before)
        self.doc.redo()
        self.assertEqual(self.doc.frames, len(self.samples) - 1200)
        self.assertEqual(self.pcm.read_bytes(), original)

    def test_keep_across_join_and_redo_branch(self):
        self.doc.select((10, 20))
        self.doc.edit(False)
        self.doc.select((5, 15))
        self.doc.edit(True)
        np.testing.assert_array_equal(self.materialize(),
                                      np.concatenate([self.samples[5:10], self.samples[20:25]]))
        self.doc.undo()
        self.doc.select((0, 3))
        self.doc.edit(True)
        self.assertFalse(self.doc.redo_stack)

    def test_delete_everything_then_undo(self):
        self.doc.select((0, self.doc.frames))
        self.doc.edit(False)
        self.assertEqual(self.doc.frames, 0)
        self.assertEqual(self.doc.envelope(0, 1), ([], [], []))
        self.doc.undo()
        self.assertEqual(self.doc.frames, len(self.samples))

    def test_export_selection_sequential_and_no_overwrite(self):
        folder = self.folder / 'exports'
        cancel = threading.Event()
        first = export_wav(self.doc, folder, 'clean_normal', 20, 80, cancel)
        first_bytes = first.read_bytes()
        second = export_wav(self.doc, folder, 'clean_normal', 0, 5, cancel)
        self.assertEqual(first.name, 'clean_normal_0001.wav')
        self.assertEqual(second.name, 'clean_normal_0002.wav')
        self.assertEqual(first.read_bytes(), first_bytes)
        with wave.open(str(first), 'rb') as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                             (44100, 2, 2))
            self.assertEqual(wav.readframes(100), self.samples[20:80].tobytes())

    def test_cancel_export_removes_incomplete_file(self):
        cancel = threading.Event()
        cancel.set()
        folder = self.folder / 'exports'
        with self.assertRaises(Cancelled):
            export_wav(self.doc, folder, 'clean_normal', 0, 100, cancel)
        self.assertEqual(list(folder.iterdir()), [])

    def test_envelope_contains_stereo_transients_and_clips_deleted_boundaries(self):
        self.doc.select((555, 15000))
        self.doc.edit(True)
        for pixels in (8, 100, 1000):
            x, low, high = self.doc.envelope(0, self.doc.frames, pixels)
            self.assertEqual(len(x), len(low))
            self.assertEqual(len(x), len(high))
            self.assertAlmostEqual(min(low), self.samples[555:15000].min() / 32768)
            self.assertAlmostEqual(max(high), self.samples[555:15000].max() / 32768)
            self.assertTrue(all(0 <= p < self.doc.frames / 44100 for p in x))


if __name__ == '__main__':
    unittest.main()
