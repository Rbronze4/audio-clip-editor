import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Cancelled, Document, Source, export_wav
from import_audio import collect_audio
from separation import make_snapshot, run_process


class ImportTests(unittest.TestCase):
    def test_recursive_supported_files_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'nested').mkdir()
            for name in ['a.MP3', 'b.wav', 'readme.txt', 'nested/c.flac']:
                (root / name).touch()
            paths, errors = collect_audio([root, root / 'a.MP3'], [root / 'b.wav'], threading.Event())
            self.assertEqual([Path(p).name for p in paths], ['a.MP3', 'c.flac'])
            self.assertFalse(errors)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(Cancelled):
                collect_audio([root], [], cancel)


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.cancel = threading.Event()
        self.original = np.arange(4000, dtype='<i2').reshape(-1, 2)
        path = self.directory / 'source.pcm'
        path.write_bytes(self.original.tobytes())
        self.doc = Document(Source('source.mp3', path, 44100, 2, 2000, 0, self.cancel))

    def tearDown(self):
        self.doc.close()
        self.temp.cleanup()

    def audio(self, version):
        self.doc.switch_version(version)
        return np.concatenate(list(self.doc.chunks()))

    def test_separation_after_cut_ab_edits_undo_redo_and_export(self):
        self.doc.select((100, 200))
        self.doc.edit(False)
        baseline = np.concatenate([self.original[:100], self.original[200:]])
        self.doc.select((50, 250))
        self.doc.seek(90)
        before = self.doc.state
        replacement = self.directory / 'vocals.pcm'
        replacement.write_bytes(np.zeros((200, 2), dtype='<i2').tobytes())
        a = make_snapshot(self.doc.snapshot('original'), self.directory, self.cancel)
        b = make_snapshot(self.doc, self.directory, self.cancel, (50, 250, replacement))
        self.doc.apply_separation(a, b)
        expected = baseline.copy()
        expected[50:250] = 0
        np.testing.assert_array_equal(self.audio('separated'), expected)
        np.testing.assert_array_equal(self.audio('original'), baseline)
        self.assertEqual(self.doc.state.selection, (50, 250))
        self.assertEqual(self.doc.state.cursor, 90)
        self.doc.switch_version('separated')
        self.doc.undo()
        self.assertEqual(self.doc.state, before)
        self.doc.redo()
        np.testing.assert_array_equal(self.audio('separated'), expected)
        self.doc.select((0, 300))
        self.doc.edit(True)
        np.testing.assert_array_equal(self.audio('original'), baseline[:300])
        np.testing.assert_array_equal(self.audio('separated'), expected[:300])
        target = export_wav(self.doc, self.directory, 'test', 0, 300, self.cancel)
        import wave
        with wave.open(str(target)) as wav:
            self.assertEqual(wav.readframes(300), expected[:300].tobytes())

    def test_repeat_separation_keeps_previous_result_outside_selection(self):
        path = self.directory / 'replacement.pcm'
        path.write_bytes(bytes(100 * 2 * 2))
        a = make_snapshot(self.doc.snapshot('original'), self.directory, self.cancel)
        b = make_snapshot(self.doc, self.directory, self.cancel, (100, 200, path))
        self.doc.apply_separation(a, b)
        a2 = make_snapshot(self.doc.snapshot('original'), self.directory, self.cancel)
        b2 = make_snapshot(self.doc.snapshot('separated'), self.directory, self.cancel, (300, 400, path))
        self.doc.apply_separation(a2, b2)
        expected = self.original.copy()
        expected[100:200] = expected[300:400] = 0
        np.testing.assert_array_equal(self.audio('separated'), expected)
        np.testing.assert_array_equal(self.audio('original'), self.original)

    def test_cancellation_does_not_commit_or_leak_snapshots(self):
        before = self.doc.state
        existing = set(self.directory.iterdir())
        self.cancel.set()
        with self.assertRaises(Cancelled):
            make_snapshot(self.doc, self.directory, self.cancel)
        self.assertEqual(self.doc.state, before)
        self.assertEqual(set(self.directory.iterdir()), existing)

    def test_subprocess_cancel_stops_running_process(self):
        marker = self.directory / 'should_not_exist'
        timer = threading.Timer(.25, self.cancel.set)
        timer.start()
        try:
            with self.assertRaises(Cancelled):
                run_process([sys.executable, '-c',
                             'import time,pathlib,sys;time.sleep(3);pathlib.Path(sys.argv[1]).touch()',
                             str(marker)], self.cancel)
            self.assertFalse(marker.exists())
        finally:
            timer.cancel()
