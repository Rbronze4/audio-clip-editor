import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from app import MainWindow
from core import Source, Document


class ViewHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        legacy = QSettings(str(self.root / 'legacy.ini'), QSettings.Format.IniFormat)
        with patch('app.QSettings', return_value=legacy):
            self.window = MainWindow(self.root / 'settings.json')
        self.window.document = Document(self.source('original'))
        self.window.fit()

    def source(self, name):
        path = self.root / (name + '.pcm')
        path.write_bytes(bytes(60000 * 2))
        return Source(name + '.wav', path, 1000, 1, 60000, 0, threading.Event())

    def tearDown(self):
        self.window.document.undo_stack.clear()
        self.window.close()
        self.temp.cleanup()

    def assert_view(self, start, end):
        left, right = self.window.view.viewRange()[0]
        self.assertAlmostEqual(left, start)
        self.assertAlmostEqual(right, end)

    def test_delete_example_and_history_restores_each_view(self):
        w = self.window
        w.view.show_range(18, 24)
        w.document.select((20500, 21200))
        w.edit(False)
        self.assert_view(18, 24)
        self.assertEqual(w.document.state.cursor, 20500)
        self.assertIsNone(w.document.state.selection)
        self.assertAlmostEqual(w.view.duration, 59.3)
        w.view.show_range(19, 22)
        w.undo()
        self.assert_view(18, 24)
        w.redo()
        self.assert_view(19, 22)
        self.assertEqual(w.document.state.cursor, 20500)

    def test_tail_clamp_and_keep_trim_coordinates(self):
        w = self.window
        w.view.show_range(54, 60)
        w.document.select((58000, 60000))
        w.edit(False)
        self.assert_view(52, 58)
        self.assertEqual(w.document.state.cursor, 58000)
        w.undo()
        self.assert_view(54, 60)
        w.view.show_range(18, 24)
        w.document.select((20500, 40000))
        w.edit(True)
        self.assert_view(0, 6)
        self.assertEqual(w.document.state.cursor, 0)
        w.undo()
        self.assert_view(18, 24)
        w.redo()
        self.assert_view(0, 6)

    def test_short_remaining_clip_and_empty_clip(self):
        w = self.window
        w.view.show_range(18, 24)
        w.document.select((20500, 21200))
        w.edit(True)
        self.assert_view(0, .7)
        w.document.select((0, 700))
        w.edit(False)
        self.assert_view(0, .01)
        self.assertEqual(w.document.state.cursor, 0)
        w.undo()
        self.assert_view(0, .7)

    def test_offscreen_edit_stays_visible_without_changing_width(self):
        w = self.window
        w.view.show_range(40, 46)
        w.document.select((20500, 21200))
        w.edit(False)
        self.assert_view(20.5, 26.5)

    def test_bgm_replacement_and_history_preserve_view(self):
        w = self.window
        w.view.show_range(18, 24)
        original, separated = self.source('a'), self.source('b')
        def job(label, operation, callback):
            callback((original, separated))
        with patch.object(w, 'run_job', side_effect=job):
            w.remove_bgm()
        self.assert_view(18, 24)
        w.undo()
        self.assert_view(18, 24)
        w.redo()
        self.assert_view(18, 24)


if __name__ == '__main__':
    unittest.main()
