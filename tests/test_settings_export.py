import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Cancelled, Document, Source, decode
from preferences import Preferences, load_preferences, save_preferences
from export_audio import export_audio
from settings_dialog import SettingsDialog
from PySide6.QtWidgets import QApplication, QInputDialog


class SettingsExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cancel = threading.Event()
        samples = (np.sin(np.arange(16000) * .1) * 4000).astype('<i2')
        pcm = self.root / 'source.pcm'
        pcm.write_bytes(samples.tobytes())
        self.doc = Document(Source('episode.mp3', pcm, 16000, 1, len(samples), 0, self.cancel))
        self.prefs = Preferences(output_dir=str(self.root))

    def tearDown(self):
        self.doc.close()
        self.temp.cleanup()

    def export(self, path=None, acting='normal'):
        return export_audio(self.doc, 100, 8100, self.prefs, 'clean', acting, 'episode', path, self.cancel)

    def test_roundtrip_order_defaults_and_sequence(self):
        self.prefs.acting_tags = ['breath', 'normal']
        self.prefs.default_acting = 'breath'
        self.prefs.template = 'light_{source}_{acting}_{number}'
        self.prefs.digits = 3
        self.prefs.number = 12
        path = self.root / 'settings.json'
        save_preferences(path, self.prefs)
        restored = load_preferences(path)
        self.assertEqual(restored, self.prefs)
        self.assertEqual(restored.filename('clean', 'breath', 'episode', 12), 'light_episode_breath_012.wav')

    def test_existing_file_and_global_counter(self):
        existing = self.root / 'clean_normal_0001.wav'
        existing.write_bytes(b'KEEP')
        path, self.prefs.number = self.export()
        self.assertEqual(path.name, 'clean_normal_0002.wav')
        self.assertEqual(existing.read_bytes(), b'KEEP')
        with wave.open(str(path)) as wav:
            self.assertEqual((wav.getnframes(), wav.getframerate(), wav.getsampwidth()), (8000, 16000, 2))
            self.assertEqual(wav.readframes(8000), self.doc.source.data[100:8100].tobytes())
        save_preferences(self.root / 'settings.json', self.prefs)
        self.prefs = load_preferences(self.root / 'settings.json')
        path, number = self.export(acting='soft')
        self.assertEqual(path.name, 'clean_soft_0003.wav')
        self.assertEqual(number, 4)

    def test_manual_mp3_and_no_overwrite(self):
        target = self.root / 'laugh.mp3'
        path, number = self.export(target)
        self.assertEqual(number, 1)
        decoded = decode(path, self.temp.name, self.cancel)
        try:
            self.assertEqual(decoded.rate, 16000)
            self.assertGreater(decoded.frames, 7000)
        finally:
            decoded.close()
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.export(target)
        self.assertEqual(target.read_bytes(), original)

    def test_cancel_cleans_reserved_manual_target(self):
        self.cancel.set()
        target = self.root / 'cancel.wav'
        with self.assertRaises(Cancelled):
            self.export(target)
        self.assertFalse(target.exists())
        self.assertEqual(self.prefs.number, 1)

    def test_reject_unsafe_templates_and_tags(self):
        for template in ['../{number}', '{number.__class__}', '{unknown}_{number}', '{quality}', 'CON.{number}', '{number!r}']:
            self.prefs.template = template
            with self.subTest(template=template), self.assertRaises(ValueError):
                self.prefs.validate()
        self.prefs.template = '{number}'
        self.prefs.acting_tags.append('NORMAL')
        with self.assertRaises(ValueError):
            self.prefs.validate()

    def test_dialog_add_rename_move_delete_default_and_mode(self):
        dialog = SettingsDialog(self.prefs)
        editor = dialog.acting
        with patch.object(QInputDialog, 'getText', return_value=('breath', True)):
            editor.add()
        editor.default.setCurrentText('breath')
        with patch.object(QInputDialog, 'getText', return_value=('whisper', True)):
            editor.rename()
        self.assertEqual(editor.default.currentText(), 'whisper')
        editor.move(-1)
        self.assertEqual(editor.values()[-2], 'whisper')
        editor.tags.setCurrentRow(0)
        editor.remove()
        dialog.mode.setCurrentIndex(1)
        dialog.accept()
        result = dialog.result_preferences
        self.assertEqual(result.default_acting, 'whisper')
        self.assertNotIn('normal', result.acting_tags)
        self.assertEqual(result.save_mode, 'manual')
        dialog.close()

    def test_main_window_save_routes_restart_and_cancel(self):
        from app import MainWindow
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QFileDialog
        settings_path = self.root / 'settings.json'
        self.prefs.acting_tags.append('breath')
        self.prefs.default_acting = 'breath'
        save_preferences(settings_path, self.prefs)
        legacy = QSettings(str(self.root / 'legacy.ini'), QSettings.Format.IniFormat)
        with patch('app.QSettings', return_value=legacy):
            window = MainWindow(settings_path)
            try:
                self.assertEqual(window.delivery.currentText(), 'breath')
                window.document = self.doc.snapshot()
                window.paths = ['episode.mp3']
                window.current_index = 0
                window.file_list.addItem('episode.mp3')
                window.document.select((100, 8100))
                window.after_save.setCurrentIndex(2)
                def wait_job():
                    deadline = time.monotonic() + 10
                    while window.thread and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(.01)
                    self.assertIsNone(window.thread)
                with patch.object(QFileDialog, 'getSaveFileName', side_effect=AssertionError('auto opened dialog')):
                    window.save()
                    wait_job()
                self.assertTrue((self.root / 'clean_breath_0001.wav').exists())
                self.assertIsNone(window.document.state.selection)
                self.assertEqual(load_preferences(settings_path).number, 2)
                window.preferences.save_mode = 'manual'
                with patch.object(QFileDialog, 'getSaveFileName', return_value=('', '')):
                    window.save()
                    self.assertIsNone(window.thread)
                with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(self.root / 'custom'), 'MP3 (*.mp3)')):
                    window.save()
                    wait_job()
                self.assertTrue((self.root / 'custom.mp3').exists())
                self.assertEqual(load_preferences(settings_path).save_mode, 'manual')
            finally:
                window.close()
            restarted = MainWindow(settings_path)
            try:
                self.assertEqual(restarted.preferences.save_mode, 'manual')
                self.assertEqual(restarted.preferences.number, 2)
                self.assertEqual(restarted.delivery.currentText(), 'breath')
            finally:
                restarted.close()


if __name__ == '__main__':
    unittest.main()
