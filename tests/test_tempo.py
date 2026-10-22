import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Document, Source
from playback import TempoFeeder
from preferences import Preferences, save_preferences, load_preferences
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from app import MainWindow


class Output:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)
        return len(data)


class TempoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.pcm = self.folder / 'tone.pcm'
        samples = (np.sin(2 * np.pi * 440 * np.arange(16000 * 12) / 16000) * 6000).astype('<i2')
        self.original = samples.tobytes()
        self.pcm.write_bytes(self.original)
        self.doc = Document(Source('tone.wav', self.pcm, 16000, 1, len(samples), 0, threading.Event()))

    def tearDown(self):
        self.doc.close()
        self.temp.cleanup()

    def test_atempo_duration_pitch_and_unchanged_source(self):
        for speed in (.5, 1.25, 1.5, 2.0):
            with self.subTest(speed=speed):
                feeder = TempoFeeder(self.doc, 16000, 80000, speed)
                output = Output()
                try:
                    deadline = time.monotonic() + 10
                    while not feeder.finished and time.monotonic() < deadline:
                        feeder.feed(output, 8192)
                        time.sleep(.001)
                    self.assertTrue(feeder.finished)
                    self.assertIsNone(feeder.error)
                finally:
                    feeder.close()
                signal = np.frombuffer(output.data, dtype='<i2')
                self.assertAlmostEqual(len(signal) / 16000, 4 / speed, delta=.12)
                spectrum = abs(np.fft.rfft(signal * np.hanning(len(signal))))
                frequency = np.argmax(spectrum) * 16000 / len(signal)
                self.assertAlmostEqual(frequency, 440, delta=3)
        self.assertEqual(self.pcm.read_bytes(), self.original)
        from export_audio import export_audio
        path, _ = export_audio(self.doc, 16000, 80000,
            Preferences(output_dir=str(self.folder), playback_speed=2.0),
            'clean', 'normal', 'tone', None, threading.Event())
        with wave.open(str(path)) as wav:
            self.assertEqual(wav.getnframes(), 64000)
            self.assertEqual(wav.readframes(64000), self.original[32000:160000])

    def test_shortcuts_and_seek_settings_roundtrip_and_conflicts(self):
        from settings_dialog import SettingsDialog
        from PySide6.QtGui import QKeySequence
        dialog = SettingsDialog(Preferences())
        dialog.normal_key.setKeySequence(QKeySequence('F6'))
        dialog.fast_key.setKeySequence(QKeySequence('F7'))
        dialog.seek_seconds.setValue(8)
        prefs = dialog.collect()
        save_preferences(self.folder / 'settings.json', prefs)
        restored = load_preferences(self.folder / 'settings.json')
        self.assertEqual((restored.normal_speed_key, restored.fast_speed_key, restored.seek_seconds), ('F6', 'F7', 8))
        dialog.fast_key.setKeySequence(QKeySequence('Ctrl+S'))
        with self.assertRaises(ValueError):
            dialog.collect()
        dialog.close()

    def test_stop_with_full_queue_releases_threads(self):
        feeder = TempoFeeder(self.doc, 0, self.doc.frames, .5)
        time.sleep(.2)
        feeder.close()
        self.assertFalse(feeder.reader.is_alive())
        self.assertFalse(feeder.writer.is_alive())

    def test_gui_seek_pause_speed_selection_finish_and_persistence(self):
        config = self.folder / 'settings.json'
        save_preferences(config, Preferences(playback_speed=2.0, seek_seconds=5))
        legacy = QSettings(str(self.folder / 'legacy.ini'), QSettings.Format.IniFormat)
        with patch('app.QSettings', return_value=legacy):
            window = MainWindow(config)
        window.document = self.doc.snapshot()
        try:
            self.assertEqual(window.speed_combo.currentData(), 2.0)
            window.document.seek(16000 * 8)
            window.seek_relative(5)
            self.assertEqual(window.document.state.cursor, self.doc.frames)
            window.seek_relative(-60)
            self.assertEqual(window.document.state.cursor, 0)
            window.document.select((16000, 64000))
            with patch.object(window, 'play') as play:
                window.play_selection()
                play.assert_called_once_with(16000, 64000, selection_return=16000)
            # A paused selection retains its end and return position across speed changes.
            window.paused_range = None
            window.sink = SimpleNamespace(state=lambda: SimpleNamespace(name='SuspendedState'),
                                          reset=lambda: None, deleteLater=lambda: None)
            window.play_end = 64000
            window.selection_return = 16000
            window.document.seek(32000)
            window.speed_combo.setCurrentIndex(window.speed_combo.findData(1.25))
            self.assertEqual(load_preferences(config).playback_speed, 1.25)
            with patch.object(window, 'play') as play:
                window.toggle_play()
                play.assert_called_once_with(32000, 64000, selection_return=16000)
            window.paused_range = None
            # Simulated output clock: 0.5 seconds at 2x advances source time by 1 second.
            window.play_start, window.play_end = 16000, 64000
            window.selection_return = 16000
            window.active_speed = 2.0
            state = {'name': 'ActiveState', 'micros': 500000}
            window.sink = SimpleNamespace(state=lambda: SimpleNamespace(name=state['name']),
                error=lambda: SimpleNamespace(name='NoError'), processedUSecs=lambda: state['micros'],
                bytesFree=lambda: 8192, bufferSize=lambda: 8192,
                reset=lambda: None, deleteLater=lambda: None)
            window.feeder = SimpleNamespace(finished=False, feed=lambda *_: None)
            window.update_playback()
            self.assertEqual(window.document.state.cursor, 32000)
            with patch.object(window, 'play') as play:
                window.seek_relative(1)
                play.assert_called_once_with(48000, 64000, selection_return=16000)
            window.sink = SimpleNamespace(state=lambda: SimpleNamespace(name='IdleState'),
                error=lambda: SimpleNamespace(name='NoError'), processedUSecs=lambda: 1500000,
                bytesFree=lambda: 8192, bufferSize=lambda: 8192,
                reset=lambda: None, deleteLater=lambda: None)
            window.selection_return = 16000
            window.feeder = SimpleNamespace(finished=True, feed=lambda *_: None)
            window.update_playback()
            self.assertEqual(window.document.state.cursor, 16000)
            self.assertIsNone(window.sink)
            self.assertEqual(window.document.state.selection, (16000, 64000))
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
