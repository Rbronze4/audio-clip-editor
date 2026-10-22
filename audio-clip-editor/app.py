from __future__ import annotations

import sys
import tempfile
import threading
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import (QIODevice, QObject, QSettings, QThread, QTimer,
                            Qt, Signal)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QVBoxLayout, QWidget)
import pyqtgraph as pg

from core import Cancelled, Document, decode, export_wav
from playback import PCMFeeder, TempoFeeder, fatal_audio_error
from import_audio import collect_audio
from separation import separate
from preferences import Preferences, load_preferences, save_preferences
from settings_dialog import SettingsDialog
from export_audio import export_audio


def timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    minutes, rest = divmod(milliseconds, 60000)
    seconds, ms = divmod(rest, 1000)
    return f'{minutes:02d}:{seconds:02d}.{ms:03d}'


class Worker(QObject):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(int)
    message = Signal(str)
    finished = Signal()

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.cancel = threading.Event()

    def run(self):
        try:
            self.result.emit(self.function(self.cancel, self.progress.emit))
        except Cancelled:
            self.failed.emit('処理をキャンセルしました。')
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.finished.emit()


class PCMStream(QIODevice):
    """Pull only the requested bytes; no full-file playback buffer."""
    def __init__(self, document, start, end, parent=None):
        super().__init__(parent)
        self.iterator = iter(document.chunks(start, end, chunk_frames=8192))
        self.pending = b''
        self.remaining = (end - start) * document.source.channels * 2
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def isSequential(self):
        return True

    def bytesAvailable(self):
        return self.remaining + super().bytesAvailable()

    def atEnd(self):
        return self.remaining == 0

    def readData(self, maximum):
        wanted = min(maximum, self.remaining)
        parts, count = [], 0
        while count < wanted:
            if not self.pending:
                try:
                    self.pending = next(self.iterator).tobytes()
                except StopIteration:
                    break
            take = min(wanted - count, len(self.pending))
            parts.append(self.pending[:take])
            self.pending = self.pending[take:]
            count += take
        self.remaining -= count
        return b''.join(parts)

    def writeData(self, data):
        return -1


class WaveView(pg.ViewBox):
    selected = Signal(float, float)
    sought = Signal(float)
    menuRequested = Signal(object, float)

    def __init__(self):
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=False, y=False)
        self.duration = 1

    def time_at(self, position):
        return max(0., min(self.duration, self.mapToView(position).x()))

    def wheelEvent(self, event, axis=None):
        left, right = self.viewRange()[0]
        width = right - left
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            anchor = self.time_at(event.pos())
            new_width = min(max(.01, width * (0.8 if event.delta() > 0 else 1.25)),
                            max(.01, self.duration))
            fraction = (anchor - left) / width if width else .5
            new_left = anchor - fraction * new_width
            self.show_range(new_left, new_left + new_width)
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            shift = width * .15 * (-1 if event.delta() > 0 else 1)
            self.show_range(left + shift, right + shift)
        event.accept()

    def show_range(self, left, right):
        width = min(right - left, max(.01, self.duration))
        left = max(0, min(left, self.duration - width))
        self.setXRange(left, left + width, padding=0)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.sought.emit(self.time_at(event.pos()))
        elif event.button() == Qt.MouseButton.RightButton:
            self.menuRequested.emit(event.screenPos().toPoint(), self.time_at(event.pos()))
        event.accept()

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.MouseButton.LeftButton:
            a = self.time_at(event.buttonDownPos())
            b = self.time_at(event.pos())
            self.selected.emit(min(a, b), max(a, b))
        elif event.button() == Qt.MouseButton.MiddleButton:
            delta = self.mapToView(event.lastPos()).x() - self.mapToView(event.pos()).x()
            left, right = self.viewRange()[0]
            self.show_range(left + delta, right + delta)
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, settings_path=None):
        super().__init__()
        self.setWindowTitle('音声クリップ編集ツール v1.5.1')
        self.setAcceptDrops(True)
        self.resize(1180, 720)
        self.settings = QSettings('AudioClipEditor', 'AudioClipEditor')
        self.settings_path = Path(settings_path) if settings_path else Path(__file__).resolve().parent / 'settings.json'
        try:
            self.preferences = load_preferences(self.settings_path)
        except Exception as error:
            # Keep an unreadable settings file intact until the user explicitly edits settings.
            self.preferences = Preferences()
            self.settings_load_error = str(error)
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, '設定の読み込みに失敗しました',
                '初期設定で起動しました。次回設定保存時に元の設定ファイルのバックアップを残します。\n' + self.settings_load_error))
        if not self.settings_path.exists():
            self.preferences.output_dir = self.settings.value('output', '')
        self.cache = tempfile.TemporaryDirectory(prefix='audio_clip_')
        self.document = None
        self.history_views = {}
        self.thread = self.worker = None
        self.sink = self.stream = None
        self.feeder = None
        self.output_description = ''
        self.play_start = self.play_end = 0
        self.selection_return = None
        self.paused_range = None
        self.active_speed = 1.0
        self.paths = []
        self.saved = set()
        self.current_index = -1
        self.output_dir = self.preferences.output_dir
        self.actions = {}
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.update_playback)
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(20)
        self.redraw_timer.timeout.connect(self.draw_wave)
        self.view.sigXRangeChanged.connect(lambda: self.redraw_timer.start())
        self.refresh()

    def action(self, name, shortcut, callback):
        action = QAction(name, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        self.addAction(action)
        self.actions[name] = action
        return action

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton('音声を追加…')
        self.open_button.clicked.connect(self.open_files)
        toolbar.addWidget(self.open_button)
        self.import_folder_button = QPushButton('フォルダを追加…')
        self.import_folder_button.clicked.connect(self.open_folder)
        toolbar.addWidget(self.import_folder_button)
        self.folder_button = QPushButton('保存先…')
        self.folder_button.clicked.connect(self.choose_folder)
        toolbar.addWidget(self.folder_button)
        self.settings_button = QPushButton('設定…')
        self.settings_button.clicked.connect(self.edit_settings)
        toolbar.addWidget(self.settings_button)
        self.folder_label = QLabel(self.output_dir or '保存先が未設定です')
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        toolbar.addWidget(self.folder_label, 1)
        root.addLayout(toolbar)
        splitter = QSplitter()
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(170)
        self.file_list.itemClicked.connect(lambda item: self.load_index(self.file_list.row(item)))
        splitter.addWidget(self.file_list)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.filename = QLabel('音声ファイル・フォルダをドロップ、または「音声を追加」')
        layout.addWidget(self.filename)
        self.view = WaveView()
        self.plot = pg.PlotWidget(viewBox=self.view)
        self.plot.setBackground('#121a26')
        self.plot.setLabel('bottom', '時間', units='s')
        self.plot.setYRange(-1.05, 1.05, padding=0)
        self.plot.hideAxis('left')
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.low = self.plot.plot(pen=pg.mkPen('#63b9ec', width=1))
        self.high = self.plot.plot(pen=pg.mkPen('#63b9ec', width=1))
        self.fill = pg.FillBetweenItem(self.low, self.high, brush='#355e83')
        self.plot.addItem(self.fill)
        self.region = pg.LinearRegionItem((0, 0), movable=False,
                                         brush=pg.mkBrush(70, 165, 250, 60))
        self.region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        for line in self.region.lines:
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.region.setZValue(5)
        self.plot.addItem(self.region)
        self.head = pg.InfiniteLine(0, angle=90, pen=pg.mkPen('#ffc76a', width=2))
        self.head.setZValue(10)
        self.plot.addItem(self.head)
        self.view.selected.connect(self.select_range)
        self.view.sought.connect(self.seek)
        self.view.menuRequested.connect(self.context_menu)
        layout.addWidget(self.plot, 1)
        self.time_label = QLabel('00:00.000 / 00:00.000')
        self.selection_label = QLabel('選択なし — 保存すると編集後の全体を書き出します')
        layout.addWidget(self.time_label)
        layout.addWidget(self.selection_label)
        buttons = QHBoxLayout()
        specs = [('再生 / 一時停止', 'Space', self.toggle_play),
                 ('停止', None, self.stop),
                 ('選択部分を再生', 'Shift+Space', self.play_selection),
                 ('選択部分だけ残す', None, lambda: self.edit(True)),
                 ('選択部分を削除', 'Delete', lambda: self.edit(False)),
                 ('保存', 'Ctrl+S', self.save)]
        self.edit_buttons = []
        for title, key, callback in specs:
            action = self.action(title, key, callback)
            button = QPushButton(title)
            button.clicked.connect(action.trigger)
            buttons.addWidget(button)
            self.edit_buttons.append(button)
        layout.addLayout(buttons)
        separation_row = QHBoxLayout()
        self.clear_button = QPushButton('選択解除 (Esc)')
        self.clear_button.clicked.connect(self.clear_selection)
        separation_row.addWidget(self.clear_button)
        self.separate_scope = QComboBox()
        self.separate_scope.addItems(['選択範囲（選択なしなら全体）', '編集後の全体'])
        separation_row.addWidget(self.separate_scope)
        self.device_combo = QComboBox()
        self.device_combo.addItem('GPU自動 / なければCPU', 'auto')
        self.device_combo.addItem('CPU', 'cpu')
        self.device_combo.addItem('CUDA (NVIDIA GPU)', 'cuda')
        separation_row.addWidget(self.device_combo)
        self.bgm_button = QPushButton('BGM除去')
        self.bgm_button.clicked.connect(self.remove_bgm)
        separation_row.addWidget(self.bgm_button)
        self.version_combo = QComboBox()
        self.version_combo.addItem('A: Original', 'original')
        self.version_combo.addItem('B: BGM Removed', 'separated')
        self.version_combo.currentIndexChanged.connect(self.change_version)
        separation_row.addWidget(self.version_combo)
        layout.addLayout(separation_row)
        self.version_label = QLabel('再生・保存対象: A（元音声）')
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel('BGM処理方式'))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem('会話向け / MossFormer2', 'speech')
        self.engine_combo.addItem('音楽向け / Demucs（従来）', 'demucs')
        engine_row.addWidget(self.engine_combo)
        engine_row.addWidget(self.version_label, 1)
        layout.addLayout(engine_row)
        options = QHBoxLayout()
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('再生速度'))
        self.speed_combo = QComboBox()
        for speed in (.5, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f'{speed}x', speed)
        self.speed_combo.setCurrentIndex(self.speed_combo.findData(self.preferences.playback_speed))
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        speed_row.addWidget(self.speed_combo)
        self.speed_hint = QLabel()
        self.update_speed_hint()
        speed_row.addWidget(self.speed_hint)
        speed_row.addStretch()
        layout.addLayout(speed_row)
        options.addWidget(QLabel('音質タグ'))
        self.quality = QComboBox()
        self.quality.addItems(self.preferences.quality_tags)
        self.quality.setCurrentText(self.preferences.default_quality)
        options.addWidget(self.quality)
        options.addWidget(QLabel('演技タグ'))
        self.delivery = QComboBox()
        self.delivery.addItems(self.preferences.acting_tags)
        self.delivery.setCurrentText(self.preferences.default_acting)
        options.addWidget(self.delivery)
        options.addWidget(QLabel('保存後'))
        self.after_save = QComboBox()
        self.after_save.addItems(['そのまま表示', '次のファイルを開く', '選択解除'])
        options.addWidget(self.after_save)
        options.addStretch()
        layout.addLayout(options)
        self.save_mode_label = QLabel()
        self.update_save_label()
        layout.addWidget(self.save_mode_label)
        layout.addWidget(QLabel('Ctrl＋ホイール: カーソル中心に拡大縮小　Shift＋ホイール / 中ドラッグ: 横移動'))
        splitter.addWidget(panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 950])
        root.addWidget(splitter, 1)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.hide()
        self.cancel_button = QPushButton('キャンセル')
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_job)
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.cancel_button)
        root.addLayout(progress_row)
        menu = self.menuBar().addMenu('編集')
        menu.addAction(self.action('元に戻す', 'Ctrl+Z', self.undo))
        menu.addAction(self.action('やり直す', 'Ctrl+Y', self.redo))
        menu.addAction(self.action('選択解除', 'Esc', self.clear_selection))
        menu.addAction(self.action('全体表示', 'Ctrl+0', self.fit))
        self.action('開く', 'Ctrl+O', self.open_files)
        self.action('通常速度', self.preferences.normal_speed_key, lambda: self.speed_combo.setCurrentIndex(self.speed_combo.findData(1.0)))
        self.action('2倍速', self.preferences.fast_speed_key, lambda: self.speed_combo.setCurrentIndex(self.speed_combo.findData(2.0)))
        self.action('戻る', 'Left', lambda: self.seek_relative(-self.preferences.seek_seconds))
        self.action('進む', 'Right', lambda: self.seek_relative(self.preferences.seek_seconds))
        self.action('1秒戻る', 'Ctrl+Left', lambda: self.seek_relative(-1))
        self.action('1秒進む', 'Ctrl+Right', lambda: self.seek_relative(1))
        self.after_save.setCurrentIndex(int(self.settings.value('after_save', 0)))
        self.statusBar().showMessage('準備完了')

    def run_job(self, label, function, callback):
        if self.thread:
            return
        self.stop()
        self.thread = QThread(self)
        self.worker = Worker(function)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(callback)
        self.worker.failed.connect(self.job_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.worker.progress.connect(self.job_progress)
        self.worker.message.connect(self.statusBar().showMessage)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.show()
        self.statusBar().showMessage(label)
        self.refresh()
        self.thread.start()

    def cancel_job(self):
        if self.worker:
            self.worker.cancel.set()

    def job_progress(self, value):
        if value < 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def job_error(self, message):
        self.statusBar().showMessage(message)
        if not message.startswith('処理をキャンセル'):
            QMessageBox.warning(self, '処理できませんでした', message)

    def job_finished(self):
        self.thread.deleteLater()
        self.thread = self.worker = None
        self.progress.hide()
        self.cancel_button.hide()
        self.refresh()
        if getattr(self, 'next_pending', False):
            self.next_pending = False
            self.load_index(self.current_index + 1, confirm=False)
        elif getattr(self, 'pending_import_index', None) is not None:
            index = self.pending_import_index
            self.pending_import_index = None
            self.load_index(index)

    def open_files(self):
        if self.thread:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, '音声を追加', '',
            '音声ファイル (*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.wma *.aiff *.opus)')
        self.import_paths(paths)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '音声フォルダを追加（サブフォルダも対象）')
        if folder:
            self.import_paths([folder])

    def dragEnterEvent(self, event):
        if self.thread is None and event.mimeData().hasUrls() and any(
                url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if self.thread is not None:
            event.ignore()
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.import_paths(paths)

    def import_paths(self, paths):
        if not paths or self.thread:
            return
        existing = tuple(self.paths)
        def imported(result):
            files, failures = result
            first = len(self.paths)
            self.paths.extend(files)
            self.file_list.addItems(['○ ' + Path(path).name for path in files])
            for index in range(first, len(self.paths)):
                self.file_list.item(index).setToolTip(self.paths[index])
            # Adding files should not discard an ongoing edit.
            if files and self.document is None:
                self.pending_import_index = first
            message = f'{len(files)}件追加しました（重複はスキップ）'
            if failures:
                message += f' / 読み取れない項目: {len(failures)}件'
                QMessageBox.warning(self, '一部の項目を追加できませんでした', '\n'.join(failures[:10]))
            self.statusBar().showMessage(message)
        self.run_job('音声ファイルを検索中…',
                     lambda cancel, progress: collect_audio(paths, existing, cancel), imported)

    def remove_bgm(self, selection_only=False):
        if not self.document or self.thread or not self.document.frames:
            return
        selection = self.document.state.selection
        a, b = (selection if selection and (selection_only or self.separate_scope.currentIndex() == 0)
                else (0, self.document.frames))
        snapshot = self.document.snapshot()
        device = self.device_combo.currentData()
        engine = self.engine_combo.currentData()
        def operation(cancel, progress):
            def report(value, message):
                progress(value)
                self.worker.message.emit(message)
            return separate(snapshot, a, b, self.cache.name, device, cancel, report, engine=engine)
        def completed(result):
            original, separated = result
            bounds = self.remember_view()
            try:
                self.document.apply_separation(original, separated)
            except Exception as error:
                original.close()
                separated.close()
                self.job_error(str(error))
                return
            self.restore_edit_view(bounds)
            self.refresh()
            self.statusBar().showMessage('BGM除去完了 — A/Bで比較し、採用する音声を選んで保存してください')
        self.run_job('BGM除去の準備中…', operation, completed)

    def change_version(self, index):
        if not self.document or self.thread:
            return
        was_playing = self.sink is not None and self.sink.state().name != 'SuspendedState'
        end = self.play_end
        if self.sink:
            self.update_playback()
            was_playing = was_playing and self.sink is not None
        self.stop()
        self.document.switch_version(self.version_combo.itemData(index))
        self.draw_wave()
        self.refresh()
        if was_playing:
            self.play(self.document.state.cursor, end)

    def load_index(self, index, confirm=True):
        if self.thread or not 0 <= index < len(self.paths) or index == self.current_index:
            return
        if confirm and self.document and self.document.undo_stack:
            answer = QMessageBox.question(self, '音声の切り替え',
                '切り替えると現在の編集履歴は破棄されます。保存済みWAVは残ります。続けますか？')
            if answer != QMessageBox.StandardButton.Yes:
                self.file_list.setCurrentRow(self.current_index)
                return
        path = self.paths[index]
        def loaded(source):
            if self.document:
                self.document.close()
            self.document = Document(source)
            self.history_views.clear()
            self.current_index = index
            self.file_list.setCurrentRow(index)
            self.filename.setText(f'{Path(path).name}　｜　{source.rate} Hz / {source.channels} ch')
            self.fit()
            self.refresh()
            self.statusBar().showMessage('読み込み完了')
        self.run_job('PCMを読み込み中… / 波形を準備中…',
                     lambda cancel, progress: decode(path, self.cache.name, cancel, progress), loaded)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '保存先フォルダ', self.output_dir)
        if folder:
            updated = deepcopy(self.preferences)
            updated.output_dir = folder
            if not self.persist_preferences(updated):
                return
            self.output_dir = folder
            self.folder_label.setText(folder)

    def persist_preferences(self, updated):
        try:
            if getattr(self, 'settings_load_error', None):
                # Preserve invalid settings for recovery before replacing them.
                backup = self.settings_path.with_name('settings.invalid.json')
                with backup.open('xb') as file:
                    file.write(self.settings_path.read_bytes())
                self.settings_load_error = None
            save_preferences(self.settings_path, updated)
        except Exception as error:
            QMessageBox.warning(self, '設定を保存できませんでした', str(error))
            return False
        self.preferences = updated
        return True

    def update_save_label(self):
        mode = '自動保存: WAV / PCM 16bit' if self.preferences.save_mode == 'auto' else '毎回保存先を指定: WAV / MP3（320kbps）'
        self.save_mode_label.setText(mode + '　｜　AI学習用途ではWAV推奨・音量正規化 OFF')

    def edit_settings(self):
        if self.thread:
            return
        dialog = SettingsDialog(self.preferences, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        if not self.persist_preferences(dialog.result_preferences):
            return
        for combo, tags, default in [(self.quality, self.preferences.quality_tags, self.preferences.default_quality),
                                     (self.delivery, self.preferences.acting_tags, self.preferences.default_acting)]:
            selected = combo.currentText()
            combo.clear()
            combo.addItems(tags)
            combo.setCurrentText(selected if selected in tags else default)
        self.output_dir = self.preferences.output_dir
        self.folder_label.setText(self.output_dir or '保存先が未設定です')
        self.update_save_label()
        self.actions['通常速度'].setShortcut(QKeySequence(self.preferences.normal_speed_key))
        self.actions['2倍速'].setShortcut(QKeySequence(self.preferences.fast_speed_key))
        self.update_speed_hint()

    def update_speed_hint(self):
        self.speed_hint.setText(f'{self.preferences.normal_speed_key}: 通常速度　{self.preferences.fast_speed_key}: 2倍速　'
                               f'← / →: {self.preferences.seek_seconds}秒　Ctrl＋← / →: 1秒')

    def fit(self):
        duration = self.document.frames / self.document.source.rate if self.document else 1
        self.view.duration = duration
        self.view.show_range(0, max(.01, duration))
        self.draw_wave()

    def remember_view(self):
        bounds = tuple(self.view.viewRange()[0])
        state = self.document.state
        # Keep the state reference too, so Python cannot reuse its identity.
        self.history_views[id(state)] = (state, bounds)
        return bounds

    def restore_edit_view(self, bounds, anchor=None):
        self.view.duration = self.document.frames / self.document.source.rate
        left, right = bounds
        width = max(.01, right - left)
        if anchor is not None:
            left = min(left, anchor)
            if anchor > left + width:
                left = anchor - width
        # show_range only shifts at the file boundaries, shrinking the width
        # when the entire remaining clip is shorter than the visible window.
        self.view.show_range(left, left + width)
        self.draw_wave()

    def draw_wave(self):
        if not self.document:
            return
        rate = self.document.source.rate
        left, right = self.view.viewRange()[0]
        xs, low, high = self.document.envelope(left * rate, right * rate)
        self.low.setData(xs, low)
        self.high.setData(xs, high)

    def refresh(self):
        enabled = self.document is not None and self.thread is None
        for action in self.actions.values():
            action.setEnabled(enabled)
        self.actions['開く'].setEnabled(self.thread is None)
        for button in self.edit_buttons:
            button.setEnabled(enabled)
        self.open_button.setEnabled(self.thread is None)
        self.import_folder_button.setEnabled(self.thread is None)
        self.folder_button.setEnabled(self.thread is None)
        self.settings_button.setEnabled(self.thread is None)
        self.speed_combo.setEnabled(self.thread is None)
        self.file_list.setEnabled(self.thread is None)
        self.plot.setEnabled(self.thread is None)
        self.clear_button.setEnabled(enabled and self.document.state.selection is not None)
        self.bgm_button.setEnabled(enabled and self.document.frames > 0)
        self.separate_scope.setEnabled(enabled)
        self.device_combo.setEnabled(self.thread is None)
        self.engine_combo.setEnabled(self.thread is None)
        self.version_combo.setEnabled(enabled and self.document.state.separated is not None)
        self.version_combo.blockSignals(True)
        self.version_combo.setCurrentIndex(1 if self.document and self.document.state.version == 'separated' else 0)
        self.version_combo.blockSignals(False)
        self.version_label.setText('再生・保存対象: ' + (
            'B（BGM除去後）' if self.document and self.document.state.version == 'separated' else 'A（元音声）'))
        if not self.document:
            self.region.hide()
            return
        doc = self.document
        rate = doc.source.rate
        self.actions['元に戻す'].setEnabled(enabled and bool(doc.undo_stack))
        self.actions['やり直す'].setEnabled(enabled and bool(doc.redo_stack))
        self.head.setValue(doc.state.cursor / rate)
        self.time_label.setText(f'{timestamp(doc.state.cursor / rate)} / {timestamp(doc.frames / rate)}')
        selection = doc.state.selection
        self.region.setVisible(selection is not None)
        if selection:
            a, b = selection
            self.region.setRegion((a / rate, b / rate))
            self.selection_label.setText(
                f'開始: {timestamp(a / rate)}　終了: {timestamp(b / rate)}　長さ: {(b-a)/rate:.3f}秒')
        else:
            self.selection_label.setText('選択なし — 保存すると編集後の全体を書き出します')

    def select_range(self, a, b):
        if self.document and not self.thread:
            self.stop()
            rate = self.document.source.rate
            self.document.select((round(a * rate), round(b * rate)))
            self.refresh()

    def clear_selection(self):
        if self.document and not self.thread:
            self.document.select(None)
            self.refresh()

    def seek(self, seconds):
        if self.document and not self.thread:
            self.stop()
            self.document.seek(round(seconds * self.document.source.rate))
            self.refresh()

    def context_menu(self, point, seconds):
        if not self.document or self.thread:
            return
        selected = self.document.state.selection
        if not selected or not selected[0] <= seconds * self.document.source.rate <= selected[1]:
            return
        menu = QMenu(self)
        menu.addAction('選択範囲のBGM除去', lambda: self.remove_bgm(selection_only=True))
        for title in ['選択部分を再生', '選択部分だけ残す', '選択部分を削除', '保存', '選択解除']:
            if title == '保存':
                menu.addAction('選択部分を書き出す', self.save)
            else:
                menu.addAction(self.actions[title])
        menu.exec(point)

    def edit(self, keep):
        if not self.document or self.thread:
            return
        self.stop()
        if not self.document.state.selection:
            return
        bounds = self.remember_view()
        start = self.document.state.selection[0] / self.document.source.rate
        if self.document.edit(keep):
            if keep:
                bounds = (bounds[0] - start, bounds[1] - start)
            self.restore_edit_view(bounds, self.document.state.cursor / self.document.source.rate)
            self.refresh()

    def undo(self):
        if self.document and not self.thread and self.document.undo_stack:
            self.stop()
            bounds = self.remember_view()
            self.document.undo()
            self.restore_edit_view(self.history_views.get(id(self.document.state), (None, bounds))[1])
            self.refresh()

    def redo(self):
        if self.document and not self.thread and self.document.redo_stack:
            self.stop()
            bounds = self.remember_view()
            self.document.redo()
            self.restore_edit_view(self.history_views.get(id(self.document.state), (None, bounds))[1])
            self.refresh()

    def toggle_play(self):
        if not self.document or self.thread:
            return
        if self.sink:
            if self.sink.state().name == 'SuspendedState':
                self.sink.resume()
                self.timer.start()
            else:
                self.update_playback()
                if self.sink:
                    self.sink.suspend()
                    self.timer.stop()
        else:
            start = self.document.state.cursor
            if start >= self.document.frames:
                start = 0
            end, return_to = self.paused_range or (self.document.frames, None)
            self.play(start, end, selection_return=return_to)

    def play_selection(self):
        if self.document and self.document.state.selection and not self.thread:
            self.play(*self.document.state.selection, selection_return=self.document.state.selection[0])

    def change_speed(self):
        speed = self.speed_combo.currentData()
        running = self.sink is not None and self.sink.state().name != 'SuspendedState'
        if running:
            self.update_playback()
        active = self.sink is not None
        end, return_to = self.paused_range or (self.play_end, self.selection_return)
        paused_range = self.paused_range
        self.stop()
        updated = deepcopy(self.preferences)
        updated.playback_speed = speed
        self.preferences = updated
        self.persist_preferences(updated)
        if active and running:
            self.play(self.document.state.cursor, end, selection_return=return_to)
        elif active:
            # Retain selection boundaries across a paused speed change.
            self.paused_range = (end, return_to)
        elif paused_range:
            self.paused_range = paused_range

    def seek_relative(self, seconds):
        if not self.document or self.thread:
            return
        running = self.sink is not None and self.sink.state().name != 'SuspendedState'
        if running:
            self.update_playback()
        target = max(0, min(self.document.frames,
            self.document.state.cursor + round(seconds * self.document.source.rate)))
        end, return_to = self.paused_range or (self.play_end, self.selection_return)
        self.stop()
        self.document.seek(target)
        # A seek outside the preview selection returns to normal playback.
        if return_to is None or not return_to <= target < end:
            end, return_to = self.document.frames, None
        self.paused_range = (end, return_to)
        if running and target < self.document.frames:
            self.play(target, end, selection_return=return_to)
        self.refresh()
        self.head.setValue(target / self.document.source.rate)

    def play(self, start, end, selection_return=None):
        self.stop()
        if end <= start:
            return
        source = self.document.source
        device = QMediaDevices.defaultAudioOutput()
        format = QAudioFormat()
        format.setSampleRate(source.rate)
        format.setChannelCount(source.channels)
        format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isNull() or not device.isFormatSupported(format):
            QMessageBox.warning(self, '再生デバイス',
                '出力デバイスがないか、このサンプルレート／チャンネル数に対応していません。\n'
                'Windowsのサウンド設定で対応する出力デバイスを選んでください。保存は可能です。')
            return
        self.play_start, self.play_end = start, end
        self.document.seek(start)
        self.refresh()
        self.selection_return = selection_return
        self.active_speed = self.preferences.playback_speed
        self.output_description = device.description()
        try:
            self.feeder = (PCMFeeder(self.document, start, end) if self.active_speed == 1.0 else
                           TempoFeeder(self.document, start, end, self.active_speed))
        except Exception as error:
            self.job_error(str(error))
            return
        self.sink = QAudioSink(device, format, self)
        self.sink.setBufferSize(max(8192, source.rate * source.channels * 2 // 4))
        # Push from the GUI thread. Pull-mode Python QIODevice callbacks can
        # contend with the GUI/GIL on the native audio thread on Windows.
        self.stream = self.sink.start()
        if self.stream is None or fatal_audio_error(self.sink.error()):
            self.playback_error('音声出力を開始できませんでした。')
            return
        if not self.feed_audio():
            return
        self.timer.start()

    def playback_error(self, message):
        code = self.sink.error().name if self.sink else 'WriteError'
        details = (f'{message}\n出力先: {self.output_description}\n'
                   f'詳細: {code}\nWindowsのサウンド設定で出力先を確認してください。')
        self.stop()
        QMessageBox.warning(self, '再生エラー', details)

    def feed_audio(self):
        try:
            self.feeder.feed(self.stream, self.sink.bytesFree())
            return True
        except Exception as error:
            self.playback_error(str(error))
            return False

    def update_playback(self):
        if not self.sink or not self.document:
            return
        if fatal_audio_error(self.sink.error()):
            self.playback_error('音声出力が停止しました。')
            return
        if self.sink.state().name == 'SuspendedState':
            return
        if self.sink.state().name == 'StoppedState':
            self.playback_error('音声出力が予期せず停止しました。')
            return
        if not self.feed_audio():
            return
        rate = self.document.source.rate
        cursor = min(self.play_end, self.play_start + round(self.sink.processedUSecs() * rate * self.active_speed / 1000000))
        self.document.seek(cursor)
        self.refresh()
        left, right = self.view.viewRange()[0]
        seconds = cursor / rate
        if seconds > right or seconds < left:
            self.view.show_range(seconds, seconds + right - left)
        # Idle alone is not EOF: it can be a temporary underrun. Refill above
        # and finish only after all selected samples have actually drained.
        if self.feeder.finished and self.sink.state().name == 'IdleState' and self.sink.bytesFree() >= self.sink.bufferSize():
            self.document.seek(self.selection_return if self.selection_return is not None else self.play_end)
            self.stop()
            self.refresh()

    def stop(self):
        self.timer.stop()
        # start() returns an object owned by QAudioSink; stop invalidates it.
        self.stream = None
        if isinstance(self.feeder, TempoFeeder):
            self.feeder.close()
        self.feeder = None
        self.paused_range = None
        self.selection_return = None
        if self.sink:
            self.sink.reset()
            self.sink.deleteLater()
            self.sink = None

    def save(self):
        if not self.document or self.thread or not self.document.frames:
            return
        manual_path = None
        if self.preferences.save_mode == 'auto':
            if not self.output_dir:
                self.choose_folder()
            if not self.output_dir:
                return
        else:
            try:
                suggested = self.preferences.filename(self.quality.currentText(), self.delivery.currentText(),
                    Path(self.paths[self.current_index]).stem, self.preferences.number)
            except ValueError as error:
                self.job_error(str(error))
                return
            path, selected_filter = QFileDialog.getSaveFileName(self, '名前を付けて保存',
                str(Path(self.preferences.manual_dir or self.output_dir) / Path(suggested).stem),
                'WAV (*.wav);;MP3 (*.mp3)', options=QFileDialog.Option.DontConfirmOverwrite)
            if not path:
                return
            manual_path = Path(path)
            if not manual_path.suffix:
                manual_path = manual_path.with_suffix('.mp3' if selected_filter.startswith('MP3') else '.wav')
            if manual_path.exists():
                self.job_error('同名のファイルがあります。別の名前を指定してください。')
                return
        a, b = self.document.state.selection or (0, self.document.frames)
        prefs = deepcopy(self.preferences)
        quality, acting = self.quality.currentText(), self.delivery.currentText()
        source = Path(self.paths[self.current_index]).stem
        behavior = self.after_save.currentIndex()
        def saved(result):
            path, next_number = result
            updated = deepcopy(prefs)
            updated.number = next_number
            if manual_path is not None:
                updated.manual_dir = str(path.parent)
            # Keep the in-memory sequence advancing even if settings storage fails.
            self.preferences = updated
            self.persist_preferences(updated)
            self.saved.add(self.current_index)
            self.file_list.item(self.current_index).setText('✓ ' + Path(self.paths[self.current_index]).name)
            self.statusBar().showMessage(f'保存完了: {path}')
            if behavior == 2:
                self.document.select(None)
            self.next_pending = behavior == 1
        self.run_job('音声を書き出し中…',
                     lambda cancel, progress: export_audio(self.document, a, b, prefs, quality, acting,
                         source, manual_path, cancel, progress), saved)

    def closeEvent(self, event):
        if self.thread:
            self.cancel_job()
            self.statusBar().showMessage('処理をキャンセル中です。完了後に閉じてください。')
            event.ignore()
            return
        if self.document and self.document.undo_stack:
            answer = QMessageBox.question(self, '終了',
                '編集履歴は終了時に破棄されます。必要な音声を書き出しましたか？')
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.stop()
        for combo, key in [(self.quality, 'quality'), (self.delivery, 'delivery'),
                           (self.after_save, 'after_save')]:
            self.settings.setValue(key, combo.currentIndex())
        if self.document:
            self.document.close()
        self.cache.cleanup()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
