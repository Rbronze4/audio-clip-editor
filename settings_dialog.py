from copy import deepcopy

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget, QKeySequenceEdit)
from PySide6.QtGui import QKeySequence

from preferences import valid_name


class TagEditor(QWidget):
    def __init__(self, tags, default):
        super().__init__()
        layout = QVBoxLayout(self)
        self.tags = QListWidget()
        self.tags.addItems(tags)
        self.tags.setCurrentRow(0)
        layout.addWidget(self.tags)
        row = QHBoxLayout()
        for name, slot in [('追加', self.add), ('名前変更', self.rename), ('削除', self.remove),
                           ('↑', lambda: self.move(-1)), ('↓', lambda: self.move(1))]:
            button = QPushButton(name)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(QLabel('起動時のデフォルト選択'))
        self.default = QComboBox()
        self.default.addItems(tags)
        self.default.setCurrentText(default)
        layout.addWidget(self.default)
        self.tags.itemDoubleClicked.connect(self.rename)

    def values(self):
        return [self.tags.item(i).text() for i in range(self.tags.count())]

    def sync(self, preferred=None):
        selected = preferred or self.default.currentText()
        self.default.clear()
        self.default.addItems(self.values())
        if selected in self.values():
            self.default.setCurrentText(selected)

    def ask(self, old=''):
        text, ok = QInputDialog.getText(self, 'タグ名', 'ファイル名に使用するタグ', text=old)
        if not ok:
            return None
        try:
            valid_name(text)
            if any(t.casefold() == text.casefold() and t != old for t in self.values()):
                raise ValueError('同名のタグが既にあります。')
            return text
        except ValueError as error:
            QMessageBox.warning(self, 'タグ名を確認してください', str(error))

    def add(self):
        text = self.ask()
        if text is not None:
            self.tags.addItem(text)
            self.tags.setCurrentRow(self.tags.count() - 1)
            self.sync()

    def rename(self, *_):
        item = self.tags.currentItem()
        if item is None:
            return
        old = item.text()
        text = self.ask(old)
        if text is not None:
            item.setText(text)
            self.sync(text if self.default.currentText() == old else None)

    def remove(self):
        if self.tags.count() <= 1:
            QMessageBox.warning(self, 'タグ', '少なくとも1つのタグを残してください。')
            return
        if self.tags.currentRow() >= 0:
            self.tags.takeItem(self.tags.currentRow())
            self.sync()

    def move(self, delta):
        index = self.tags.currentRow()
        if 0 <= index + delta < self.tags.count() and index >= 0:
            item = self.tags.takeItem(index)
            self.tags.insertItem(index + delta, item)
            self.tags.setCurrentRow(index + delta)
            self.sync()


class SettingsDialog(QDialog):
    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle('タグ・保存設定')
        self.resize(650, 510)
        self.result_preferences = deepcopy(preferences)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.quality = TagEditor(preferences.quality_tags, preferences.default_quality)
        self.acting = TagEditor(preferences.acting_tags, preferences.default_acting)
        tabs.addTab(self.quality, '音質タグ')
        tabs.addTab(self.acting, '演技タグ')
        page = QWidget()
        form = QFormLayout(page)
        self.mode = QComboBox()
        self.mode.addItem('自動保存（WAV）', 'auto')
        self.mode.addItem('毎回保存先とファイル名を指定', 'manual')
        self.mode.setCurrentIndex(self.mode.findData(preferences.save_mode))
        form.addRow('保存方式', self.mode)
        row = QHBoxLayout()
        self.folder = QLineEdit(preferences.output_dir)
        row.addWidget(self.folder)
        browse = QPushButton('参照…')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        form.addRow('自動保存先', row)
        self.template = QLineEdit(preferences.template)
        form.addRow('ファイル名ルール', self.template)
        hint = QLabel('{quality} 音質 / {acting} 演技 / {number} 連番（必須）\n{source} 元ファイル名（拡張子なし）\n拡張子 .wav は自動で付加します。AI学習用途ではWAV推奨。')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.digits = QSpinBox()
        self.digits.setRange(1, 10)
        self.digits.setValue(preferences.digits)
        form.addRow('連番桁数（最小桁数）', self.digits)
        self.number = QSpinBox()
        self.number.setRange(1, 2147483647)
        self.number.setValue(preferences.number)
        form.addRow('次回の連番', self.number)
        form.addRow(QLabel('連番はタグをまたいで進みます。既存ファイルの番号は飛ばします。'))
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        form.addRow('名前のプレビュー', self.preview)
        for signal in [self.template.textChanged, self.digits.valueChanged, self.number.valueChanged,
                       self.quality.default.currentTextChanged, self.acting.default.currentTextChanged]:
            signal.connect(self.update_preview)
        tabs.addTab(page, '保存')
        playback = QWidget()
        playback_form = QFormLayout(playback)
        self.seek_seconds = QSpinBox()
        self.seek_seconds.setRange(1, 60)
        self.seek_seconds.setValue(preferences.seek_seconds)
        self.seek_seconds.setSuffix(' 秒')
        playback_form.addRow('← / → のシーク秒数', self.seek_seconds)
        self.normal_key = QKeySequenceEdit(QKeySequence(preferences.normal_speed_key))
        self.fast_key = QKeySequenceEdit(QKeySequence(preferences.fast_speed_key))
        playback_form.addRow('1.0倍速に切り替えるキー', self.normal_key)
        playback_form.addRow('2.0倍速に切り替えるキー', self.fast_key)
        playback_form.addRow(QLabel('Ctrl＋← / →: 1秒移動\n速度はプレビューのみに適用され、最後の速度を記憶します。'))
        tabs.addTab(playback, '再生')
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_preview()

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, '自動保存先', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def collect(self):
        result = deepcopy(self.result_preferences)
        result.quality_tags = self.quality.values()
        result.acting_tags = self.acting.values()
        result.default_quality = self.quality.default.currentText()
        result.default_acting = self.acting.default.currentText()
        result.save_mode = self.mode.currentData()
        result.output_dir = self.folder.text().strip()
        result.template = self.template.text()
        result.digits = self.digits.value()
        result.number = self.number.value()
        if hasattr(self, 'seek_seconds'):
            result.seek_seconds = self.seek_seconds.value()
        if hasattr(self, 'normal_key'):
            result.normal_speed_key = self.normal_key.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            result.fast_speed_key = self.fast_key.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        return result.validate()

    def update_preview(self, *_):
        try:
            prefs = self.collect()
            self.preview.setText(prefs.filename(prefs.default_quality, prefs.default_acting, 'episode01', prefs.number))
        except ValueError as error:
            self.preview.setText(str(error))

    def accept(self):
        try:
            self.result_preferences = self.collect()
        except ValueError as error:
            QMessageBox.warning(self, '設定を確認してください', str(error))
            return
        super().accept()
