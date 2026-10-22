"""Persistent user preferences and Windows-safe export names."""
import json
import os
import re
import string
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


def valid_name(value):
    if (not value or value != value.strip() or value.endswith('.') or
            re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or len(value) > 200 or
            re.match(r'^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)', value, re.I)):
        raise ValueError('名前に使用できない文字・末尾の空白・予約名が含まれています: ' + value)
    return value


@dataclass
class Preferences:
    quality_tags: list = field(default_factory=lambda: ['clean', 'dirty'])
    acting_tags: list = field(default_factory=lambda: ['normal', 'serious', 'soft', 'other'])
    default_quality: str = 'clean'
    default_acting: str = 'normal'
    save_mode: str = 'auto'
    output_dir: str = ''
    template: str = '{quality}_{acting}_{number}'
    digits: int = 4
    number: int = 1
    manual_dir: str = ''
    playback_speed: float = 1.0
    seek_seconds: int = 5
    normal_speed_key: str = '1'
    fast_speed_key: str = '2'

    def validate(self):
        from PySide6.QtGui import QKeySequence
        reserved = {'Space', 'Shift+Space', 'Left', 'Right', 'Ctrl+Left', 'Ctrl+Right',
                    'Esc', 'Delete', 'Ctrl+Z', 'Ctrl+Y', 'Ctrl+S', 'Ctrl+O', 'Ctrl+0'}
        keys = []
        for key in (self.normal_speed_key, self.fast_speed_key):
            if not isinstance(key, str):
                raise ValueError('速度ショートカットが不正です。')
            sequence = QKeySequence(key)
            canonical = sequence.toString(QKeySequence.SequenceFormat.PortableText)
            if sequence.count() != 1 or not canonical or canonical in reserved:
                raise ValueError('速度キーには他の操作と重複しないキーを指定してください。')
            keys.append(canonical)
        if keys[0] == keys[1]:
            raise ValueError('通常速度と2倍速には別のキーを指定してください。')
        if self.playback_speed not in (.5, 1.0, 1.25, 1.5, 2.0):
            raise ValueError('再生速度が不正です。')
        if type(self.seek_seconds) is not int or not 1 <= self.seek_seconds <= 60:
            raise ValueError('シーク秒数は1〜60秒で指定してください。')
        for tags, default in [(self.quality_tags, self.default_quality), (self.acting_tags, self.default_acting)]:
            if not isinstance(tags, list) or not tags or not all(isinstance(t, str) for t in tags):
                raise ValueError('各タグ一覧には1つ以上のタグが必要です。')
            for tag in tags:
                valid_name(tag)
            if len({t.casefold() for t in tags}) != len(tags) or default not in tags:
                raise ValueError('タグの重複、またはデフォルトタグの指定が不正です。')
        if self.save_mode not in ('auto', 'manual'):
            raise ValueError('保存方式が不正です。')
        if type(self.digits) is not int or not 1 <= self.digits <= 10 or type(self.number) is not int or not 1 <= self.number <= 2147483647:
            raise ValueError('連番・桁数が範囲外です。')
        if not all(isinstance(v, str) for v in [self.output_dir, self.manual_dir, self.template]):
            raise ValueError('保存先・テンプレートが不正です。')
        fields = []
        for _, name, spec, conversion in string.Formatter().parse(self.template):
            if name is not None:
                if name not in ('quality', 'acting', 'number', 'source') or spec or conversion:
                    raise ValueError('使用できる項目は {quality} {acting} {number} {source} です。')
                fields.append(name)
        if 'number' not in fields:
            raise ValueError('テンプレートには {number} が必要です。')
        self.filename(self.default_quality, self.default_acting, 'source', self.number)
        return self

    def filename(self, quality, acting, source, number):
        return valid_name(self.template.format(quality=quality, acting=acting,
                          source=source, number=str(number).zfill(self.digits))) + '.wav'


def load_preferences(path):
    path = Path(path)
    if not path.exists():
        return Preferences()
    return Preferences(**json.loads(path.read_text(encoding='utf-8'))).validate()


def save_preferences(path, preferences):
    preferences.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='settings-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(asdict(preferences), file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
