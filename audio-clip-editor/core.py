"""PCM storage, non-destructive edits and exclusive sequential WAV export."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, replace

import numpy as np


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event.is_set():
        raise Cancelled("処理をキャンセルしました。")


@dataclass(frozen=True)
class EditState:
    spans: tuple[tuple[int, int], ...]
    selection: tuple[int, int] | None = None
    cursor: int = 0
    original: object = None
    separated: object = None
    version: str = 'original'


class Source:
    def __init__(self, path, pcm_path, rate, channels, frames, offset, cancel):
        self.path = Path(path)
        self.pcm_path = Path(pcm_path)
        self.rate, self.channels, self.frames = rate, channels, frames
        self.data = np.memmap(pcm_path, dtype='<i2', mode='r', offset=offset,
                              shape=(frames, channels))
        try:
            self._build_levels(cancel)
        except BaseException:
            self.close()
            raise

    def _build_levels(self, cancel):
        frames = self.frames
        # Cached extrema preserve narrow transients and both stereo channels.
        self.levels = {}
        block = 256
        lows, highs = [], []
        for start in range(0, frames, block * 4096):
            check_cancel(cancel)
            a = self.data[start:min(frames, start + block * 4096)]
            lo, hi = a.min(axis=1), a.max(axis=1)
            indices = np.arange(0, len(a), block)
            lows.append(np.minimum.reduceat(lo, indices))
            highs.append(np.maximum.reduceat(hi, indices))
        low, high = np.concatenate(lows), np.concatenate(highs)
        self.levels[block] = (low, high)
        while len(low) > 2048:
            check_cancel(cancel)
            indices = np.arange(0, len(low), 4)
            low = np.minimum.reduceat(low, indices)
            high = np.maximum.reduceat(high, indices)
            block *= 4
            self.levels[block] = (low, high)

    def close(self):
        if getattr(self, 'data', None) is not None:
            self.data._mmap.close()
            self.data = None
        self.pcm_path.unlink(missing_ok=True)


def find_ffmpeg():
    """Find an app-local binary, PATH executable, or pip-packaged executable."""
    executable = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    app_dir = Path(__file__).resolve().parent
    for candidate in (app_dir / 'bin' / executable, app_dir / executable):
        if candidate.is_file():
            return str(candidate)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError(
            'FFmpegが見つかりません。アプリを閉じて、同じフォルダのsetup.batを実行してください。\n'
            'FFmpegを含む依存パッケージをインストールします。PATHの設定は不要です。\n'
            '手動配置する場合は、このアプリのbinフォルダへffmpeg.exeを入れてください。'
        ) from error


def decode(path, cache_dir, cancel, progress=lambda value: None):
    ffmpeg = find_ffmpeg()
    fd, name = tempfile.mkstemp(suffix='.wav', dir=cache_dir)
    os.close(fd)
    source = None
    try:
        # A file-backed log avoids a full stderr pipe blocking the subprocess.
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                [ffmpeg, '-nostdin', '-v', 'error', '-y', '-i', str(path),
                 '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', '-rf64', 'never', name],
                stdout=subprocess.DEVNULL, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                while process.poll() is None:
                    check_cancel(cancel)
                    time.sleep(.05)
                log.seek(0)
                error = log.read().decode('utf-8', errors='replace')
                if process.returncode:
                    raise RuntimeError('音声の読み込みに失敗しました。\n' + error[-2500:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        check_cancel(cancel)
        if Path(name).stat().st_size >= 2**32:
            raise RuntimeError('デコード後のWAVが4GBを超えています。入力音声を分割してください。')
        # wave supports RIFF ancillary chunks; data need not begin at byte 44.
        with wave.open(name, 'rb') as wav:
            rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
            if wav.getsampwidth() != 2 or not frames:
                raise RuntimeError('空の音声、または未対応のPCM形式です。')
            # Locate the data chunk independently of wave's private internals.
        with open(name, 'rb') as stream:
            stream.seek(12)
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    raise RuntimeError('WAVデータが破損しています。')
                size = int.from_bytes(header[4:], 'little')
                if header[:4] == b'data':
                    offset = stream.tell()
                    break
                stream.seek(size + size % 2, 1)
        progress(50)
        source = Source(path, name, rate, channels, frames, offset, cancel)
        progress(100)
        return source
    except BaseException:
        if source is not None:
            source.close()
        else:
            # On Windows a failed Source constructor may hold a mmap until GC.
            import gc
            gc.collect()
            try:
                Path(name).unlink(missing_ok=True)
            except PermissionError:
                pass
        raise


class Document:
    def __init__(self, source):
        self._sources = [source]
        self.state = EditState(((0, source.frames),), original=source)
        self.undo_stack = []
        self.redo_stack = []

    @property
    def source(self):
        return (self.state.separated if self.state.version == 'separated'
                and self.state.separated is not None else self.state.original)

    def snapshot(self, version=None):
        other = Document(self.state.original)
        other.state = replace(self.state, version=version or self.state.version)
        other._sources = []  # Borrowed references; only the owning document closes them.
        return other

    def switch_version(self, version):
        if version not in ('original', 'separated'):
            raise ValueError('Unknown audio version')
        if version == 'separated' and self.state.separated is None:
            return
        self.state = replace(self.state, version=version)

    def apply_separation(self, original, separated):
        if (original.frames != self.frames or separated.frames != self.frames or
                original.rate != self.source.rate or separated.rate != self.source.rate or
                original.channels != self.source.channels or separated.channels != self.source.channels):
            raise ValueError('分離音声の長さ・サンプルレート・チャンネル数が一致しません。')
        self.undo_stack.append(self.state)
        self.redo_stack.clear()
        self._sources.extend([original, separated])
        self.state = replace(self.state, spans=((0, self.frames),), original=original,
                             separated=separated, version='separated')

    def close(self):
        for source in self._sources:
            source.close()
        self._sources.clear()

    @property
    def frames(self):
        return sum(b - a for a, b in self.state.spans)

    def slice_spans(self, start, end, spans=None):
        result, offset = [], 0
        for a, b in self.state.spans if spans is None else spans:
            length = b - a
            left, right = max(start, offset), min(end, offset + length)
            if right > left:
                result.append((a + left - offset, a + right - offset))
            offset += length
        return tuple(result)

    def select(self, selection):
        if selection is not None:
            a, b = sorted(selection)
            a, b = max(0, min(a, self.frames)), max(0, min(b, self.frames))
            selection = (a, b) if b > a else None
        self.state = replace(self.state, selection=selection)

    def seek(self, frame):
        self.state = replace(self.state, cursor=max(0, min(self.frames, frame)))

    def edit(self, keep):
        if not self.state.selection:
            return False
        a, b = self.state.selection
        spans = self.slice_spans(a, b) if keep else (
            self.slice_spans(0, a) + self.slice_spans(b, self.frames))
        self.undo_stack.append(self.state)
        self.redo_stack.clear()
        self.state = replace(self.state, spans=spans, selection=None,
                             cursor=0 if keep else min(a, self.frames - (b - a)))
        return True

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.state)
            self.state = self.undo_stack.pop()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.state)
            self.state = self.redo_stack.pop()

    def chunks(self, start=0, end=None, spans=None, chunk_frames=32768):
        for a, b in self.slice_spans(start, self.frames if end is None else end, spans):
            for pos in range(a, b, chunk_frames):
                yield self.source.data[pos:min(b, pos + chunk_frames)]

    def envelope(self, start, end, pixels=1800):
        start, end = max(0, int(start)), min(self.frames, int(end))
        if end <= start:
            return [], [], []
        desired = max(1, (end - start) // pixels)
        block = max([1] + [b for b in self.source.levels if b <= desired])
        xs, lows, highs = [], [], []
        offset = start
        for a, b in self.slice_spans(start, end):
            if block == 1:
                data = self.source.data[a:b]
                low, high = data.min(axis=1), data.max(axis=1)
                indices = np.arange(0, b - a, desired)
                low, high = np.minimum.reduceat(low, indices), np.maximum.reduceat(high, indices)
                x = indices + offset
            else:
                # Boundary fragments use exact samples, never deleted neighbours.
                first, last = (a + block - 1) // block, b // block
                l, h, positions = [], [], []
                if first >= last:
                    fragment = self.source.data[a:b]
                    l, h, positions = [fragment.min()], [fragment.max()], [offset]
                else:
                    if a < first * block:
                        fragment = self.source.data[a:first * block]
                        l.append(fragment.min()); h.append(fragment.max()); positions.append(offset)
                    cached_l, cached_h = self.source.levels[block]
                    l.extend(cached_l[first:last]); h.extend(cached_h[first:last])
                    positions.extend(np.arange(first, last) * block - a + offset)
                    if last * block < b:
                        fragment = self.source.data[last * block:b]
                        l.append(fragment.min()); h.append(fragment.max())
                        positions.append(last * block - a + offset)
                low, high, x = np.asarray(l), np.asarray(h), np.asarray(positions)
            xs.extend(x / self.source.rate)
            lows.extend(np.asarray(low, dtype=np.float32) / 32768)
            highs.extend(np.asarray(high, dtype=np.float32) / 32768)
            offset += b - a
        return xs, lows, highs


def export_wav(document, folder, prefix, start, end, cancel, progress=lambda value: None):
    if end <= start:
        raise ValueError('保存する音声がありません。')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not prefix or any(c in prefix for c in '<>:"/\\|?*'):
        raise ValueError('ファイル名の接頭辞が不正です。')
    size = (end - start) * document.source.channels * 2
    if size + 44 >= 2**32:
        raise ValueError('WAVの4GB制限を超えています。範囲を分割してください。')
    if shutil.disk_usage(folder).free < size + 1024 * 1024:
        raise OSError('保存先のディスク空き容量が不足しています。')
    # Exclusive creation also prevents collisions between multiple app instances.
    for number in range(1, 10000000):
        target = folder / f'{prefix}_{number:04d}.wav'
        try:
            file = target.open('xb')
            break
        except FileExistsError:
            continue
    else:
        raise OSError('連番を確保できませんでした。')
    try:
        with file:
            with wave.open(file, 'wb') as wav:
                wav.setnchannels(document.source.channels)
                wav.setsampwidth(2)
                wav.setframerate(document.source.rate)
                done = 0
                for chunk in document.chunks(start, end):
                    check_cancel(cancel)
                    wav.writeframesraw(chunk.tobytes())
                    done += len(chunk)
                    progress(int(done * 100 / (end - start)))
            file.flush()
            os.fsync(file.fileno())
        check_cancel(cancel)
        return target
    except BaseException:
        target.unlink(missing_ok=True)
        raise
