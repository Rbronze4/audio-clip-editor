"""Cancelable subprocess separation; aligned Original/Separated PCM snapshots."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from core import Source, check_cancel, find_ffmpeg, export_wav


def run_process(command, cancel, status_path=None, report=lambda value, message: None):
    check_cancel(cancel)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        last = None
        try:
            while process.poll() is None:
                check_cancel(cancel)
                if status_path and status_path.exists():
                    try:
                        state = json.loads(status_path.read_text(encoding='utf-8'))
                        if state != last:
                            last = state
                            report(state['progress'], state['message'])
                    except (OSError, ValueError):
                        pass
                time.sleep(.1)
            check_cancel(cancel)
            if process.returncode:
                if status_path and status_path.exists():
                    state = json.loads(status_path.read_text(encoding='utf-8'))
                    if state.get('error'):
                        raise RuntimeError(state['message'])
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 4000))
                detail = log.read().decode('utf-8', errors='replace')
                raise RuntimeError('BGM除去の処理に失敗しました。\n' + detail)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def engine_python():
    directory = Path(__file__).resolve().parent / '.venv-bgm'
    python = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    return str(python) if python.is_file() else sys.executable


def convert_for_model(input_path, output_path, channels, cancel, rate=44100):
    # FFmpeg's default mono upmix attenuates by sqrt(2). Explicit duplication
    # followed by an explicit stereo average gives unity gain for mono input.
    channel_filter = ['-af', 'pan=stereo|c0=c0|c1=c0'] if channels == 1 else []
    run_process([find_ffmpeg(), '-nostdin', '-v', 'error', '-y', '-i', str(input_path),
                 *channel_filter, '-ar', str(rate), '-ac', '2', '-c:a', 'pcm_s16le',
                 str(output_path)], cancel)


def restore_model_output(input_path, output_path, rate, channels, frames, cancel):
    filters = [f'aresample={rate}']
    if channels == 1:
        filters.append('pan=mono|c0=0.5*c0+0.5*c1')
    filters.extend(['apad', f'atrim=end_sample={frames}'])
    run_process([find_ffmpeg(), '-nostdin', '-v', 'error', '-y', '-i', str(input_path),
                 '-af', ','.join(filters), '-ac', str(channels), '-c:a', 'pcm_s16le',
                 '-f', 's16le', str(output_path)], cancel)


def make_snapshot(document, cache_dir, cancel, replacement=None):
    fd, filename = tempfile.mkstemp(suffix='.pcm', dir=cache_dir)
    source = None
    try:
        with os.fdopen(fd, 'wb') as stream:
            if replacement is None:
                ranges = [(0, document.frames, None)]
            else:
                a, b, path = replacement
                ranges = [(0, a, None), (a, b, path), (b, document.frames, None)]
            for a, b, path in ranges:
                if path is None:
                    for chunk in document.chunks(a, b):
                        check_cancel(cancel)
                        stream.write(chunk.tobytes())
                else:
                    with open(path, 'rb') as data:
                        while block := data.read(1024 * 1024):
                            check_cancel(cancel)
                            stream.write(block)
        check_cancel(cancel)
        source = Source(document.source.path, filename, document.source.rate,
                        document.source.channels, document.frames, 0, cancel)
        return source
    except BaseException:
        if source:
            source.close()
        Path(filename).unlink(missing_ok=True)
        raise


def separate(document, a, b, cache_dir, device, cancel, report=lambda value, message: None, engine='demucs'):
    if engine not in ('demucs', 'speech'):
        raise ValueError('未対応の分離方式です。')
    if not 0 <= a < b <= document.frames:
        raise ValueError('BGM除去する範囲を選択してください。')
    if document.source.channels not in (1, 2):
        raise ValueError('BGM除去はモノラル／ステレオの音声に対応しています。')
    # Two full aligned versions plus inference input/output intermediates.
    required = document.frames * document.source.channels * 4 + (b-a) * 32 + 50_000_000
    if shutil.disk_usage(cache_dir).free < required:
        raise OSError('BGM除去の一時ファイル用ディスク容量が不足しています。')
    original_view = document.snapshot('original')
    base = document.snapshot('separated' if document.state.separated is not None else 'original')
    originals = vocals = None
    try:
        with tempfile.TemporaryDirectory(prefix='separation_', dir=cache_dir) as temp:
            temp = Path(temp)
            report(-1, '選択範囲を分離用に準備中…')
            input_wav = export_wav(original_view, temp, 'input', a, b, cancel)
            model_input, model_output = temp / 'model.wav', temp / 'vocals.wav'
            convert_for_model(input_wav, model_input, document.source.channels, cancel,
                              rate=48000 if engine == 'speech' else 44100)
            app_dir = Path(__file__).resolve().parent
            status_path = temp / 'status.json'
            runner = 'speech_runner.py' if engine == 'speech' else 'demucs_runner.py'
            run_process([engine_python(), str(app_dir / runner),
                         '--input', str(model_input), '--output', str(model_output),
                         '--status', str(status_path), '--models', str(app_dir / 'models'),
                         '--device', device], cancel, status_path, report)
            report(-1, '原音の形式へ変換・A/B比較を準備中…')
            normalized = temp / 'vocals.pcm'
            # Restore source rate/channels and force exactly the selected frame
            # count, including resampler rounding. No peak normalization.
            restore_model_output(model_output, normalized, document.source.rate,
                                 document.source.channels, b-a, cancel)
            if normalized.stat().st_size != (b-a) * document.source.channels * 2:
                raise RuntimeError('分離音声の長さが一致しません。原音は変更していません。')
            originals = make_snapshot(original_view, cache_dir, cancel)
            vocals = make_snapshot(base, cache_dir, cancel, (a, b, normalized))
            check_cancel(cancel)
            report(100, 'BGM除去完了 — A/Bを切り替えて比較できます')
            return originals, vocals
    except BaseException:
        if originals:
            originals.close()
        if vocals:
            vocals.close()
        raise
