"""Exclusive exports: neither automatic nor manual exports overwrite files."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core import check_cancel, export_wav, find_ffmpeg
from preferences import valid_name


def export_audio(document, start, end, preferences, quality, acting, source,
                 manual_path, cancel, progress=lambda value: None):
    preferences.validate()
    number = preferences.number
    if manual_path is None:
        folder = Path(preferences.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        while number < 2147483647:
            check_cancel(cancel)
            target = folder / preferences.filename(quality, acting, source, number)
            try:
                output = target.open('xb')
                break
            except FileExistsError:
                number += 1
        else:
            raise ValueError('連番の上限に達しました。設定から連番を変更してください。')
    else:
        target = Path(manual_path)
        valid_name(target.name)
        if target.suffix.lower() not in ('.wav', '.mp3'):
            raise ValueError('保存形式は WAV または MP3 を選んでください。')
        output = target.open('xb')
    try:
        with output, tempfile.TemporaryDirectory(prefix='clip-export-', dir=target.parent) as temp:
            # Encode to a temporary file first; remove our reserved target on any failure.
            wav = export_wav(document, temp, 'clip', start, end, cancel,
                             lambda value: progress(value // 2))
            encoded = wav
            if target.suffix.lower() == '.mp3':
                encoded = Path(temp) / 'clip.mp3'
                with (Path(temp) / 'ffmpeg.log').open('w+b') as log:
                    process = subprocess.Popen([find_ffmpeg(), '-v', 'error', '-nostdin', '-y',
                        '-i', str(wav), '-c:a', 'libmp3lame', '-b:a', '320k', str(encoded)],
                        stdout=subprocess.DEVNULL, stderr=log,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    try:
                        while process.poll() is None:
                            cancel.wait(.1)
                            check_cancel(cancel)
                        if process.returncode:
                            log.seek(0)
                            raise RuntimeError('MP3保存に失敗しました: ' + log.read().decode('utf-8', errors='replace')[-2000:])
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
            if shutil.disk_usage(target.parent).free < encoded.stat().st_size + 1048576:
                raise OSError('保存先のディスク空き容量が不足しています。')
            with encoded.open('rb') as file:
                while chunk := file.read(1024 * 1024):
                    check_cancel(cancel)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
            check_cancel(cancel)
        progress(100)
        return target, number + 1 if manual_path is None else preferences.number
    except BaseException:
        target.unlink(missing_ok=True)
        raise
