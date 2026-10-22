"""Cancellation-aware recursive imports with deterministic order and deduplication."""
import os
from pathlib import Path
from core import check_cancel

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.aiff', '.aif', '.opus'}


def collect_audio(inputs, existing, cancel):
    seen = {os.path.normcase(str(Path(p).resolve())) for p in existing}
    result, failures = [], []

    def add(path):
        check_cancel(cancel)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        resolved = str(path.resolve())
        key = os.path.normcase(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)

    for name in inputs:
        check_cancel(cancel)
        path = Path(name)
        if path.is_file():
            add(path)
        elif path.is_dir():
            def onerror(error):
                failures.append(str(error))
            for directory, folders, files in os.walk(path, followlinks=False, onerror=onerror):
                check_cancel(cancel)
                folders[:] = sorted(d for d in folders
                                    if not (Path(directory) / d).is_symlink()
                                    and not (Path(directory) / d).is_junction())
                for file in sorted(files, key=str.casefold):
                    add(Path(directory) / file)
        else:
            failures.append(f'見つかりません: {path}')
    return result, failures
