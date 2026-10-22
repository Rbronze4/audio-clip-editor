"""Bounded PCM delivery on the GUI thread (no Python audio-thread callbacks)."""
import queue
import subprocess
import tempfile
import threading

from core import find_ffmpeg


class TempoFeeder:
    """Stream edited PCM through atempo with bounded buffers, never block Qt on reads."""
    def __init__(self, document, start, end, speed):
        self.pending = b''
        self.queue = queue.Queue(maxsize=8)
        self.cancel = threading.Event()
        self.eof = False
        self.error = None
        self.log = tempfile.TemporaryFile()
        source = document.source
        try:
            self.process = subprocess.Popen([find_ffmpeg(), '-v', 'error', '-nostdin',
                '-f', 's16le', '-ar', str(source.rate), '-ac', str(source.channels),
                '-i', 'pipe:0', '-af', f'atempo={speed}', '-f', 's16le', 'pipe:1'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.log.close()
            raise

        def write():
            try:
                for chunk in document.chunks(start, end, chunk_frames=8192):
                    if self.cancel.is_set():
                        break
                    self.process.stdin.write(chunk.tobytes())
                self.process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass  # The reader reports the FFmpeg diagnostic.
            except Exception as error:
                self.error = str(error)
                self.process.kill()

        def read():
            try:
                while not self.cancel.is_set():
                    data = self.process.stdout.read(16384)
                    if not data:
                        break
                    while not self.cancel.is_set():
                        try:
                            self.queue.put(data, timeout=.05)
                            break
                        except queue.Full:
                            pass
                code = self.process.wait()
                if code and not self.cancel.is_set():
                    self.log.seek(0)
                    self.error = '倍速プレビューを生成できませんでした: ' + self.log.read().decode('utf-8', errors='replace')[-1000:]
            except Exception as error:
                if not self.cancel.is_set():
                    self.error = str(error)
            finally:
                self.eof = True

        self.writer = threading.Thread(target=write, daemon=True)
        self.reader = threading.Thread(target=read, daemon=True)
        self.writer.start()
        self.reader.start()

    @property
    def finished(self):
        return self.eof and self.queue.empty() and not self.pending

    def feed(self, output, capacity):
        if self.error:
            raise OSError(self.error)
        budget = max(0, int(capacity))
        while budget:
            if not self.pending:
                try:
                    self.pending = self.queue.get_nowait()
                except queue.Empty:
                    break
            written = output.write(self.pending[:budget])
            if written < 0:
                raise OSError('音声出力への書き込みに失敗しました。')
            if written == 0:
                break
            self.pending = self.pending[written:]
            budget -= written

    def close(self):
        self.cancel.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.writer.join()
        self.reader.join()
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        self.process.stdout.close()
        self.log.close()


class PCMFeeder:
    def __init__(self, document, start, end):
        self.chunks = iter(document.chunks(start, end, chunk_frames=8192))
        self.pending = b''
        self.remaining = (end - start) * document.source.channels * 2

    @property
    def finished(self):
        return self.remaining == 0

    def feed(self, output, capacity):
        """Respect output backpressure; retain any bytes not accepted yet."""
        budget = max(0, min(int(capacity), self.remaining))
        while budget:
            if not self.pending:
                self.pending = next(self.chunks).tobytes()
            data = self.pending[:budget]
            written = output.write(data)
            if written < 0:
                raise OSError('音声出力への書き込みに失敗しました。')
            if written == 0:
                break
            self.pending = self.pending[written:]
            self.remaining -= written
            budget -= written


def fatal_audio_error(error):
    # Older Qt releases report a drained buffer as UnderrunError, even at EOF.
    return error.name not in ('NoError', 'UnderrunError')
