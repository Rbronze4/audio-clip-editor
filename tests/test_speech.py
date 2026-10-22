import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech_runner import ensure_model


class ModelDownloadTests(unittest.TestCase):
    def test_verified_cache_is_used_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / 'mossformer2_se_48k.pt'
            target.write_bytes(b'verified fixture')
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with patch('speech_runner.MODEL_SHA256', digest), patch('urllib.request.urlopen') as network:
                self.assertEqual(ensure_model(folder, lambda *args: None), target)
                network.assert_not_called()

    def test_invalid_download_never_replaces_cached_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / 'mossformer2_se_48k.pt'
            target.write_bytes(b'existing')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'corrupt download')):
                with self.assertRaisesRegex(RuntimeError, '検証に失敗'):
                    ensure_model(folder, lambda *args: None)
            self.assertEqual(target.read_bytes(), b'existing')
            self.assertEqual(list(folder.glob('*.part')), [])
