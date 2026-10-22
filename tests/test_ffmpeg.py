import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import find_ffmpeg


class FFmpegDiscoveryTests(unittest.TestCase):
    def test_app_local_without_path(self):
        with patch('core.Path.is_file', return_value=True), patch('core.shutil.which') as which:
            found = Path(find_ffmpeg())
            self.assertEqual(found.parent.name, 'bin')
            self.assertTrue(found.is_absolute())
            which.assert_not_called()

    def test_path_without_app_local(self):
        with patch('core.Path.is_file', return_value=False), \
                patch('core.shutil.which', return_value='C:/tools/ffmpeg.exe'):
            self.assertEqual(find_ffmpeg(), 'C:/tools/ffmpeg.exe')

    def test_packaged_binary_without_path(self):
        package = Mock()
        package.get_ffmpeg_exe.return_value = 'C:/packages/ffmpeg.exe'
        with patch('core.Path.is_file', return_value=False), \
                patch('core.shutil.which', return_value=None), \
                patch.dict(sys.modules, {'imageio_ffmpeg': package}):
            self.assertEqual(find_ffmpeg(), 'C:/packages/ffmpeg.exe')

    def test_missing_dependency_explains_recovery(self):
        with patch('core.Path.is_file', return_value=False), \
                patch('core.shutil.which', return_value=None), \
                patch.dict(sys.modules, {'imageio_ffmpeg': None}):
            with self.assertRaisesRegex(RuntimeError, 'setup.bat'):
                find_ffmpeg()
