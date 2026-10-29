"""Asset integrity and private reporting checks; no model or NPU needed."""
import hashlib
import io
import json
import pathlib
import stat
import tempfile
import unittest
from unittest import mock

from fetch_tokenizer import fetch
from model_host import write_private_report


class AssetTests(unittest.TestCase):
    @staticmethod
    def manifest(content):
        return {'support_files':[{'path':'config.json','bytes':len(content),
                                  'sha256':hashlib.sha256(content).hexdigest(),
                                  'source_url':'https://huggingface.co/pinned/config.json'}]}

    def test_verified_download_is_private_and_reused(self):
        content=b'{"public":true}\n'
        with tempfile.TemporaryDirectory() as directory:
            destination=pathlib.Path(directory)/'tokenizer'
            with mock.patch('urllib.request.urlopen',return_value=io.BytesIO(content)) as request:
                result=fetch(destination,self.manifest(content))
                self.assertEqual(result[0]['status'],'downloaded-verified')
                file=destination/'config.json'
                self.assertEqual(file.read_bytes(),content)
                self.assertEqual(stat.S_IMODE(file.stat().st_mode),0o600)
                fetch(destination,self.manifest(content))
                self.assertEqual(request.call_count,1)

    def test_existing_corruption_is_preserved_and_refused(self):
        content=b'good'
        with tempfile.TemporaryDirectory() as directory:
            file=pathlib.Path(directory)/'config.json'
            file.write_bytes(b'evil')
            with mock.patch('urllib.request.urlopen') as request:
                with self.assertRaisesRegex(ValueError,'differs from pinned'):
                    fetch(directory,self.manifest(content))
                request.assert_not_called()
            self.assertEqual(file.read_bytes(),b'evil')

    def test_corrupt_download_is_not_published(self):
        content=b'good'
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('urllib.request.urlopen',return_value=io.BytesIO(b'evil')):
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    fetch(directory,self.manifest(content))
            self.assertEqual(list(pathlib.Path(directory).iterdir()),[])

    def test_download_cannot_exceed_pinned_size(self):
        content=b'good'
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('urllib.request.urlopen',return_value=io.BytesIO(b'good-extra')):
                with self.assertRaisesRegex(ValueError,'exceeds pinned'):
                    fetch(directory,self.manifest(content))
            self.assertEqual(list(pathlib.Path(directory).iterdir()),[])

    def test_symlink_asset_is_refused(self):
        content=b'good'
        with tempfile.TemporaryDirectory() as directory:
            outside=pathlib.Path(directory)/'outside'
            outside.write_bytes(content)
            (pathlib.Path(directory)/'config.json').symlink_to(outside)
            with self.assertRaises(ValueError):
                fetch(directory,self.manifest(content))


class ReportTests(unittest.TestCase):
    def test_report_is_private_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            report=pathlib.Path(directory)/'private'/'run.json'
            write_private_report(report,{'completed':True})
            self.assertEqual(stat.S_IMODE(report.stat().st_mode),0o600)
            self.assertEqual(json.loads(report.read_text()),{'completed':True})
            with self.assertRaises(FileExistsError):
                write_private_report(report,{'completed':False})
            self.assertEqual(json.loads(report.read_text()),{'completed':True})


if __name__=='__main__':
    unittest.main()
