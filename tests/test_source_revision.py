"""Portable source-version regressions; standard library only, no H3 evidence."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from wushu_bridge.source_revision import REPOSITORY, revision


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = "a" * 40

    def app(self, commit):
        (self.root/'app_version.json').write_text(json.dumps({'repository':REPOSITORY,'source_commit':commit}),encoding='utf-8')

    def test_release_metadata_without_git_or_receipt(self):
        self.app(self.commit)
        with patch('wushu_bridge.source_revision.subprocess.run',side_effect=AssertionError('Git must not run')):
            self.assertEqual(revision(self.root),self.commit)

    def test_updated_metadata_overrides_stale_receipt(self):
        self.app(self.commit)
        (self.root/'SOURCE_REVISION.txt').write_text('b'*40+'\n')
        self.assertEqual(revision(self.root),self.commit)

    def test_old_human_readable_packaging_bug_recovers(self):
        self.app(self.commit)
        (self.root/'SOURCE_REVISION.txt').write_text('Trainer release 0.2.2\nPublic source commit '+self.commit+'\nExact distributed contents are bound by PORTABLE_MANIFEST.json.\n')
        self.assertEqual(revision(self.root),self.commit)

    def test_invalid_release_commit_still_rejected(self):
        self.app('not-a-commit')
        with self.assertRaisesRegex(ValueError,'Invalid app_version'):
            revision(self.root)

    def test_original_receipts_remain_strict(self):
        p=self.root/'SOURCE_REVISION.txt'
        for value in [self.commit,'unversioned']:
            p.write_text(value+'\n')
            self.assertEqual(revision(self.root),value)
        p.write_text('garbage')
        with self.assertRaisesRegex(ValueError,'Invalid SOURCE_REVISION'):
            revision(self.root)

    def test_without_git_reports_unversioned(self):
        with patch('wushu_bridge.source_revision.subprocess.run',side_effect=FileNotFoundError):
            self.assertEqual(revision(self.root),'unversioned')


if __name__=='__main__':unittest.main()
