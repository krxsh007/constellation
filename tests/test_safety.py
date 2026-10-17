"""Tests for the safety net around writes: backups/undo, the strip command and
the per-vault run lock.

Run with:  python3 -m unittest discover tests
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constellation.core import backup, config, lock
from constellation.core import log as logmod
from constellation.nodes import io


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


class TmpVault(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p


class TestBackups(TmpVault):
    def test_create_list_restore(self):
        a = self.write("A.md", "original A")
        self.write("Sub/B.md", "original B")
        bdir = backup.create_backup(self.vault, [a, "Sub/B.md"])
        self.assertTrue(str(bdir).startswith(str(self.vault / ".constellation_backups")))
        self.assertEqual((bdir / "Sub" / "B.md").read_text(), "original B")

        a.write_text("changed A")
        (self.vault / "Sub/B.md").write_text("changed B")

        listed = backup.list_backups(self.vault)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], bdir.name)
        self.assertEqual(listed[0]["file_count"], 2)
        self.assertIn("T", listed[0]["created"])

        self.assertEqual(backup.restore_backup(self.vault, bdir.name), 2)
        self.assertEqual(a.read_text(), "original A")
        self.assertEqual((self.vault / "Sub/B.md").read_text(), "original B")

    def test_newest_first_and_pruned(self):
        a = self.write("A.md", "x")
        with mock.patch.object(config, "MAX_BACKUPS", 3):
            made = [backup.create_backup(self.vault, [a]).name for _ in range(5)]
        names = [b["name"] for b in backup.list_backups(self.vault)]
        self.assertEqual(names, list(reversed(made))[:3])

    def test_restore_rejects_path_traversal_and_unknown(self):
        with self.assertRaises(ValueError):
            backup.restore_backup(self.vault, "../etc")
        with self.assertRaises(FileNotFoundError):
            backup.restore_backup(self.vault, "20000101-000000")

    def test_restore_forgets_cache_hashes(self):
        a = self.write("A.md", "x")
        self.write("B.md", "y")
        (self.vault / io.CACHE_FILENAME).write_text(json.dumps(
            {"files": {"A.md": {"hash": "1"}, "B.md": {"hash": "2"}}}))
        name = backup.create_backup(self.vault, [a]).name
        backup.restore_backup(self.vault, name)
        files = json.loads((self.vault / io.CACHE_FILENAME).read_text())["files"]
        self.assertEqual(set(files), {"B.md"})

    def test_backups_are_not_read_as_notes(self):
        a = self.write("A.md", "x")
        backup.create_backup(self.vault, [a])
        self.assertEqual([p.name for p in io.iter_notes(self.vault, [])], ["A.md"])

    def test_link_writer_backs_up_before_writing(self):
        self.write("A.md", "# A")
        self.write("B.md", "# B")
        state = io.vault_reader({"dir": str(self.vault)})
        state["links"] = [{"from_note": "A.md", "to_note": "B.md", "score": 1.0}]
        out = io.link_writer(state)
        self.assertTrue(out["backup_dir"])
        self.assertEqual((Path(out["backup_dir"]) / "A.md").read_text(), "# A")
        self.assertIn("[[B]]", (self.vault / "A.md").read_text())


class TestStripVault(TmpVault):
    def test_strip_removes_generated_sections_and_state(self):
        self.write("A.md", "# A\n\nBody.\n\n## Tags\n#x\n\n## Related Links\n- [[B]]\n")
        self.write("B.md", "# B\n")
        (self.vault / io.CACHE_FILENAME).write_text("{}")
        (self.vault / ".linker_faiss_index").mkdir()

        preview = io.strip_vault(self.vault, dry_run=True)
        self.assertEqual(preview, {"notes_cleaned": 1, "backup_dir": None})
        self.assertIn("## Tags", (self.vault / "A.md").read_text())

        result = io.strip_vault(self.vault)
        self.assertEqual(result["notes_cleaned"], 1)
        self.assertEqual((self.vault / "A.md").read_text(), "# A\n\nBody.\n")
        self.assertIn("## Tags", (Path(result["backup_dir"]) / "A.md").read_text())
        self.assertFalse((self.vault / io.CACHE_FILENAME).exists())
        self.assertFalse((self.vault / ".linker_faiss_index").exists())
        self.assertFalse((self.vault / lock.LOCK_FILENAME).exists())

    def test_strip_refuses_when_locked(self):
        with lock.vault_lock(self.vault):
            with self.assertRaises(lock.VaultLockedError):
                io.strip_vault(self.vault)


class TestVaultLock(TmpVault):
    def test_lock_excludes_second_run_and_is_released(self):
        with lock.vault_lock(self.vault) as path:
            data = json.loads(path.read_text())
            self.assertEqual(data["pid"], os.getpid())
            with self.assertRaises(lock.VaultLockedError):
                with lock.vault_lock(self.vault):
                    pass
        self.assertFalse(lock.lock_path(self.vault).exists())

    def test_dead_pid_lock_is_stale(self):
        lock.lock_path(self.vault).write_text(json.dumps({"pid": 999999999, "timestamp": time.time()}))
        with lock.vault_lock(self.vault):
            pass

    def test_old_lock_is_stale(self):
        old = time.time() - lock.STALE_AFTER_SECONDS - 10
        lock.lock_path(self.vault).write_text(json.dumps({"pid": os.getpid(), "timestamp": old}))
        with lock.vault_lock(self.vault):
            pass

    def test_garbage_lock_is_stale(self):
        lock.lock_path(self.vault).write_text("not json")
        with lock.vault_lock(self.vault):
            pass

    def test_pipeline_takes_the_lock(self):
        import main
        with mock.patch.object(config, "missing_keys", return_value=[]):
            with lock.vault_lock(self.vault):
                with self.assertRaises(lock.VaultLockedError):
                    asyncio.run(main.arun_pipeline(str(self.vault)))


if __name__ == "__main__":
    unittest.main()
