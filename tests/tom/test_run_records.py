import os
import stat

import pytest


def test_new_run_ancestors_are_durable_before_no_replace_publish(tmp_path, monkeypatch):
    from werewolf.artifact_io import canonical
    from werewolf.tom.run_records import publish_run_bytes

    target = tmp_path / "runs" / "experiment" / "implicit" / "0" / "record.json"
    calls = []
    sync = canonical._fsync_directory
    link = os.link
    fsync = os.fsync

    def directory_sync(path):
        sync(path)
        calls.append(("directory", path))

    def file_sync(fd):
        fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            calls.append(("file", None))

    def no_replace(source, destination):
        assert not destination.exists()
        assert calls[-1][0] == "file"
        assert all(("directory", parent) in calls for parent in target.parents)
        calls.append(("publish", destination))
        link(source, destination)

    monkeypatch.setattr(canonical, "_fsync_directory", directory_sync)
    monkeypatch.setattr("werewolf.tom.run_records._fsync_directory", directory_sync)
    monkeypatch.setattr(os, "fsync", file_sync)
    monkeypatch.setattr(os, "link", no_replace)
    publish_run_bytes(target, b"complete")
    assert calls[-1] == ("directory", target.parent)
    calls.clear()
    publish_run_bytes(target, b"complete")
    assert ("file", None) in calls
    assert not any(kind == "publish" for kind, _ in calls)
    assert calls[-1] == ("directory", target.parent)
    with pytest.raises(ValueError, match="immutable"):
        publish_run_bytes(target, b"replacement")


def test_run_record_rejects_symlink_ancestor(tmp_path):
    from werewolf.tom.run_records import publish_run_bytes
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        publish_run_bytes(tmp_path / "alias" / "record", b"complete")
