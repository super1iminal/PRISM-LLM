"""Log files: one logger per file, and closing a logger releases its file (Windows locks open files)."""
import shutil

from logging_utils import close_logger, setup_logger


def test_setting_up_a_file_again_closes_the_old_handler(tmp_path):
    first = setup_logger("worker_0", run_dir=str(tmp_path), include_timestamp=False)
    old_handler = first.handlers[0]
    again = setup_logger("worker_0", run_dir=str(tmp_path), include_timestamp=False)
    assert again is first and len(again.handlers) == 1
    assert old_handler.stream is None   # closed, not just dropped
    close_logger(again)


def test_same_name_in_another_run_dir_is_another_logger(tmp_path):
    a = setup_logger("main", run_dir=str(tmp_path / "a"), include_timestamp=False)
    b = setup_logger("main", run_dir=str(tmp_path / "b"), include_timestamp=False)
    assert a is not b
    a.info("to a")
    b.info("to b")
    close_logger(a)
    close_logger(b)
    text_a = (tmp_path / "a" / "main.log").read_text(encoding="utf-8")
    assert "to a" in text_a and "to b" not in text_a


def test_closed_logger_releases_its_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    logger = setup_logger("main", run_dir=str(run_dir), include_timestamp=False)
    logger.info("hello")
    close_logger(logger)
    assert not logger.handlers
    shutil.rmtree(run_dir)   # raises on Windows while main.log is still open
    assert not run_dir.exists()
