import json

import pytest

from tools.files import (
    MAX_READ_CHARS,
    MAX_WRITE_CHARS,
    read_email,
    read_note,
    search_web,
    send_message,
    write_note,
)


@pytest.fixture
def world(tmp_path):
    """A fake mock_world/ with a little data, plus a secret file outside it."""
    root = tmp_path / "mock_world"
    for sub in ("emails", "notes", "web", "calendar"):
        (root / sub).mkdir(parents=True)
    (root / "emails" / "001.txt").write_text("Subject: Hello\n\nHi there.")
    (root / "notes" / "todo.txt").write_text("buy milk")
    (root / "web" / "coffee.txt").write_text("How to brew good coffee at home.")
    (root / "web" / "library.txt").write_text("Library opening hours: 9am to 7pm.")
    (tmp_path / "secret.txt").write_text("TOP-SECRET")
    return root


def files_in(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# --- the real sample data is usable ---------------------------------------

def test_real_sample_data_can_be_read():
    assert "Subject:" in read_email("001.txt")
    assert read_note("shopping-list.txt").startswith("Shopping list")
    assert "[python-virtual-environments.txt]" in search_web("virtual environment")


# --- read_email ------------------------------------------------------------

def test_read_email_returns_contents(world):
    assert "Hi there." in read_email("001.txt", world=world)


@pytest.mark.parametrize("bad", ["../secret.txt", "../notes/todo.txt", "/etc/passwd", ""])
def test_read_email_bad_path_returns_error(world, bad):
    result = read_email(bad, world=world)
    assert result.startswith("Error:")
    assert "TOP-SECRET" not in result and "buy milk" not in result


def test_read_email_symlink_escape_returns_error(world):
    (world / "emails" / "link.txt").symlink_to(world.parent / "secret.txt")
    result = read_email("link.txt", world=world)
    assert result.startswith("Error:") and "TOP-SECRET" not in result


def test_read_email_missing_file_returns_error(world):
    assert read_email("999.txt", world=world) == "Error: '999.txt' not found."


def test_read_email_folder_returns_error(world):
    (world / "emails" / "sub").mkdir()
    assert read_email("sub", world=world).startswith("Error:")


def test_read_email_truncates_huge_file(world):
    (world / "emails" / "big.txt").write_text("x" * (MAX_READ_CHARS + 500))
    result = read_email("big.txt", world=world)
    assert len(result) < MAX_READ_CHARS + 50 and result.endswith("[...truncated]")


# --- read_note -------------------------------------------------------------

def test_read_note_returns_contents(world):
    assert read_note("todo.txt", world=world) == "buy milk"


@pytest.mark.parametrize("bad", ["../secret.txt", "../emails/001.txt", "/etc/passwd"])
def test_read_note_bad_path_returns_error(world, bad):
    result = read_note(bad, world=world)
    assert result.startswith("Error:") and "TOP-SECRET" not in result


def test_read_note_missing_file_returns_error(world):
    assert read_note("nope.txt", world=world).startswith("Error:")


# --- write_note ------------------------------------------------------------

def test_write_note_creates_and_overwrites(world):
    assert write_note("new.txt", "first", world=world).startswith("Saved")
    assert read_note("new.txt", world=world) == "first"
    write_note("new.txt", "second", world=world)
    assert read_note("new.txt", world=world) == "second"


@pytest.mark.parametrize("bad", ["../escaped.txt", "/tmp/escaped.txt", "../mock_world_evil/x.txt"])
def test_write_note_bad_path_returns_error_and_writes_nothing(world, bad):
    before = files_in(world.parent)
    assert write_note(bad, "pwned", world=world).startswith("Error:")
    assert files_in(world.parent) == before


def test_write_note_symlink_escape_returns_error(world):
    (world / "notes" / "link.txt").symlink_to(world.parent / "secret.txt")
    assert write_note("link.txt", "overwritten", world=world).startswith("Error:")
    assert (world.parent / "secret.txt").read_text() == "TOP-SECRET"


def test_write_note_to_a_folder_returns_error(world):
    (world / "notes" / "sub").mkdir()
    assert write_note("sub", "x", world=world).startswith("Error:")


def test_write_note_too_long_returns_error(world):
    result = write_note("big.txt", "x" * (MAX_WRITE_CHARS + 1), world=world)
    assert result.startswith("Error:")
    assert not (world / "notes" / "big.txt").exists()


def test_write_note_non_text_content_returns_error(world):
    assert write_note("x.txt", 123, world=world).startswith("Error:")  # type: ignore[arg-type]


# --- search_web ------------------------------------------------------------

def test_search_web_finds_matching_page(world):
    result = search_web("coffee", world=world)
    assert "[coffee.txt]" in result and "library.txt" not in result


def test_search_web_is_case_insensitive_and_ranks(world):
    result = search_web("LIBRARY hours", world=world)
    assert result.startswith("[library.txt]")


def test_search_web_no_match_returns_message_not_crash(world):
    assert search_web("zebra", world=world) == "No results for 'zebra'."


@pytest.mark.parametrize("bad", ["", "   "])
def test_search_web_empty_query_returns_error(world, bad):
    assert search_web(bad, world=world).startswith("Error:")


def test_search_web_missing_web_folder_returns_error(world):
    (world / "web" / "coffee.txt").unlink()
    (world / "web" / "library.txt").unlink()
    (world / "web").rmdir()
    assert search_web("coffee", world=world).startswith("Error:")


def test_search_web_does_not_follow_symlink_outside(world):
    (world / "web" / "leak.txt").symlink_to(world.parent / "secret.txt")
    assert "TOP-SECRET" not in search_web("secret", world=world)


# --- send_message ----------------------------------------------------------

def test_send_message_appends_to_log_only(world):
    before = files_in(world)
    assert "not actually sent" in send_message("bob@example.com", "hi", world=world)
    send_message("amy@example.com", "second", world=world)
    assert files_in(world) == sorted(before + ["outbox.log"])  # nothing else created
    lines = (world / "outbox.log").read_text().splitlines()
    assert [json.loads(line)["to"] for line in lines] == ["bob@example.com", "amy@example.com"]
    assert json.loads(lines[0])["body"] == "hi"


def test_send_message_multiline_body_stays_one_log_line(world):
    send_message("bob@example.com", "line1\nline2\nFAKE-ENTRY", world=world)
    assert len((world / "outbox.log").read_text().splitlines()) == 1


def test_send_message_cannot_be_pointed_at_another_file(world):
    # There is no path argument; a path-looking 'to' is just text in the log.
    send_message("../secret.txt", "overwrite?", world=world)
    assert (world.parent / "secret.txt").read_text() == "TOP-SECRET"
    assert "../secret.txt" in (world / "outbox.log").read_text()


def test_send_message_symlinked_log_is_refused(world):
    (world / "outbox.log").symlink_to(world.parent / "secret.txt")
    result = send_message("bob@example.com", "hi", world=world)
    assert result.startswith("Error:")
    assert (world.parent / "secret.txt").read_text() == "TOP-SECRET"


def test_send_message_unwritable_log_returns_error(world):
    (world / "outbox.log").mkdir()  # a folder where the file should be
    assert send_message("bob@example.com", "hi", world=world).startswith("Error:")


@pytest.mark.parametrize("to, body", [("", "hi"), ("bob@example.com", ""), ("  ", "hi"), ("a", "   ")])
def test_send_message_empty_fields_return_error(world, to, body):
    assert send_message(to, body, world=world).startswith("Error:")
    assert not (world / "outbox.log").exists()


def test_send_message_too_long_returns_error(world):
    assert send_message("bob@example.com", "x" * (MAX_WRITE_CHARS + 1), world=world).startswith("Error:")
