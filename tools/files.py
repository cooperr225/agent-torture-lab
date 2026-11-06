"""The five tools the agent can use. Every one works only inside mock_world/.

Rules all five follow:
- Every file path goes through safe_path.resolve_in_world first.
- A problem (bad path, missing file...) returns a short "Error: ..." string
  instead of crashing, so the agent loop can hand it back to the model.
- send_message never sends anything: it only appends to one fixed log file.

Each tool takes an optional `world` argument. The agent never sets it; tests
use it to point the tool at a temporary fake mock_world/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from tools.safe_path import MOCK_WORLD, PathNotAllowed, resolve_in_world

MAX_READ_CHARS = 20_000  # long files are cut off so they can't flood the model
MAX_WRITE_CHARS = 10_000
MAX_SEARCH_RESULTS = 3
MAX_SEARCH_CHARS = 1_500  # per result

OUTBOX_LOG = "outbox.log"  # the ONE file send_message may write to


def _read(subfolder: str, name: str, world: Path) -> str:
    """Read a text file from mock_world/<subfolder>/, or return an error string."""
    try:
        path = resolve_in_world(name, root=world / subfolder)
        text = path.read_text(encoding="utf-8")
    except PathNotAllowed as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: '{name}' not found."
    except (IsADirectoryError, PermissionError):
        return f"Error: '{name}' is not a readable file."
    except (UnicodeDecodeError, OSError):
        return f"Error: could not read '{name}'."
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n[...truncated]"
    return text


def read_email(name: str, world: Path = MOCK_WORLD) -> str:
    """Read one email by file name, e.g. '001.txt', from mock_world/emails/."""
    return _read("emails", name, world)


def read_note(name: str, world: Path = MOCK_WORLD) -> str:
    """Read one note by file name, e.g. 'shopping-list.txt', from mock_world/notes/."""
    return _read("notes", name, world)


def write_note(name: str, content: str, world: Path = MOCK_WORLD) -> str:
    """Create or overwrite a note in mock_world/notes/."""
    if not isinstance(content, str):
        return "Error: content must be text."
    if len(content) > MAX_WRITE_CHARS:
        return f"Error: note is too long (limit {MAX_WRITE_CHARS} characters)."
    try:
        path = resolve_in_world(name, root=world / "notes")
        if path.is_dir():
            return f"Error: '{name}' is a folder, not a file."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except PathNotAllowed as exc:
        return f"Error: {exc}"
    except OSError:
        return f"Error: could not write '{name}'."
    return f"Saved note '{name}' ({len(content)} characters)."


def search_web(query: str, world: Path = MOCK_WORLD) -> str:
    """Search the local mock web pages in mock_world/web/ for the query words.

    This never touches the internet. It returns the text of the best matches.
    """
    if not isinstance(query, str) or not query.strip():
        return "Error: query must be non-empty text."
    words = query.lower().split()
    web_dir = world / "web"
    try:
        names = sorted(p.name for p in web_dir.iterdir() if p.is_file() or p.is_symlink())
    except OSError:
        return "Error: could not search the web pages."

    scored = []
    for name in names:
        try:
            # Same safety check as every other file read (blocks symlink escapes).
            text = resolve_in_world(name, root=web_dir).read_text(encoding="utf-8")
        except (PathNotAllowed, OSError, UnicodeDecodeError):
            continue
        haystack = (name + " " + text).lower()
        score = sum(haystack.count(w) for w in words)
        if score:
            scored.append((score, name, text))

    if not scored:
        return f"No results for '{query}'."
    scored.sort(key=lambda item: (-item[0], item[1]))
    results = [
        f"[{name}]\n{text[:MAX_SEARCH_CHARS]}"
        for _, name, text in scored[:MAX_SEARCH_RESULTS]
    ]
    return "\n\n".join(results)


def send_message(to: str, body: str, world: Path = MOCK_WORLD) -> str:
    """Pretend to send a message. It only APPENDS one line to mock_world/outbox.log.

    Nothing is ever sent. The log file name is fixed; the agent cannot choose it.
    """
    if not isinstance(to, str) or not to.strip():
        return "Error: 'to' must be non-empty text."
    if not isinstance(body, str) or not body.strip():
        return "Error: 'body' must be non-empty text."
    if len(body) > MAX_WRITE_CHARS:
        return f"Error: message is too long (limit {MAX_WRITE_CHARS} characters)."
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "to": to.strip(),
        "body": body,
    }
    try:
        log_path = resolve_in_world(OUTBOX_LOG, root=world)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")  # one JSON line, even if body has newlines
    except (PathNotAllowed, OSError):
        return "Error: could not log the message."
    return f"Message to {entry['to']} logged (not actually sent)."
