"""One shared check that every tool uses before touching a file.

Rule: a tool may only touch files inside mock_world/. We take the path the
agent asked for, work out where it REALLY points (following '..' and
symlinks), and refuse if that real location is not inside mock_world/.
"""

from pathlib import Path

MOCK_WORLD = Path(__file__).resolve().parent.parent / "mock_world"


class PathNotAllowed(Exception):
    """Raised when a requested path would leave mock_world/."""


def resolve_in_world(requested: str, root: Path = MOCK_WORLD) -> Path:
    """Return the real, absolute path for `requested`, or raise PathNotAllowed.

    `requested` must be relative to `root` (e.g. "emails/001.txt").
    The file does not have to exist yet, so write tools can use this too.
    """
    root = root.resolve()

    if not isinstance(requested, str) or not requested.strip():
        raise PathNotAllowed("Path must be a non-empty string.")
    if "\x00" in requested:
        raise PathNotAllowed("Path contains an invalid character.")
    if Path(requested).is_absolute():
        raise PathNotAllowed("Absolute paths are not allowed.")

    # resolve() collapses '..' and follows symlinks to the real location.
    real = (root / requested).resolve()

    # is_relative_to compares whole folder names, so 'mock_world_evil' does
    # NOT count as being inside 'mock_world' (a plain startswith would).
    if not real.is_relative_to(root):
        raise PathNotAllowed("Path is outside mock_world/.")
    return real
