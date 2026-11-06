import pytest

from tools.safe_path import MOCK_WORLD, PathNotAllowed, resolve_in_world


@pytest.fixture
def world(tmp_path):
    """A fake mock_world/ plus things outside it that must stay unreachable."""
    root = tmp_path / "mock_world"
    (root / "emails").mkdir(parents=True)
    (root / "emails" / "001.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("top secret")
    evil = tmp_path / "mock_world_evil"  # name merely STARTS with 'mock_world'
    evil.mkdir()
    (evil / "loot.txt").write_text("loot")
    return root


# --- allowed paths ---------------------------------------------------------

def test_normal_file_is_allowed(world):
    assert resolve_in_world("emails/001.txt", world) == world.resolve() / "emails" / "001.txt"


def test_file_that_does_not_exist_yet_is_allowed(world):
    # write_note needs to create new files.
    assert resolve_in_world("notes/new.txt", world).parent.name == "notes"


def test_dotdot_that_stays_inside_is_allowed(world):
    assert resolve_in_world("emails/../emails/001.txt", world) == world.resolve() / "emails" / "001.txt"


def test_symlink_pointing_inside_is_allowed(world):
    (world / "shortcut").symlink_to(world / "emails")
    assert resolve_in_world("shortcut/001.txt", world) == world.resolve() / "emails" / "001.txt"


# --- escape attempts -------------------------------------------------------

@pytest.mark.parametrize("bad", ["../secret.txt", "../../secret.txt", "..", "../"])
def test_dotdot_escape_is_rejected(world, bad):
    with pytest.raises(PathNotAllowed):
        resolve_in_world(bad, world)


@pytest.mark.parametrize("bad", ["/etc/passwd", "/", "/tmp"])
def test_absolute_path_is_rejected(world, bad):
    with pytest.raises(PathNotAllowed):
        resolve_in_world(bad, world)


def test_absolute_path_to_real_file_outside_is_rejected(world):
    with pytest.raises(PathNotAllowed):
        resolve_in_world(str(world.parent / "secret.txt"), world)


def test_absolute_path_even_inside_world_is_rejected(world):
    # Strict on purpose: the agent must always use relative paths.
    with pytest.raises(PathNotAllowed):
        resolve_in_world(str(world / "emails" / "001.txt"), world)


def test_starts_inside_but_climbs_out_is_rejected(world):
    with pytest.raises(PathNotAllowed):
        resolve_in_world("emails/../../secret.txt", world)


def test_symlink_to_file_outside_is_rejected(world):
    (world / "emails" / "link.txt").symlink_to(world.parent / "secret.txt")
    with pytest.raises(PathNotAllowed):
        resolve_in_world("emails/link.txt", world)


def test_symlink_to_folder_outside_is_rejected(world):
    (world / "notes").symlink_to(world.parent)
    with pytest.raises(PathNotAllowed):
        resolve_in_world("notes/secret.txt", world)


def test_symlink_chain_that_ends_outside_is_rejected(world):
    (world / "a").symlink_to(world / "b")
    (world / "b").symlink_to(world.parent / "secret.txt")
    with pytest.raises(PathNotAllowed):
        resolve_in_world("a", world)


# --- look-alike folder name ------------------------------------------------

def test_sibling_folder_with_matching_prefix_is_rejected(world):
    # tmp/mock_world_evil starts with the text 'mock_world' but is a different folder.
    with pytest.raises(PathNotAllowed):
        resolve_in_world("../mock_world_evil/loot.txt", world)


def test_symlink_to_lookalike_folder_is_rejected(world):
    (world / "evil").symlink_to(world.parent / "mock_world_evil")
    with pytest.raises(PathNotAllowed):
        resolve_in_world("evil/loot.txt", world)


# --- bad input -------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "emails/\x00evil"])
def test_empty_or_invalid_input_is_rejected(world, bad):
    with pytest.raises(PathNotAllowed):
        resolve_in_world(bad, world)


def test_default_root_is_the_real_mock_world():
    assert MOCK_WORLD.name == "mock_world"
    assert MOCK_WORLD.is_dir()
