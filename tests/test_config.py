import pytest

from agent.config import ConfigError, load_settings

ENV_NAMES = ["ANTHROPIC_API_KEY", "AGENT_MODEL", "JUDGE_MODEL", "MAX_STEPS"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_loads_values_from_env_file(tmp_path):
    env = write_env(tmp_path, "ANTHROPIC_API_KEY=sk-test-123\nMAX_STEPS=7\n")
    settings = load_settings(env)
    assert settings.max_steps == 7
    assert settings.agent_model == "claude-haiku-5-5"
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-123"


def test_api_key_never_appears_in_printed_output(tmp_path):
    env = write_env(tmp_path, "ANTHROPIC_API_KEY=sk-secret-abc\n")
    settings = load_settings(env)
    assert "sk-secret-abc" not in repr(settings)
    assert "sk-secret-abc" not in str(settings)


def test_missing_key_gives_clear_error(tmp_path):
    env = write_env(tmp_path, "AGENT_MODEL=claude-haiku-5-5\n")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is missing"):
        load_settings(env)


def test_invalid_value_error_does_not_leak_key(tmp_path):
    env = write_env(tmp_path, "ANTHROPIC_API_KEY=sk-secret-abc\nMAX_STEPS=banana\n")
    with pytest.raises(ConfigError) as err:
        load_settings(env)
    assert "sk-secret-abc" not in str(err.value)
    assert "max_steps" in str(err.value)
