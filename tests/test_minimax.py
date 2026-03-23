"""Unit tests for MiniMax language model integration."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from graph_of_thoughts.language_models.minimax import MiniMax


def _make_config(overrides=None):
    """Create a temporary config file with MiniMax settings."""
    config = {
        "minimax": {
            "model_id": "MiniMax-M2.7",
            "prompt_token_cost": 0.001,
            "response_token_cost": 0.005,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stop": None,
            "base_url": "https://api.minimax.io/v1",
            "api_key": "test-key-123",
        }
    }
    if overrides:
        config["minimax"].update(overrides)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(config, tmp)
    tmp.close()
    return tmp.name


def _mock_chat_completion(content="Hello!", prompt_tokens=10, completion_tokens=5):
    """Create a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestMiniMaxInit:
    """Tests for MiniMax initialization."""

    def test_init_with_config(self):
        """Test initialization with a config file."""
        config_path = _make_config()
        try:
            env = os.environ.copy()
            env.pop("MINIMAX_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                lm = MiniMax(config_path=config_path, model_name="minimax")
                assert lm.model_id == "MiniMax-M2.7"
                assert lm.temperature == 0.7
                assert lm.max_tokens == 4096
                assert lm.api_key == "test-key-123"
                assert lm.base_url == "https://api.minimax.io/v1"
        finally:
            os.unlink(config_path)

    def test_init_with_env_api_key(self):
        """Test that MINIMAX_API_KEY env var takes precedence."""
        config_path = _make_config({"api_key": "config-key"})
        try:
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
                lm = MiniMax(config_path=config_path, model_name="minimax")
                assert lm.api_key == "env-key"
        finally:
            os.unlink(config_path)

    def test_init_raises_without_api_key(self):
        """Test that missing API key raises ValueError."""
        config_path = _make_config({"api_key": ""})
        try:
            with patch.dict(os.environ, {}, clear=True):
                env = os.environ.copy()
                env.pop("MINIMAX_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                        MiniMax(config_path=config_path, model_name="minimax")
        finally:
            os.unlink(config_path)

    def test_temperature_clamping_high(self):
        """Test that temperature above 1.0 is clamped to 1.0."""
        config_path = _make_config({"temperature": 1.5})
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            assert lm.temperature == 1.0
        finally:
            os.unlink(config_path)

    def test_temperature_clamping_low(self):
        """Test that temperature below 0.0 is clamped to 0.0."""
        config_path = _make_config({"temperature": -0.5})
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            assert lm.temperature == 0.0
        finally:
            os.unlink(config_path)

    def test_temperature_zero(self):
        """Test that temperature=0 is accepted."""
        config_path = _make_config({"temperature": 0.0})
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            assert lm.temperature == 0.0
        finally:
            os.unlink(config_path)

    def test_default_base_url(self):
        """Test that default base URL is used when not specified."""
        config = {
            "minimax": {
                "model_id": "MiniMax-M2.7",
                "prompt_token_cost": 0.001,
                "response_token_cost": 0.005,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stop": None,
                "api_key": "test-key",
            }
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(config, tmp)
        tmp.close()
        try:
            lm = MiniMax(config_path=tmp.name, model_name="minimax")
            assert lm.base_url == "https://api.minimax.io/v1"
        finally:
            os.unlink(tmp.name)

    def test_custom_base_url(self):
        """Test that custom base URL is used when specified."""
        config_path = _make_config(
            {"base_url": "https://custom.api.example.com/v1"}
        )
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            assert lm.base_url == "https://custom.api.example.com/v1"
        finally:
            os.unlink(config_path)


class TestMiniMaxQuery:
    """Tests for MiniMax query method."""

    def test_single_response(self):
        """Test querying with a single response."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            mock_response = _mock_chat_completion("sorted list")
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = mock_response

            result = lm.query("Sort this list", num_responses=1)
            assert result == mock_response
            lm.client.chat.completions.create.assert_called_once_with(
                model="MiniMax-M2.7",
                messages=[{"role": "user", "content": "Sort this list"}],
                temperature=0.7,
                max_tokens=4096,
                stop=None,
            )
        finally:
            os.unlink(config_path)

    def test_multiple_responses(self):
        """Test querying with multiple responses makes individual calls."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            mock_response = _mock_chat_completion("response")
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = mock_response

            result = lm.query("Sort this list", num_responses=3)
            assert isinstance(result, list)
            assert len(result) == 3
            assert lm.client.chat.completions.create.call_count == 3
        finally:
            os.unlink(config_path)

    def test_caching(self):
        """Test that caching works for repeated queries."""
        config_path = _make_config()
        try:
            lm = MiniMax(
                config_path=config_path, model_name="minimax", cache=True
            )
            mock_response = _mock_chat_completion("cached result")
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = mock_response

            result1 = lm.query("Same query")
            result2 = lm.query("Same query")
            assert result1 == result2
            # Should only be called once due to caching
            lm.client.chat.completions.create.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_no_n_parameter(self):
        """Test that the n parameter is NOT passed to MiniMax API."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            mock_response = _mock_chat_completion("result")
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = mock_response

            lm.query("test", num_responses=1)
            call_kwargs = (
                lm.client.chat.completions.create.call_args.kwargs
            )
            assert "n" not in call_kwargs
        finally:
            os.unlink(config_path)


class TestMiniMaxGetResponseTexts:
    """Tests for MiniMax get_response_texts method."""

    def test_single_response(self):
        """Test extracting text from a single response."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            response = _mock_chat_completion("Hello world")
            texts = lm.get_response_texts(response)
            assert texts == ["Hello world"]
        finally:
            os.unlink(config_path)

    def test_list_of_responses(self):
        """Test extracting texts from a list of responses."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            responses = [
                _mock_chat_completion("First"),
                _mock_chat_completion("Second"),
                _mock_chat_completion("Third"),
            ]
            texts = lm.get_response_texts(responses)
            assert texts == ["First", "Second", "Third"]
        finally:
            os.unlink(config_path)


class TestMiniMaxTokenTracking:
    """Tests for MiniMax token and cost tracking."""

    def test_token_tracking(self):
        """Test that tokens are accumulated across calls."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            resp1 = _mock_chat_completion(
                "r1", prompt_tokens=100, completion_tokens=50
            )
            resp2 = _mock_chat_completion(
                "r2", prompt_tokens=200, completion_tokens=100
            )
            lm.client = MagicMock()
            lm.client.chat.completions.create.side_effect = [resp1, resp2]

            lm.chat([{"role": "user", "content": "q1"}])
            lm.chat([{"role": "user", "content": "q2"}])

            assert lm.prompt_tokens == 300
            assert lm.completion_tokens == 150
        finally:
            os.unlink(config_path)

    def test_cost_calculation(self):
        """Test that cost is calculated correctly."""
        config_path = _make_config(
            {"prompt_token_cost": 0.01, "response_token_cost": 0.03}
        )
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            response = _mock_chat_completion(
                "test", prompt_tokens=1000, completion_tokens=500
            )
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = response

            lm.chat([{"role": "user", "content": "test"}])

            # cost = 0.01 * 1.0 + 0.03 * 0.5 = 0.025
            assert abs(lm.cost - 0.025) < 1e-9
        finally:
            os.unlink(config_path)


class TestMiniMaxClearCache:
    """Tests for cache clearing."""

    def test_clear_cache(self):
        """Test that cache is cleared."""
        config_path = _make_config()
        try:
            lm = MiniMax(
                config_path=config_path, model_name="minimax", cache=True
            )
            mock_response = _mock_chat_completion("result")
            lm.client = MagicMock()
            lm.client.chat.completions.create.return_value = mock_response

            lm.query("test query")
            assert "test query" in lm.response_cache
            lm.clear_cache()
            assert len(lm.response_cache) == 0
        finally:
            os.unlink(config_path)


class TestMiniMaxConfigTemplate:
    """Tests for the config template containing MiniMax entries."""

    def test_config_template_has_minimax(self):
        """Test that config_template.json contains minimax entries."""
        config_dir = os.path.dirname(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "graph_of_thoughts",
                    "language_models",
                    "config_template.json",
                )
            )
        )
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "graph_of_thoughts",
            "language_models",
            "config_template.json",
        )
        with open(template_path) as f:
            config = json.load(f)

        assert "minimax" in config
        assert config["minimax"]["model_id"] == "MiniMax-M2.7"
        assert config["minimax"]["base_url"] == "https://api.minimax.io/v1"

        assert "minimax-highspeed" in config
        assert config["minimax-highspeed"]["model_id"] == "MiniMax-M2.7-highspeed"

    def test_config_template_minimax_fields(self):
        """Test that minimax config has all required fields."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "graph_of_thoughts",
            "language_models",
            "config_template.json",
        )
        with open(template_path) as f:
            config = json.load(f)

        required_fields = [
            "model_id",
            "prompt_token_cost",
            "response_token_cost",
            "temperature",
            "max_tokens",
            "stop",
            "api_key",
        ]
        for field in required_fields:
            assert field in config["minimax"], f"Missing field: {field}"
            assert field in config["minimax-highspeed"], (
                f"Missing field in highspeed: {field}"
            )


class TestMiniMaxImport:
    """Tests for MiniMax import from the language_models package."""

    def test_import_minimax(self):
        """Test that MiniMax can be imported from the package."""
        from graph_of_thoughts.language_models import MiniMax

        assert MiniMax is not None

    def test_minimax_is_abstract_language_model(self):
        """Test that MiniMax is a subclass of AbstractLanguageModel."""
        from graph_of_thoughts.language_models import (
            AbstractLanguageModel,
            MiniMax,
        )

        assert issubclass(MiniMax, AbstractLanguageModel)
