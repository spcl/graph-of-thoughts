"""Integration tests for MiniMax language model.

These tests require a valid MINIMAX_API_KEY environment variable.
They make real API calls to the MiniMax API.

Run with: pytest tests/test_minimax_integration.py -v
"""

import json
import os
import tempfile

import pytest

from graph_of_thoughts.language_models.minimax import MiniMax


pytestmark = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


def _make_config(model_id="MiniMax-M2.7", model_name="minimax"):
    """Create a temporary config file for integration tests."""
    config = {
        model_name: {
            "model_id": model_id,
            "prompt_token_cost": 0.001,
            "response_token_cost": 0.005,
            "temperature": 0.7,
            "max_tokens": 256,
            "stop": None,
            "base_url": "https://api.minimax.io/v1",
            "api_key": "",
        }
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(config, tmp)
    tmp.close()
    return tmp.name


class TestMiniMaxIntegration:
    """Integration tests that make real API calls."""

    def test_simple_query(self):
        """Test a simple query to MiniMax M2.7."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            response = lm.query("What is 2+2? Reply with just the number.")
            texts = lm.get_response_texts(response)
            assert len(texts) == 1
            assert "4" in texts[0]
            assert lm.prompt_tokens > 0
            assert lm.completion_tokens > 0
            assert lm.cost > 0
        finally:
            os.unlink(config_path)

    def test_multiple_responses(self):
        """Test querying with multiple responses."""
        config_path = _make_config()
        try:
            lm = MiniMax(config_path=config_path, model_name="minimax")
            response = lm.query(
                "What is the capital of France? Reply with just the city name.",
                num_responses=2,
            )
            texts = lm.get_response_texts(response)
            assert len(texts) == 2
            for text in texts:
                assert "Paris" in text or "paris" in text.lower()
        finally:
            os.unlink(config_path)

    def test_highspeed_model(self):
        """Test querying with the highspeed model variant."""
        config_path = _make_config(
            model_id="MiniMax-M2.7-highspeed",
            model_name="minimax-hs",
        )
        try:
            lm = MiniMax(
                config_path=config_path, model_name="minimax-hs"
            )
            response = lm.query("Say 'hello' in one word.")
            texts = lm.get_response_texts(response)
            assert len(texts) == 1
            assert len(texts[0]) > 0
        finally:
            os.unlink(config_path)
