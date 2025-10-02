"""Tests for main module initialization."""
import os
import pytest
from unittest.mock import patch
from vulnhuntr.__main__ import initialize_llm
from vulnhuntr.LLMs import Claude, ChatGPT, Ollama, OpenRouter


class TestInitializeLLM:
    """Test LLM initialization function."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_initialize_claude(self):
        """Test Claude initialization."""
        llm = initialize_llm("claude", "Test prompt")
        assert isinstance(llm, Claude)
        assert llm.system_prompt == "Test prompt"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_initialize_gpt(self):
        """Test ChatGPT initialization."""
        llm = initialize_llm("gpt", "Test prompt")
        assert isinstance(llm, ChatGPT)
        assert llm.system_prompt == "Test prompt"

    def test_initialize_ollama(self):
        """Test Ollama initialization."""
        llm = initialize_llm("ollama", "Test prompt")
        assert isinstance(llm, Ollama)
        assert llm.system_prompt == "Test prompt"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_initialize_openrouter(self):
        """Test OpenRouter initialization."""
        llm = initialize_llm("openrouter", "Test prompt")
        assert isinstance(llm, OpenRouter)
        assert llm.system_prompt == "Test prompt"
        assert llm.model == "anthropic/claude-3.5-sonnet"  # default model

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_MODEL": "openai/gpt-4"
    })
    def test_initialize_openrouter_custom_model(self):
        """Test OpenRouter with custom model from env."""
        llm = initialize_llm("openrouter")
        assert isinstance(llm, OpenRouter)
        assert llm.model == "openai/gpt-4"

    def test_initialize_invalid_llm(self):
        """Test invalid LLM raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LLM argument"):
            initialize_llm("invalid")

    def test_initialize_case_insensitive(self):
        """Test LLM argument is case insensitive."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = initialize_llm("OpenRouter")
            assert isinstance(llm, OpenRouter)
