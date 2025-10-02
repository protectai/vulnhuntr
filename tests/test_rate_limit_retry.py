"""Tests for rate limit retry logic and state persistence."""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from vulnhuntr.LLMs import Claude, ChatGPT, RateLimitError
import anthropic
import openai


class TestRateLimitRetry:
    """Test exponential backoff retry logic for rate limits."""

    def test_claude_retry_on_rate_limit(self):
        """Test that Claude retries with exponential backoff on rate limit."""
        claude = Claude(model="claude-3-sonnet-20240229", base_url="https://api.anthropic.com", system_prompt="test")

        # Mock the client to raise RateLimitError twice, then succeed
        mock_response = Mock()
        mock_response.content = [Mock(text="test response")]
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {}

        # Create proper RateLimitError with required parameters
        mock_error_response = Mock()
        mock_error = anthropic.RateLimitError("Rate limited", response=mock_error_response, body={})

        claude.client.messages.create = Mock(side_effect=[
            mock_error,
            mock_error,
            mock_response
        ])

        messages = [{"role": "user", "content": "test"}]
        start_time = time.time()
        result = claude.send_message(messages, max_tokens=100, response_model=None)
        elapsed = time.time() - start_time

        # Should succeed after 2 retries
        assert result == mock_response
        # Should have waited ~3 seconds (1s + 2s backoff)
        assert elapsed >= 3.0
        assert claude.client.messages.create.call_count == 3

    def test_claude_max_retries_exceeded(self):
        """Test that Claude raises error after max retries exceeded."""
        claude = Claude(model="claude-3-sonnet-20240229", base_url="https://api.anthropic.com", system_prompt="test")

        # Mock to always raise RateLimitError
        mock_error_response = Mock()
        mock_error = anthropic.RateLimitError("Rate limited", response=mock_error_response, body={})
        claude.client.messages.create = Mock(side_effect=mock_error)

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(RateLimitError, match="Max retries exceeded"):
            claude.send_message(messages, max_tokens=100, response_model=None)

        # Should attempt initial + 5 retries = 6 total
        assert claude.client.messages.create.call_count == 6

    def test_gpt_retry_on_rate_limit(self):
        """Test that ChatGPT retries with exponential backoff on rate limit."""
        gpt = ChatGPT(model="gpt-4", base_url="https://api.openai.com/v1", system_prompt="test")

        # Mock the client to raise RateLimitError once, then succeed
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="test response"))]
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {}

        mock_error_response = Mock()
        mock_error = openai.RateLimitError("Rate limited", response=mock_error_response, body={})

        gpt.client.chat.completions.create = Mock(side_effect=[
            mock_error,
            mock_response
        ])

        messages = [{"role": "user", "content": "test"}]
        start_time = time.time()
        result = gpt.send_message(messages, max_tokens=100, response_model=None)
        elapsed = time.time() - start_time

        # Should succeed after 1 retry
        assert result == mock_response
        # Should have waited ~1 second
        assert elapsed >= 1.0
        assert gpt.client.chat.completions.create.call_count == 2

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are correct."""
        claude = Claude(model="claude-3-sonnet-20240229", base_url="https://api.anthropic.com", system_prompt="test")

        mock_response = Mock()
        mock_response.content = [Mock(text="test response")]
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {}

        # Create proper RateLimitError with required parameters
        mock_error_response = Mock()
        mock_error = anthropic.RateLimitError("Rate limited", response=mock_error_response, body={})

        # Fail 3 times, then succeed
        claude.client.messages.create = Mock(side_effect=[
            mock_error,
            mock_error,
            mock_error,
            mock_response
        ])

        messages = [{"role": "user", "content": "test"}]
        start_time = time.time()
        result = claude.send_message(messages, max_tokens=100, response_model=None)
        elapsed = time.time() - start_time

        # Should wait 1s + 2s + 4s = 7s total
        assert elapsed >= 7.0
        assert elapsed < 8.0  # Allow small margin
        assert result == mock_response

    def test_non_rate_limit_error_not_retried(self):
        """Test that non-rate-limit errors are not retried."""
        claude = Claude(model="claude-3-sonnet-20240229", base_url="https://api.anthropic.com", system_prompt="test")

        # Mock to raise a different error
        from vulnhuntr.LLMs import APIConnectionError
        # APIConnectionError requires request parameter
        mock_request = Mock()
        mock_conn_error = anthropic.APIConnectionError(request=mock_request)
        claude.client.messages.create = Mock(side_effect=mock_conn_error)

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(APIConnectionError):
            claude.send_message(messages, max_tokens=100, response_model=None)

        # Should only attempt once (no retries)
        assert claude.client.messages.create.call_count == 1


class TestStateCheckpointing:
    """Test state persistence for resuming analysis."""

    def test_save_checkpoint(self):
        """Test saving analysis checkpoint to file."""
        from vulnhuntr.__main__ import save_checkpoint
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_file = f.name

        checkpoint_data = {
            'file': '/path/to/file.py',
            'stage': 'initial_analysis',
            'analysis_report': {'confidence_score': 5, 'vulnerability_types': ['RCE']},
            'vuln_type': None,
            'iteration': 0
        }

        save_checkpoint(checkpoint_file, checkpoint_data)

        # Verify file was created and contains correct data
        with open(checkpoint_file, 'r') as f:
            loaded = json.load(f)

        assert loaded == checkpoint_data

    def test_load_checkpoint(self):
        """Test loading analysis checkpoint from file."""
        from vulnhuntr.__main__ import load_checkpoint
        import tempfile
        import json

        checkpoint_data = {
            'file': '/path/to/file.py',
            'stage': 'secondary_analysis',
            'analysis_report': {'confidence_score': 7, 'vulnerability_types': ['SQLI']},
            'vuln_type': 'SQLI',
            'iteration': 2
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(checkpoint_data, f)
            checkpoint_file = f.name

        loaded = load_checkpoint(checkpoint_file)
        assert loaded == checkpoint_data

    def test_load_nonexistent_checkpoint(self):
        """Test loading checkpoint that doesn't exist returns None."""
        from vulnhuntr.__main__ import load_checkpoint

        result = load_checkpoint('/nonexistent/checkpoint.json')
        assert result is None

    def test_resume_from_checkpoint(self):
        """Test that analysis can resume from checkpoint."""
        from vulnhuntr.__main__ import should_skip_file

        checkpoint_data = {
            'file': '/path/to/file.py',
            'stage': 'completed',
            'analysis_report': {'confidence_score': 8, 'vulnerability_types': ['XSS']},
            'vuln_type': None,
            'iteration': 0
        }

        # Should skip if file is completed
        assert should_skip_file('/path/to/file.py', checkpoint_data) == True

        # Should not skip if file is different
        assert should_skip_file('/path/to/other.py', checkpoint_data) == False

        # Should not skip if same file but not completed
        checkpoint_data['stage'] = 'secondary_analysis'
        assert should_skip_file('/path/to/file.py', checkpoint_data) == False
