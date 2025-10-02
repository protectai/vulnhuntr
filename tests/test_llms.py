"""Tests for LLM classes."""
import os
import pytest
from unittest.mock import Mock, patch
from vulnhuntr.LLMs import OpenRouter, APIConnectionError, RateLimitError, APIStatusError
import openai


class TestOpenRouter:
    """Test OpenRouter LLM implementation."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_initialization(self):
        """Test OpenRouter initializes correctly."""
        model = "anthropic/claude-3.5-sonnet"
        base_url = "https://openrouter.ai/api/v1"
        system_prompt = "Test prompt"
        
        llm = OpenRouter(model, base_url, system_prompt)
        
        assert llm.model == model
        assert llm.system_prompt == system_prompt
        assert llm.client is not None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_create_messages(self):
        """Test message creation follows OpenAI format."""
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1", "System")
        
        messages = llm.create_messages("Hello")
        
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_send_message_success(self, mock_openai_class):
        """Test successful API call."""
        # Create mock response - create usage separately
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = mock_usage
        
        # Setup mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1", "System")
        llm.client = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        response = llm.send_message(messages, max_tokens=4096)
        
        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_send_message_api_connection_error(self, mock_openai_class):
        """Test handling of API connection errors."""
        mock_client = Mock()
        # OpenAI APIConnectionError requires a request parameter
        mock_request = Mock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(request=mock_request)
        mock_openai_class.return_value = mock_client
        
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1")
        llm.client = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(APIConnectionError):
            llm.send_message(messages, max_tokens=4096)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_send_message_rate_limit_error(self, mock_openai_class):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        # OpenAI RateLimitError requires response and body
        mock_response = Mock(status_code=429)
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body={}
        )
        mock_openai_class.return_value = mock_client
        
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1")
        llm.client = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(RateLimitError):
            llm.send_message(messages, max_tokens=4096)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_get_response(self, mock_openai_class):
        """Test response extraction."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1")
        
        result = llm.get_response(mock_response)
        
        assert result == "Test response"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_send_message_with_response_model(self, mock_openai_class):
        """Test send_message with response_model sets json_object format."""
        mock_usage = Mock()
        mock_usage.total_tokens = 50
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_response.usage = mock_usage
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        llm = OpenRouter("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1")
        llm.client = mock_client
        
        from pydantic import BaseModel
        class TestModel(BaseModel):
            key: str
        
        messages = [{"role": "user", "content": "Test"}]
        response = llm.send_message(messages, max_tokens=4096, response_model=TestModel)
        
        # Verify response_format was set
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_object"
