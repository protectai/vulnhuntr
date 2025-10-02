import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
from vulnhuntr.LLMs import LLM, Claude, ChatGPT, Ollama


class SampleResponseModel(BaseModel):
    """Sample response model for testing."""
    result: str
    confidence: int


class TestLLMResponseFormat:
    """Test that response format is properly handled by LLM classes."""

    def test_base_llm_appends_response_format_to_prompt(self):
        """Base LLM class should append JSON schema to user prompt."""
        llm = LLM(system_prompt="test")
        user_prompt = "Analyze this code"
        response_model = SampleResponseModel

        prepared_prompt = llm.prepare_user_prompt(user_prompt, response_model)

        # Should contain original prompt
        assert "Analyze this code" in prepared_prompt
        # Should contain response format instructions
        assert "response format" in prepared_prompt.lower() or "json" in prepared_prompt.lower()
        # Should contain schema
        schema_str = json.dumps(response_model.model_json_schema(), indent=4)
        assert schema_str in prepared_prompt

    def test_base_llm_without_response_model_returns_prompt_unchanged(self):
        """Base LLM should return prompt unchanged when no response model provided."""
        llm = LLM(system_prompt="test")
        user_prompt = "Analyze this code"

        prepared_prompt = llm.prepare_user_prompt(user_prompt, None)

        assert prepared_prompt == user_prompt

    def test_chatgpt_excludes_response_format_from_prompt(self):
        """ChatGPT should NOT append JSON schema to prompt (uses native API support)."""
        with patch('vulnhuntr.LLMs.openai.OpenAI'):
            llm = ChatGPT(model="gpt-4", base_url="https://api.openai.com/v1", system_prompt="test")
            user_prompt = "Analyze this code"
            response_model = SampleResponseModel

            prepared_prompt = llm.prepare_user_prompt(user_prompt, response_model)

            # Should return prompt unchanged (ChatGPT uses native API support)
            assert prepared_prompt == user_prompt
            # Should NOT contain schema
            schema_str = json.dumps(response_model.model_json_schema(), indent=4)
            assert schema_str not in prepared_prompt

    def test_claude_appends_response_format_to_prompt(self):
        """Claude should append JSON schema to user prompt."""
        with patch('vulnhuntr.LLMs.anthropic.Anthropic'):
            llm = Claude(model="claude-3", base_url="https://api.anthropic.com", system_prompt="test")
            user_prompt = "Analyze this code"
            response_model = SampleResponseModel

            prepared_prompt = llm.prepare_user_prompt(user_prompt, response_model)

            # Should contain original prompt
            assert "Analyze this code" in prepared_prompt
            # Should contain schema
            schema_str = json.dumps(response_model.model_json_schema(), indent=4)
            assert schema_str in prepared_prompt

    def test_ollama_appends_response_format_to_prompt(self):
        """Ollama should append JSON schema to user prompt."""
        llm = Ollama(model="llama2", base_url="http://localhost:11434/api/generate", system_prompt="test")
        user_prompt = "Analyze this code"
        response_model = SampleResponseModel

        prepared_prompt = llm.prepare_user_prompt(user_prompt, response_model)

        # Should contain original prompt
        assert "Analyze this code" in prepared_prompt
        # Should contain schema
        schema_str = json.dumps(response_model.model_json_schema(), indent=4)
        assert schema_str in prepared_prompt

    def test_chat_method_uses_prepare_user_prompt(self):
        """The chat method should use prepare_user_prompt to process the prompt."""
        with patch('vulnhuntr.LLMs.anthropic.Anthropic'):
            llm = Claude(model="claude-3", base_url="https://api.anthropic.com", system_prompt="test")

            # Mock the methods
            llm.prepare_user_prompt = Mock(return_value="prepared prompt")
            llm.create_messages = Mock(return_value=[])

            # Create mock response using MagicMock for complex attributes
            mock_content = MagicMock()
            mock_content.text = '{"result": "test", "confidence": 8}'
            mock_response = MagicMock()
            mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
            mock_response.content = [mock_content]

            llm.send_message = Mock(return_value=mock_response)
            llm.get_response = Mock(return_value='{"result": "test", "confidence": 8}')

            user_prompt = "Analyze this"
            response_model = SampleResponseModel

            llm.chat(user_prompt, response_model=response_model)

            # Verify prepare_user_prompt was called
            llm.prepare_user_prompt.assert_called_once_with(user_prompt, response_model)
            # Verify create_messages received prepared prompt
            llm.create_messages.assert_called_once_with("prepared prompt")

    def test_chatgpt_send_message_receives_response_model(self):
        """ChatGPT send_message should still receive response_model for native API support."""
        mock_client = Mock()

        # Create mock response using MagicMock
        mock_message = MagicMock()
        mock_message.content = '{"result": "test", "confidence": 8}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.usage.__dict__ = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        with patch('vulnhuntr.LLMs.openai.OpenAI', return_value=mock_client):
            llm = ChatGPT(model="gpt-4", base_url="https://api.openai.com/v1", system_prompt="test")

            messages = [{"role": "user", "content": "test"}]
            response_model = SampleResponseModel

            llm.send_message(messages, max_tokens=4096, response_model=response_model)

            # Verify API was called with response_format
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"]["type"] == "json_object"
