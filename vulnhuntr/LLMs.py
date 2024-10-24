import logging
from typing import List, Union, Dict, Any
from pydantic import BaseModel, ValidationError
import anthropic
import os
import openai
import requests
import google.generativeai as genai


log = logging.getLogger(__name__)

class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")


# Base LLM class to handle common functionality
class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        try:
            if self.prefill:
                response_text = self.prefill + response_text
            return response_model.model_validate_json(response_text)
        except ValidationError as e:
            log.warning("Response validation failed", exc_info=e)
            raise LLMError("Validation failed") from e
            # try:
            #     response_clean_attempt = response_text.split('{', 1)[1]
            #     return response_model.model_validate_json(response_clean_attempt)
            # except ValidationError as e:
            #     log.warning("Response validation failed", exc_info=e)
            #    raise LLMError("Validation failed") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Dict[str, Any]) -> None:
        usage_info = response.usage.__dict__
        log.debug("Received chat response", extra={"usage": usage_info})

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 4096) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        self._log_response(response)

        response_text = self.get_response(response)
        if response_model:
            response_text = self._validate_response(response_text, response_model) if response_model else response_text
        self._add_to_history("assistant", response_text)
        return response_text

class Claude(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = anthropic.Anthropic(max_retries=3)

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{    \"scratchpad\": \"1."
            messages = [{"role": "user", "content": user_prompt}, 
                        {"role": "assistant", "content": self.prefill}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            # response_model is not used here, only in ChatGPT
            return self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text.replace('\n', '')


class ChatGPT(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Retrieves API key from environment variable

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model) -> Dict[str, Any]:
        try:
            # For analyzing files and context code, use the beta endpoint and parse so we can feed it the pydantic model
            if response_model:
                return self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_model
                )
            else:
                return self.client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    max_tokens=max_tokens,
                )
        except openai.APIConnectionError as e:
            raise APIConnectionError("The server could not be reached") from e
        except openai.RateLimitError as e:
            raise RateLimitError("Request was rate-limited; consider backing off") from e
        except openai.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred: {str(e)}") from e

    def _clean_response(self, response: str) -> str:
        # Step 1: Remove markdown code block wrappers
        cleaned_text = response.strip('```json\n').strip('```')
        # Step 2: Correctly handle newlines and escaped characters
        cleaned_text = cleaned_text.replace('\n', '').replace('\\\'', '\'')
        # Step 3: Replace escaped double quotes with regular double quotes
        cleaned_text = cleaned_text.replace('\\"', '"')

        return response.replace('\n', '')

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        cleaned_response = self._clean_response(response)
        return cleaned_response

class Ollama(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = "http://localhost:11434/api/chat"

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        payload = {
            "model": "codellama:7b",
            "messages": messages,
            "options": {
            "seed": 101,
            "temperature": 1
            }
            ,"stream":False,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            return response
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif e.response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(e.response.status_code, e.response.json()) from e


    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.json()['message']['content']
        return response


    def _log_response(self, response: Dict[str, Any]) -> None:
        log.debug("Received chat response", extra={"usage": "Ollama"})

class Gemini(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client=genai.GenerativeModel("gemini-1.5-pro")

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "model", "parts": self.system_prompt}, {"role": "user", "parts": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:

        if "response_format" in messages[1]['parts']:
            messages[1]['parts'] = "use None instaed of null, only return properties without $defs in response for adapt python\n" + messages[1]['parts']
        try:
            response = self.client.generate_content(contents=messages, generation_config={"temperature": 1},safety_settings="block_none",stream=False)
            return response
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif e.response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(e.response.status_code, e.response.json()) from e

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        try:
            if self.prefill:
                response_text = self.prefill + response_text
            return response_model.model_validate_json(response_text)
        except ValidationError as e:
            log.warning("Response validation failed", exc_info=e)
            log.warning(response_text)
            raise LLMError("Validation failed") from e

    def _clean_response(self, response: str) -> str:
        cleaned_text = response.strip('```json').strip('```xml').strip('```')
        cleaned_text = cleaned_text.replace("\n", '').replace("\\'", '\'').strip()
        return cleaned_text

    def get_response(self, response: Dict[str, Any]) -> str:
        response = self._clean_response(response.text)
        return response


    def _log_response(self, response: Dict[str, Any]) -> None:
        log.debug("Received chat response", extra={"usage": response.usage_metadata.total_token_count})