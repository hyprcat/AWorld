import json
import os
from typing import Any, Dict, List, Generator, AsyncGenerator

from aworld.utils import import_package
from aworld.logs.util import logger
from aworld.core.llm_provider import LLMProviderBase
from aworld.models.model_response import ModelResponse, LLMResponseError


class VertexAIProvider(LLMProviderBase):
    """Google Vertex AI / Gemini API provider using the google-genai SDK.

    Supports two modes:
    - Vertex AI mode: Uses GCP project + Application Default Credentials
    - Gemini Developer API mode: Uses API key authentication

    Mode is determined by explicit config, environment variables, or auto-detection.
    """

    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 sync_enabled: bool = None,
                 async_enabled: bool = None,
                 **kwargs):
        # Extract Vertex-specific kwargs before super().__init__ calls _init_provider
        self._vertex_project = kwargs.pop("project", None)
        self._vertex_location = kwargs.pop("location", None)
        self._vertex_explicit = kwargs.pop("vertexai", None)

        import_package("google.genai", install_name="google-genai")

        super().__init__(api_key, base_url, model_name, sync_enabled, async_enabled, **kwargs)

    def _determine_mode(self) -> bool:
        """Determine whether to use Vertex AI mode or Gemini Developer API mode.

        Returns:
            True for Vertex AI mode, False for Gemini Developer API mode.
        """
        # 1. Explicit kwarg
        if self._vertex_explicit is not None:
            return bool(self._vertex_explicit)

        # 2. Environment variable
        env_val = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")

        # 3. Auto-detect: project set → Vertex, api_key set → Gemini API
        if self._vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT"):
            return True
        if self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return False

        # Default to Vertex AI mode
        return True

    def _init_provider(self):
        """Initialize google-genai client.

        Returns:
            google.genai.Client instance.
        """
        from google import genai

        use_vertex = self._determine_mode()

        if use_vertex:
            project = self._vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = self._vertex_location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

            if not project:
                raise ValueError(
                    "Google Cloud project not found. Set GOOGLE_CLOUD_PROJECT environment variable "
                    "or provide 'project' in ext_config."
                )

            logger.info(f"Initializing Vertex AI provider (project={project}, location={location})")
            return genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                    "environment variable or provide api_key in the parameters."
                )

            logger.info("Initializing Gemini Developer API provider")
            return genai.Client(api_key=api_key)

    def _init_async_provider(self):
        """Initialize async provider. google-genai uses client.aio for async operations,
        so we return the same client instance.

        Returns:
            Same google.genai.Client instance (uses .aio for async calls).
        """
        if self.provider:
            return self.provider
        return self._init_provider()

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gemini-.*"]

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI-format messages to google-genai format.

        Args:
            messages: OpenAI format message list.

        Returns:
            Dictionary with 'contents' and 'system_instruction' keys.
        """
        from google.genai import types

        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content

            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)] if content else []
                    )
                )

            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))

                # Handle tool calls in assistant messages
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            name = func.get("name", "")
                            args_str = func.get("arguments", "{}")
                        else:
                            name = tc.function.name if hasattr(tc, 'function') else ""
                            args_str = tc.function.arguments if hasattr(tc, 'function') else "{}"

                        try:
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except (json.JSONDecodeError, TypeError):
                            args_dict = {}

                        parts.append(
                            types.Part.from_function_call(
                                name=name,
                                args=args_dict
                            )
                        )

                contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                tool_content = content
                try:
                    response_dict = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                except (json.JSONDecodeError, TypeError):
                    response_dict = {"result": tool_content}

                if not isinstance(response_dict, dict):
                    response_dict = {"result": str(response_dict)}

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response=response_dict
                            )
                        ]
                    )
                )

        return {
            "contents": contents,
            "system_instruction": system_instruction
        }

    def _convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Any]:
        """Convert OpenAI tool format to google-genai FunctionDeclaration list.

        Args:
            openai_tools: Tools in OpenAI format.

        Returns:
            List of google.genai.types.FunctionDeclaration objects.
        """
        from google.genai import types

        declarations = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", tool)
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters")

                declarations.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=parameters
                    )
                )

        return declarations

    def _build_generate_params(self,
                               processed_data: Dict[str, Any],
                               temperature: float = 0.0,
                               max_tokens: int = None,
                               stop: List[str] = None,
                               **kwargs) -> Dict[str, Any]:
        """Build parameters for generate_content call.

        Args:
            processed_data: Preprocessed message data with 'contents' and 'system_instruction'.
            temperature: Temperature parameter.
            max_tokens: Maximum output tokens.
            stop: Stop sequences.
            **kwargs: Additional parameters.

        Returns:
            Dictionary of parameters for generate_content.
        """
        from google.genai import types

        config_params = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        if stop:
            config_params["stop_sequences"] = stop

        if "top_p" in kwargs:
            config_params["top_p"] = kwargs["top_p"]

        if "top_k" in kwargs:
            config_params["top_k"] = kwargs["top_k"]

        # System instruction
        system_instruction = processed_data.get("system_instruction")
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        # Tools
        if "tools" in kwargs and kwargs["tools"]:
            declarations = self._convert_tools(kwargs["tools"])
            if declarations:
                config_params["tools"] = [types.Tool(function_declarations=declarations)]

                # Tool choice
                tool_choice = kwargs.get("tool_choice")
                if tool_choice:
                    if isinstance(tool_choice, str):
                        choice_map = {
                            "auto": "AUTO",
                            "none": "NONE",
                            "required": "ANY",
                        }
                        mode = choice_map.get(tool_choice, "AUTO")
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(mode=mode)
                        )
                    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                        func_name = tool_choice.get("function", {}).get("name")
                        fcc_kwargs = {"mode": "ANY"}
                        if func_name:
                            fcc_kwargs["allowed_function_names"] = [func_name]
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(**fcc_kwargs)
                        )

        # Response format
        response_format = kwargs.get("response_format")
        if response_format:
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                config_params["response_mime_type"] = "application/json"
            elif isinstance(response_format, str) and response_format == "json_object":
                config_params["response_mime_type"] = "application/json"

        params = {
            "model": kwargs.get("model_name", self.model_name or ""),
            "contents": processed_data["contents"],
            "config": types.GenerateContentConfig(**config_params),
        }

        return params

    def postprocess_response(self, response: Any) -> ModelResponse:
        """Process Vertex AI response to unified ModelResponse.

        Args:
            response: google-genai response object.

        Returns:
            ModelResponse object.

        Raises:
            LLMResponseError: When response contains errors.
        """
        if not response:
            raise LLMResponseError("Empty response from Vertex AI", self.model_name or "gemini", response)

        return ModelResponse.from_vertex_response(response, self.model_name or "gemini")

    def postprocess_stream_response(self, chunk: Any) -> ModelResponse:
        """Process Vertex AI streaming chunk to unified ModelResponse.

        Args:
            chunk: google-genai streaming chunk.

        Returns:
            ModelResponse object.

        Raises:
            LLMResponseError: When chunk contains errors.
        """
        if not chunk:
            raise LLMResponseError("Empty chunk from Vertex AI", self.model_name or "gemini", chunk)

        return ModelResponse.from_vertex_stream_chunk(chunk, self.model_name or "gemini")

    def completion(self,
                   messages: List[Dict[str, str]],
                   temperature: float = 0.0,
                   max_tokens: int = None,
                   stop: List[str] = None,
                   **kwargs) -> ModelResponse:
        """Synchronously call Vertex AI to generate response.

        Args:
            messages: Message list.
            temperature: Temperature parameter.
            max_tokens: Maximum number of tokens to generate.
            stop: List of stop sequences.
            **kwargs: Other parameters.

        Returns:
            ModelResponse object.

        Raises:
            LLMResponseError: When API call fails.
        """
        if not self.provider:
            raise RuntimeError(
                "Sync provider not initialized. Make sure 'sync_enabled' parameter is set to True in initialization.")

        try:
            processed_data = self.preprocess_messages(messages)
            params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
            response = self.provider.models.generate_content(**params)
            return self.postprocess_response(response)
        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise e
            logger.warning(f"Error in Vertex AI completion: {e}")
            raise LLMResponseError(str(e), kwargs.get("model_name", self.model_name or "gemini"))

    async def acompletion(self,
                          messages: List[Dict[str, str]],
                          temperature: float = 0.0,
                          max_tokens: int = None,
                          stop: List[str] = None,
                          **kwargs) -> ModelResponse:
        """Asynchronously call Vertex AI to generate response.

        Args:
            messages: Message list.
            temperature: Temperature parameter.
            max_tokens: Maximum number of tokens to generate.
            stop: List of stop sequences.
            **kwargs: Other parameters.

        Returns:
            ModelResponse object.

        Raises:
            LLMResponseError: When API call fails.
        """
        if not self.async_provider:
            raise RuntimeError(
                "Async provider not initialized. Make sure 'async_enabled' parameter is set to True in initialization.")

        try:
            processed_data = self.preprocess_messages(messages)
            params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
            response = await self.async_provider.aio.models.generate_content(**params)
            return self.postprocess_response(response)
        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise e
            logger.warning(f"Error in Vertex AI acompletion: {e}")
            raise LLMResponseError(str(e), kwargs.get("model_name", self.model_name or "gemini"))

    def stream_completion(self,
                          messages: List[Dict[str, str]],
                          temperature: float = 0.0,
                          max_tokens: int = None,
                          stop: List[str] = None,
                          **kwargs) -> Generator[ModelResponse, None, None]:
        """Synchronously call Vertex AI to generate streaming response.

        Args:
            messages: Message list.
            temperature: Temperature parameter.
            max_tokens: Maximum number of tokens to generate.
            stop: List of stop sequences.
            **kwargs: Other parameters.

        Returns:
            Generator yielding ModelResponse chunks.

        Raises:
            LLMResponseError: When API call fails.
        """
        if not self.provider:
            raise RuntimeError(
                "Sync provider not initialized. Make sure 'sync_enabled' parameter is set to True in initialization.")

        try:
            processed_data = self.preprocess_messages(messages)
            params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)

            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

            for chunk in self.provider.models.generate_content_stream(**params):
                if not chunk:
                    continue
                resp = self.postprocess_stream_response(chunk)
                self._accumulate_chunk_usage(usage, resp.usage)
                yield resp

        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise e
            logger.warning(f"Error in Vertex AI stream_completion: {e}")
            raise LLMResponseError(str(e), kwargs.get("model_name", self.model_name or "gemini"))

    async def astream_completion(self,
                                 messages: List[Dict[str, str]],
                                 temperature: float = 0.0,
                                 max_tokens: int = None,
                                 stop: List[str] = None,
                                 **kwargs) -> AsyncGenerator[ModelResponse, None]:
        """Asynchronously call Vertex AI to generate streaming response.

        Args:
            messages: Message list.
            temperature: Temperature parameter.
            max_tokens: Maximum number of tokens to generate.
            stop: List of stop sequences.
            **kwargs: Other parameters.

        Returns:
            AsyncGenerator yielding ModelResponse chunks.

        Raises:
            LLMResponseError: When API call fails.
        """
        if not self.async_provider:
            raise RuntimeError(
                "Async provider not initialized. Make sure 'async_enabled' parameter is set to True in initialization.")

        try:
            processed_data = self.preprocess_messages(messages)
            params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)

            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

            async for chunk in await self.async_provider.aio.models.generate_content_stream(**params):
                if not chunk:
                    continue
                resp = self.postprocess_stream_response(chunk)
                self._accumulate_chunk_usage(usage, resp.usage)
                yield resp

        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise e
            logger.warning(f"Error in Vertex AI astream_completion: {e}")
            raise LLMResponseError(str(e), kwargs.get("model_name", self.model_name or "gemini"))
