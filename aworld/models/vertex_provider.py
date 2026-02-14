import json
import os
from typing import Any, Dict, List, Generator, AsyncGenerator

import httpx

from aworld.utils import import_package
from aworld.logs.util import logger
from aworld.core.llm_provider import LLMProviderBase
from aworld.models.model_response import ModelResponse, LLMResponseError


class _VertexAuth(httpx.Auth):
    """httpx Auth handler that injects a fresh GCP access token on every request.

    Handles automatic token refresh for ADC and explicit credentials,
    so OpenAI-compatible Model Garden calls always use a valid bearer token.
    """

    def __init__(self, credentials):
        self.credentials = credentials

    def auth_flow(self, request):
        import google.auth.transport.requests
        if not self.credentials.token or (hasattr(self.credentials, 'expired') and self.credentials.expired):
            self.credentials.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


class VertexAIProvider(LLMProviderBase):
    """Google Vertex AI / Gemini API provider using the google-genai SDK.

    Supports two modes:
    - Vertex AI mode: Uses GCP project + credentials (ADC, access token, or credentials object)
    - Gemini Developer API mode: Uses API key authentication

    Vertex AI authentication options (via ext_config or env vars):
    - Application Default Credentials (default, no extra config needed)
    - access_token: Raw OAuth2 access token string (or GOOGLE_ACCESS_TOKEN env var)
    - credentials: A pre-built google.oauth2.credentials.Credentials object

    Model routing:
    - Gemini models (gemini-*): Uses google-genai SDK generateContent API
    - Model Garden models (e.g. zai-org/glm-4.7-maas): Uses Vertex AI's OpenAI-compatible endpoint

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
        self._vertex_credentials = kwargs.pop("credentials", None)
        self._vertex_access_token = kwargs.pop("access_token", None)

        import_package("google.genai", install_name="google-genai")

        # Model Garden OpenAI clients (initialized lazily in _init_provider if needed)
        self._mg_provider = None
        self._mg_async_provider = None

        super().__init__(api_key, base_url, model_name, sync_enabled, async_enabled, **kwargs)
        self._post_init()

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

        # 3. Auto-detect: credentials/token/project set → Vertex, api_key set → Gemini API
        if self._vertex_credentials or self._vertex_access_token or os.getenv("GOOGLE_ACCESS_TOKEN"):
            return True
        if self._vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT"):
            return True
        if self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return False

        # Default to Vertex AI mode
        return True

    def _is_gemini_model(self) -> bool:
        """Check if the current model is a native Gemini model.

        Gemini models use the google-genai generateContent API.
        Non-Gemini models (Model Garden) use the OpenAI-compatible endpoint.
        """
        model = self.model_name or ""
        return model.startswith("gemini-")

    def _build_vertex_openai_base_url(self) -> str:
        """Build the Vertex AI OpenAI-compatible endpoint URL for Model Garden models.

        For 'global' location: https://aiplatform.googleapis.com/v1/projects/{p}/locations/global/endpoints/openapi
        For regional locations: https://{loc}-aiplatform.googleapis.com/v1beta1/projects/{p}/locations/{loc}/endpoints/openapi
        """
        location = self._vertex_location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        project = self._vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if location == "global":
            return (
                f"https://aiplatform.googleapis.com/v1/"
                f"projects/{project}/locations/global/endpoints/openapi"
            )
        return (
            f"https://{location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{project}/locations/{location}/endpoints/openapi"
        )

    def _get_gcp_credentials(self):
        """Get GCP credentials for Model Garden API calls.

        Returns a google.auth.credentials.Credentials object with auto-refresh capability.
        """
        # 1. Explicit credentials object
        if self._vertex_credentials is not None:
            return self._vertex_credentials

        # 2. Access token → wrap in OAuth2 Credentials
        access_token = self._vertex_access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
        if access_token:
            from google.oauth2.credentials import Credentials as OAuth2Credentials
            return OAuth2Credentials(token=access_token)

        # 3. ADC
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return creds

    def _init_model_garden_clients(self):
        """Initialize OpenAI-compatible clients for Model Garden models on Vertex AI.

        Uses httpx auth middleware to inject fresh GCP bearer tokens automatically.
        """
        from openai import OpenAI, AsyncOpenAI

        credentials = self._get_gcp_credentials()
        base_url = self._build_vertex_openai_base_url()
        auth = _VertexAuth(credentials)

        if self.need_sync:
            self._mg_provider = OpenAI(
                api_key="PLACEHOLDER",
                base_url=base_url,
                http_client=httpx.Client(auth=auth),
            )
        if self.need_async:
            self._mg_async_provider = AsyncOpenAI(
                api_key="PLACEHOLDER",
                base_url=base_url,
                http_client=httpx.AsyncClient(auth=auth),
            )

        logger.info(f"Initialized Model Garden OpenAI client (base_url={base_url})")

    def _resolve_credentials(self):
        """Resolve credentials for Vertex AI mode.

        Priority:
        1. Explicit credentials object passed via ext_config
        2. Access token string (from ext_config or GOOGLE_ACCESS_TOKEN env var)
        3. None (falls back to Application Default Credentials in the SDK)

        Returns:
            A google.oauth2.credentials.Credentials object, or None for ADC.
        """
        # 1. Pre-built credentials object
        if self._vertex_credentials is not None:
            return self._vertex_credentials

        # 2. Raw access token → wrap in OAuth2 Credentials
        access_token = self._vertex_access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
        if access_token:
            from google.oauth2.credentials import Credentials as OAuth2Credentials
            return OAuth2Credentials(token=access_token)

        # 3. ADC (let the SDK handle it)
        return None

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

            credentials = self._resolve_credentials()
            client_kwargs = dict(vertexai=True, project=project, location=location)
            if credentials is not None:
                client_kwargs["credentials"] = credentials
                logger.info(f"Initializing Vertex AI provider with explicit credentials "
                            f"(project={project}, location={location})")
            else:
                logger.info(f"Initializing Vertex AI provider with ADC "
                            f"(project={project}, location={location})")

            return genai.Client(**client_kwargs)
        else:
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                    "environment variable or provide api_key in the parameters."
                )

            logger.info("Initializing Gemini Developer API provider")
            return genai.Client(api_key=api_key)

    def _post_init(self):
        """Called after both sync and async providers are initialized.
        Sets up Model Garden OpenAI clients for non-Gemini models.
        """
        if self._determine_mode() and not self._is_gemini_model():
            self._init_model_garden_clients()

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

    # ------------------------------------------------------------------ #
    #  Model Garden helpers (OpenAI-compatible endpoint)                   #
    # ------------------------------------------------------------------ #

    def _preprocess_messages_for_openai(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Preprocess messages for OpenAI-compatible Model Garden endpoint.

        Ensures tool_calls in assistant messages are plain dicts (not ToolCall objects).
        """
        from aworld.models.model_response import ToolCall
        processed = []
        for msg in messages:
            msg_copy = dict(msg)
            if msg_copy.get("role") == "assistant" and "tool_calls" in msg_copy and msg_copy["tool_calls"]:
                if msg_copy.get("content") is None:
                    msg_copy["content"] = ""
                serialized_tcs = []
                for tc in msg_copy["tool_calls"]:
                    if isinstance(tc, ToolCall):
                        serialized_tcs.append(tc.to_dict())
                    elif isinstance(tc, dict):
                        serialized_tcs.append(tc)
                    elif hasattr(tc, '__dict__'):
                        serialized_tcs.append(tc.__dict__)
                    else:
                        serialized_tcs.append(tc)
                msg_copy["tool_calls"] = serialized_tcs
            processed.append(msg_copy)
        return processed

    @staticmethod
    def _sanitize_usage(usage: dict) -> dict:
        """Replace None values in usage dict with 0 so Counter arithmetic works."""
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {k: (v if v is not None else 0) for k, v in usage.items()}

    def _postprocess_mg_stream_chunk(self, chunk: Any):
        """Process a Model Garden (OpenAI-compatible) stream chunk.

        Handles incremental tool call accumulation via stream_tool_buffer,
        matching the OpenAI provider's streaming logic.

        Returns:
            Tuple of (ModelResponse or None, finish_reason or None).
            Returns (None, None) for tool-call-only chunks with no text content.
        """
        from aworld.models.model_response import ToolCall

        # Error check
        if hasattr(chunk, 'error') or (isinstance(chunk, dict) and chunk.get('error')):
            error_msg = chunk.error if hasattr(chunk, 'error') else chunk.get('error', 'Unknown error')
            raise LLMResponseError(error_msg, self.model_name or "unknown", chunk)

        chunk_choice = None
        if hasattr(chunk, 'choices') and chunk.choices:
            chunk_choice = chunk.choices[0]
        elif isinstance(chunk, dict) and chunk.get("choices") and chunk["choices"]:
            chunk_choice = chunk["choices"][0]

        if not chunk_choice:
            return None, None

        # Accumulate tool call arguments
        delta_tool_calls = None
        if hasattr(chunk_choice, 'delta') and chunk_choice.delta:
            delta_tool_calls = getattr(chunk_choice.delta, 'tool_calls', None)
        elif isinstance(chunk_choice, dict):
            delta_tool_calls = chunk_choice.get("delta", {}).get("tool_calls")

        if delta_tool_calls:
            for tool_call in delta_tool_calls:
                index = getattr(tool_call, 'index', None)
                if index is None:
                    index = tool_call.get("index") if isinstance(tool_call, dict) else len(self.stream_tool_buffer)
                if index is None:
                    index = len(self.stream_tool_buffer)

                func = getattr(tool_call, 'function', None)
                if func is None and isinstance(tool_call, dict):
                    func_dict = tool_call.get("function", {})
                    func_name = func_dict.get("name")
                    func_args = func_dict.get("arguments", "")
                else:
                    func_name = getattr(func, 'name', None) if func else None
                    func_args = getattr(func, 'arguments', "") if func else ""

                tc_id = getattr(tool_call, 'id', None)
                if tc_id is None and isinstance(tool_call, dict):
                    tc_id = tool_call.get("id")

                if index >= len(self.stream_tool_buffer):
                    self.stream_tool_buffer.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": func_args or ""
                        }
                    })
                else:
                    self.stream_tool_buffer[index]["function"]["arguments"] += (func_args or "")

            # Tool-call-only chunk with no text — suppress
            delta_content = None
            if hasattr(chunk_choice, 'delta') and chunk_choice.delta:
                delta_content = getattr(chunk_choice.delta, 'content', None)
            elif isinstance(chunk_choice, dict):
                delta_content = chunk_choice.get("delta", {}).get("content")
            if not delta_content:
                return None, None

        # Check finish reason
        finish_reason = None
        if hasattr(chunk_choice, 'finish_reason'):
            finish_reason = chunk_choice.finish_reason
        elif isinstance(chunk_choice, dict):
            finish_reason = chunk_choice.get("finish_reason")

        # On finish with buffered tool calls, emit them as a complete chunk
        if finish_reason and self.stream_tool_buffer:
            tool_call_chunk = {
                "id": chunk.id if hasattr(chunk, 'id') else chunk.get("id", "unknown"),
                "model": chunk.model if hasattr(chunk, 'model') else chunk.get("model", "unknown"),
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": self.stream_tool_buffer
                    }
                }]
            }
            self.stream_tool_buffer = []
            resp = ModelResponse.from_openai_stream_chunk(tool_call_chunk)
            return resp, finish_reason

        resp = ModelResponse.from_openai_stream_chunk(chunk)
        return resp, finish_reason

    def _build_openai_params(self,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.0,
                             max_tokens: int = None,
                             stop: List[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """Build parameters for OpenAI-compatible chat completions (Model Garden)."""
        processed_messages = self._preprocess_messages_for_openai(messages)
        params = {
            "model": kwargs.get("model_name", self.model_name or ""),
            "messages": processed_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = stop
        for key in ("top_p", "tools", "tool_choice", "response_format"):
            if key in kwargs and kwargs[key]:
                params[key] = kwargs[key]
        return params

    # ------------------------------------------------------------------ #
    #  Completion methods                                                  #
    # ------------------------------------------------------------------ #

    def completion(self,
                   messages: List[Dict[str, str]],
                   temperature: float = 0.0,
                   max_tokens: int = None,
                   stop: List[str] = None,
                   **kwargs) -> ModelResponse:
        """Synchronously call Vertex AI to generate response.

        Routes to google-genai for Gemini models, OpenAI-compatible endpoint for Model Garden.
        """
        try:
            is_gemini = self._is_gemini_model()
            logger.info(f"Vertex AI completion: model={self.model_name}, "
                        f"is_gemini={is_gemini}, mg_provider={'yes' if self._mg_provider else 'no'}")
            if is_gemini:
                if not self.provider:
                    raise RuntimeError(
                        "Sync provider not initialized. Set 'sync_enabled' to True.")
                processed_data = self.preprocess_messages(messages)
                params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
                response = self.provider.models.generate_content(**params)
                return self.postprocess_response(response)
            else:
                if not self._mg_provider:
                    raise RuntimeError(
                        "Model Garden provider not initialized. Ensure Vertex AI mode is enabled "
                        "with a valid project. Set GOOGLE_CLOUD_PROJECT or pass project in ext_config.")
                openai_params = self._build_openai_params(messages, temperature, max_tokens, stop, **kwargs)
                response = self._mg_provider.chat.completions.create(**openai_params)
                return ModelResponse.from_openai_response(response)
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

        Routes to google-genai for Gemini models, OpenAI-compatible endpoint for Model Garden.
        """
        try:
            is_gemini = self._is_gemini_model()
            logger.info(f"Vertex AI acompletion: model={self.model_name}, "
                        f"is_gemini={is_gemini}, mg_async_provider={'yes' if self._mg_async_provider else 'no'}")
            if is_gemini:
                if not self.async_provider:
                    raise RuntimeError(
                        "Async provider not initialized. Set 'async_enabled' to True.")
                processed_data = self.preprocess_messages(messages)
                params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
                response = await self.async_provider.aio.models.generate_content(**params)
                return self.postprocess_response(response)
            else:
                if not self._mg_async_provider:
                    raise RuntimeError(
                        "Model Garden async provider not initialized. Ensure Vertex AI mode is enabled "
                        "with a valid project. Set GOOGLE_CLOUD_PROJECT or pass project in ext_config.")
                openai_params = self._build_openai_params(messages, temperature, max_tokens, stop, **kwargs)
                response = await self._mg_async_provider.chat.completions.create(**openai_params)
                return ModelResponse.from_openai_response(response)
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

        Routes to google-genai for Gemini models, OpenAI-compatible endpoint for Model Garden.
        """
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        try:
            if self._is_gemini_model():
                if not self.provider:
                    raise RuntimeError(
                        "Sync provider not initialized. Set 'sync_enabled' to True.")
                processed_data = self.preprocess_messages(messages)
                params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
                for chunk in self.provider.models.generate_content_stream(**params):
                    if not chunk:
                        continue
                    resp = self.postprocess_stream_response(chunk)
                    self._accumulate_chunk_usage(usage, self._sanitize_usage(resp.usage))
                    yield resp
            else:
                if not self._mg_provider:
                    raise RuntimeError(
                        "Model Garden provider not initialized. Ensure Vertex AI mode is enabled.")
                openai_params = self._build_openai_params(messages, temperature, max_tokens, stop, **kwargs)
                openai_params["stream"] = True
                openai_params["stream_options"] = {"include_usage": True}
                self.stream_tool_buffer = []
                for chunk in self._mg_provider.chat.completions.create(**openai_params):
                    if not chunk:
                        continue
                    resp, finish_reason = self._postprocess_mg_stream_chunk(chunk)
                    if resp:
                        self._accumulate_chunk_usage(usage, self._sanitize_usage(resp.usage))
                        yield resp
                        if finish_reason and resp.tool_calls:
                            yield ModelResponse(
                                id=resp.id, model=resp.model,
                                content="", finish_reason=finish_reason, usage=usage)

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

        Routes to google-genai for Gemini models, OpenAI-compatible endpoint for Model Garden.
        """
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        try:
            if self._is_gemini_model():
                if not self.async_provider:
                    raise RuntimeError(
                        "Async provider not initialized. Set 'async_enabled' to True.")
                processed_data = self.preprocess_messages(messages)
                params = self._build_generate_params(processed_data, temperature, max_tokens, stop, **kwargs)
                async for chunk in await self.async_provider.aio.models.generate_content_stream(**params):
                    if not chunk:
                        continue
                    resp = self.postprocess_stream_response(chunk)
                    self._accumulate_chunk_usage(usage, self._sanitize_usage(resp.usage))
                    yield resp
            else:
                if not self._mg_async_provider:
                    raise RuntimeError(
                        "Model Garden async provider not initialized. Ensure Vertex AI mode is enabled.")
                openai_params = self._build_openai_params(messages, temperature, max_tokens, stop, **kwargs)
                openai_params["stream"] = True
                openai_params["stream_options"] = {"include_usage": True}
                self.stream_tool_buffer = []
                stream = await self._mg_async_provider.chat.completions.create(**openai_params)
                async for chunk in stream:
                    if not chunk:
                        continue
                    resp, finish_reason = self._postprocess_mg_stream_chunk(chunk)
                    if resp:
                        self._accumulate_chunk_usage(usage, self._sanitize_usage(resp.usage))
                        yield resp
                        if finish_reason and resp.tool_calls:
                            yield ModelResponse(
                                id=resp.id, model=resp.model,
                                content="", finish_reason=finish_reason, usage=usage)

        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise e
            logger.warning(f"Error in Vertex AI astream_completion: {e}")
            raise LLMResponseError(str(e), kwargs.get("model_name", self.model_name or "gemini"))
