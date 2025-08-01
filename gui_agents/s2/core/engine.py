import os

import backoff
import numpy as np
from anthropic import Anthropic
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)
from google import genai
from google.genai import types


class LMMEngine:
    pass


class OpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key=None,
    ):
        """Init an OpenAI Embedding engine

        Args:
            embedding_model (str, optional): Model name. Defaults to "text-embedding-3-small".
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
        """
        self.model = embedding_model
        self.api_key = api_key

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
        ),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model, input=text)
        return np.array([data.embedding for data in response.data])


class GeminiEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-004",
        api_key=None,
    ):
        """Init an Gemini Embedding engine

        Args:
            embedding_model (str, optional): Model name. Defaults to "text-embedding-004".
            api_key (_type_, optional): Auth key from Gemini. Defaults to None.
        """
        self.model = embedding_model
        self.api_key = api_key

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
        ),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        client = genai.Client(api_key=api_key)

        result = client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        return np.array([i.values for i in result.embeddings])


class AzureOpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key=None,
        api_version=None,
        endpoint_url=None,
    ):
        """Init an Azure OpenAI Embedding engine

        Args:
            embedding_model (str, optional): Model name. Defaults to "text-embedding-3-small".
            api_key (_type_, optional): Auth key from Azure OpenAI. Defaults to None.
            api_version (_type_, optional): API version. Defaults to None.
            endpoint_url (_type_, optional): Endpoint URL. Defaults to None.
        """
        self.model = embedding_model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint_url = endpoint_url

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
        ),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "An API Version needs to be provided in either the api_version parameter or as an environment variable named OPENAI_API_VERSION"
            )
        endpoint_url = self.endpoint_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint_url is None:
            raise ValueError(
                "An Endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint_url,
        )
        response = client.embeddings.create(input=text, model=self.model)
        return np.array([data.embedding for data in response.data])


class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(api_key=api_key)
            else:
                self.llm_client = OpenAI(base_url=self.base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, thinking=False, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        if not self.llm_client:
            self.llm_client = Anthropic(api_key=api_key)
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **kwargs,
            )
            thoughts = full_response.content[0].thinking
            print("CLAUDE 3.7 THOUGHTS:", thoughts)
            return full_response.content[1].text
        return (
            self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .content[0]
            .text
        )


class LMMEngineGemini(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named GEMINI_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineOpenRouter(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENROUTER_API_KEY"
            )
        base_url = self.base_url or os.getenv("OPEN_ROUTER_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named OPEN_ROUTER_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_version = api_version
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.cost = 0.0

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "api_version must be provided either as a parameter or as an environment variable named OPENAI_API_VERSION"
            )
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        if not self.llm_client:
            self.llm_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost += 0.02 * ((total_tokens + 500) / 1000)
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs
    ):
        api_key = self.api_key or os.getenv("vLLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "A vLLM API key needs to be provided in either the api_key parameter or as an environment variable named vLLM_API_KEY"
            )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content


class LMMEngineHuggingFace(LMMEngine):
    def __init__(self, base_url=None, api_key=None, rate_limit=-1, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("HF_TOKEN")
        if api_key is None:
            raise ValueError(
                "A HuggingFace token needs to be provided in either the api_key parameter or as an environment variable named HF_TOKEN"
            )
        base_url = self.base_url
        if base_url is None:
            raise ValueError(
                "HuggingFace endpoint must be provided as base_url parameter."
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model="tgi",
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineParasail(LMMEngine):
    def __init__(self, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "Parasail model id must be provided"
        self.model = model
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("PARASAIL_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Parasail API key needs to be provided in either the api_key parameter or as an environment variable named PARASAIL_API_KEY"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url="https://api.parasail.io/v1", api_key=api_key
            )
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )
