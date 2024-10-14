# Author: Saaket Agashe
# Date: 2021-09-15
# License: MIT

import os
import backoff
import openai
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)

from anthropic import Anthropic

import requests
from PIL import Image
from io import BytesIO
import re
import torch
import numpy as np


# TODO: Import only if module exists, else ignore
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import (
#     process_images,
#     tokenizer_image_token,
#     get_model_name_from_path,
#     KeywordsStoppingCriteria,
# )
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
#     IMAGE_PLACEHOLDER,
# )
# from llava.conversation import conv_templates, SeparatorStyle


# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class LMMEngine:
    pass

class LMMEngineOpenAI(LMMEngine):
    def __init__(self, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY")

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(api_key=self.api_key)

    @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60)
    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        ).choices[0].message.content

class LMMEngineAnthropic(LMMEngine):
    def __init__(
            self,
            api_key=None,
            model=None,
            **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY")

        self.api_key = api_key

        self.llm_client = Anthropic(api_key=self.api_key)


    @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60)
    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        return self.llm_client.messages.create(
            system=messages[0]['content'][0]['text'],
            model=self.model,
            messages=messages[1:],
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        ).content[0].text


class LMMEngineQwen(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        self.model = model
        self.api_key = api_key

        self.base_url = base_url or os.getenv("QWEN_ENDPOINT_URL")
        if self.base_url is None:
            raise ValueError("An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL")


    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''

        data = {
            'messages': messages,
        }

        response = requests.post(self.base_url, json=data)
        # Check the response
        if response.status_code == 200:
            return response.json()['response'][0]
        else:
            print(f"Qwen LLM generation failed with status code: {response.status_code}")
            print("Error message:", response.text)


class OpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        api_key = None,
        rate_limit: int = -1,
        display_cost: bool = True,
    ):
        """Init an OpenAI Embedding engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            display_cost (bool, optional): Display cost of API call. Defaults to True.
        """
        self.model = "text-embedding-3-small"
        self.cost_per_thousand_tokens = 0.00002

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY")
        self.api_key = api_key
        self.display_cost = display_cost
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
        ),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=text)
        if self.display_cost:
            total_tokens = response.usage.total_tokens
            cost = self.cost_per_thousand_tokens * total_tokens / 1000
            # print(f"Total cost for this embedding API call: {cost}")
        return np.array([data.embedding for data in response.data])

class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(self, api_key=None, azure_endpoint=None, model=None, api_version=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        assert api_version is not None, "api_version must be provided"
        self.api_version = api_version

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY")

        self.api_key = api_key

        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint is None:
            raise ValueError("An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_API_BASE")

        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = AzureOpenAI(azure_endpoint=self.azure_endpoint, api_key=self.api_key, api_version=self.api_version)
        self.cost = 0.

    # @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_tries=10)
    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost +=  0.02 * ((total_tokens+500) / 1000)
        return completion.choices[0].message.content

class LMMEnginevLLM(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key

        self.base_url = base_url or os.getenv("vLLM_ENDPOINT_URL")
        if self.base_url is None:
            raise ValueError("An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL")

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_tries=10)
    # TODO: Default params chosen for the Qwen model
    def generate(self, messages, temperature=0., top_p=0.8, repetition_penalty=1.05, max_new_tokens=512, **kwargs):
        '''Generate the next message based on previous messages'''
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content


class LMMEngineLlava(LMMEngine):
    def __init__(self, model_path=None, model = None, tokenizer=None, image_processor=None, context_len=None, max_new_tokens=None, rate_limit=-1, **kwargs):

        assert model_path is not None, "model path must be provided"
        self.model_path = model_path

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit


        self.args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": None,
            "conv_mode": None,
            "image_file": None,
            "sep": ",",
            "temperature": 0.,
            "top_p":1,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens if max_new_tokens else 2048
        })()

        if not model:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, self.args.model_name)
        else:
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self.context_len = context_len

        # Check model base type for conversation template
        if "llama-2" in self.args.model_name.lower():
            self.args.conv_mode = "llava_llama_2"
        elif "v1" in self.args.model_name.lower():
            self.args.conv_mode = "llava_v1"
        elif "mpt" in self.args.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.args.conv_mode = "llava_v0"

        self.conversation = conv_templates[self.args.conv_mode].copy()


    def generate(self, messages, image=None, temperature=0., max_new_tokens=None, **kwargs):

        # Refresh the conversation holder everytime
        self.conversation = conv_templates[self.args.conv_mode].copy()
        '''Generate the next message based on previous messages'''
        for idx, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[idx % 2], message['content'])

        # Add the "ASSISTANT:" starter before generation

        self.conversation.append_message(self.conversation.roles[1], None)
        prompt = self.conversation.get_prompt()
        self.args.image_files = [self.args.image_file]
        image_files = image_parser(self.args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)


        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

class LMMEngineCogVLM(LMMEngine):
    def __init__(self, model_path=None, model = None, tokenizer=None, image_processor=None, context_len=None, max_new_tokens=None, device=None, rate_limit=-1, **kwargs):
        assert model_path is not None, "model path must be provided"
        self.model_path = model_path

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        if device:
            self.device = device
        else: self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }
        if not model:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_type,
                trust_remote_code=True
            ).eval().to(self.device)
        else:
            self.tokenizer = tokenizer
            self.model = model

        self.history = None

    def generate(self, messages, image=None, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        if image:
            image = Image.open(image).convert('RGB')
        history = []
        if len(messages) > 1:
            history_list = [m["content"]["text"] for m in messages[:-1]]
            # Group two messages at a time add them as a tuple to history
            history = list(zip(history_list[0::2], history_list[1::2]))

        if image is None:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=messages[-1]["content"]["text"],
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=messages[-1]["content"]["text"],
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return respons
