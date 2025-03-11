# Author: Saaket Agashe
# Date: 2021-09-15
# License: MIT

import base64
import re

from gui_agents.s1.mllm.MultimodalEngine import (
    LMMEngineAnthropic,
    LMMEngineAzureOpenAI,
    LMMEngineOpenAI,
    LMMEnginevLLM,
)

data_type_map = {
    "openai": {"image_url": "image_url"},
    "anthropic": {"image_url": "image"},
}


class LMMAgent:
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")
                if engine_type == "openai":
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "anthropic":
                    self.engine = LMMEngineAnthropic(**engine_params)
                elif engine_type == "azure":
                    self.engine = LMMEngineAzureOpenAI(**engine_params)
                elif engine_type == "vllm":
                    self.engine = LMMEnginevLLM(**engine_params)
                else:
                    raise ValueError("engine_type must be either 'openai' or 'azure'")
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages = []  # Empty messages

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def encode_image(self, image_content):
        # if image_content is a path to an image file, check type of the image_content to verify
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return base64.b64encode(image_content).decode("utf-8")

    def reset(
        self,
    ):

        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index):
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self, index, text_content, image_content=None, image_detail="high"
    ):
        """Replace a message at a given index"""
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }
            if image_content:
                base64_image = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

    def add_message(
        self, text_content, image_content=None, role=None, image_detail="high"
    ):
        """Add a new message to the list of messages"""

        # API-style inference from OpenAI and AzureOpenAI
        if isinstance(self.engine, (LMMEngineOpenAI, LMMEngineAzureOpenAI)):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        }
                    )
            self.messages.append(message)

        # For API-style inference from Anthropic
        elif isinstance(self.engine, LMMEngineAnthropic):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }
                    )
            self.messages.append(message)

        # Locally hosted vLLM model inference
        elif isinstance(self.engine, LMMEnginevLLM):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image",
                                "image": f"data:image;base64,{base64_image}",
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {"type": "image", "image": f"data:image;base64,{base64_image}"}
                    )
            self.messages.append(message)

    def get_response(
        self,
        user_message=None,
        image=None,
        messages=None,
        temperature=0.0,
        max_new_tokens=None,
        **kwargs,
    ):
        """Generate the next response based on previous messages"""
        if messages is None:
            messages = self.messages
        if user_message:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )

        return self.engine.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
