import base64
import re

from gui_agents.mllm.MultimodalAsyncEngine import (
    LMMEngineAsyncAzureOpenAI,
    LMMEngineAsyncOpenAI,
)
from gui_agents.mllm.MultimodalEngine import LMMEngineCogVLM, LMMEngineLlava


class LMMAgentAsync:
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")
                if engine_type == "openai":
                    self.engine = LMMEngineAsyncOpenAI(**engine_params)
                elif engine_type == "azure":
                    self.engine = LMMEngineAsyncAzureOpenAI(**engine_params)
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
        if isinstance(self.engine, (LMMEngineCogVLM, LMMEngineLlava)):
            self.messages = []
        else:
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

        # Don't add the system prompt if we are using llava or other hf models
        if isinstance(self.engine, LMMEngineLlava) or isinstance(
            self.engine, LMMEngineCogVLM
        ):
            self.messages = []

    def remove_message_at(self, index):
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def add_message(self, text_content, image_content=None, detail="high", role=None):
        """Add a new message to the list of messages"""

        # For inference from locally hosted llava based on https://github.com/haotian-liu/LLaVA/
        if isinstance(self.engine, LMMEngineLlava):

            # No system prompt so first message will be from user
            if len(self.messages) == 0:
                role = "user"
            else:
                # infer role from previous message
                if self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            image_token_se = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )

            qs = text_content
            if role == "user":
                if len(self.messages) == 0:
                    # If this is the very first user message, add the system prompt to it to dictate behavior
                    qs = self.system_prompt + "\n" + qs
                    # TODO: Add comment explaining what this next part does
                    if IMAGE_PLACEHOLDER in qs:
                        if self.engine.model.config.mm_use_im_start_end:
                            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                        else:
                            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                    else:
                        if self.engine.model.config.mm_use_im_start_end:
                            qs = image_token_se + "\n" + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                message = {"role": role, "content": qs}
            else:
                message = {"role": role, "content": text_content}

            # Capable of handling only one image right now. TODO: make capable of handling more images
            if image_content:
                if self.engine.args.image_file == None:
                    self.engine.args.image_file = image_content

            self.messages.append(message)

        elif isinstance(self.engine, LMMEngineCogVLM):
            # No system prompt so first message will be from user
            if len(self.messages) == 0:
                role = "user"
            else:
                # infer role from previous message
                if self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            # Add message content as a new message, if this is the first message prepend with system prompt
            if len(self.messages) == 0:
                self.messages.append(
                    {
                        "role": role,
                        "content": {
                            "type": "text",
                            "text": self.system_prompt + "\n\n" + text_content,
                        },
                    }
                )
            else:
                self.messages.append(
                    {"role": role, "content": {"type": "text", "text": text_content}}
                )

        # For API-style inference from OpenAI and AzureOpenAI
        else:
            # infer role from previous message
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
                base64_image = self.encode_image(image_content)
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    }
                )

            self.messages.append(message)

    async def get_response(
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

        response = await self.engine.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs
        )

        return response
