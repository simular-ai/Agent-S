from gui_agents.s3.core.messages import LMMMessage


class LMMEngine:
    """Common base for multimodal generation engines."""

    def generate(
        self,
        messages: list[LMMMessage],
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str | None:
        raise NotImplementedError

    def generate_with_thinking(
        self,
        messages: list[LMMMessage],
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str | None:
        return self.generate(messages, temperature, max_new_tokens, **kwargs)
