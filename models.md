We support the following APIs for MLLM inference: OpenAI, Anthropic, Gemini, Azure OpenAI, vLLM for local models, and Open Router. To use these APIs, you need to set the corresponding environment variables:

1. OpenAI

```
export OPENAI_API_KEY=<YOUR_API_KEY>
```

2. Anthropic

```
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
```

3. Gemini

```
export GEMINI_API_KEY=<YOUR_API_KEY>
export GEMINI_ENDPOINT_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

4. OpenAI on Azure

```
export AZURE_OPENAI_API_BASE=<DEPLOYMENT_NAME>
export AZURE_OPENAI_API_KEY=<YOUR_API_KEY>
```

5. vLLM for Local Models

```
export vLLM_ENDPOINT_URL=<YOUR_DEPLOYMENT_URL>
```

6. LM Studio (or any OpenAI-compatible local server)

No pip or env setup needed. In LM Studio: load a **vision-capable** chat
model for planning (e.g. Qwen2-VL / Qwen2.5-VL) and **UI-TARS-1.5-7B** for
grounding, then Start Server (default `http://localhost:1234/v1`, any API
key works — loopback endpoints use a dummy key when no key is configured).

```python
engine_params = {
    "engine_type": "lmstudio",  # alias of openai + local defaults
    "model": "<model id shown in LM Studio>",
    "base_url": "http://localhost:1234/v1",  # optional, this is the default
    "api_key": "lm-studio",                 # optional, this is the default
}
```

`engine_type="openai_compatible"` works the same for other local servers
(Ollama at `http://localhost:11434/v1`, llama.cpp, text-generation-webui…).
Grounding dimensions for UI-TARS-1.5-7B: `grounding_width=1920`,
`grounding_height=1080`. Note: local planners are weaker than frontier
models at following the strict action format — keep `max_trajectory_length`
small (3–4) if the context window is tight.

Prefer a UI? Run `agent_s_ui`, open http://127.0.0.1:8000, and configure
both the planner and grounding endpoints from the browser. That server also
exposes `GET /v1/models` and `POST /v1/chat/completions` (use
`"model": "agent-s3"` to run the desktop agent, any other model id is
proxied to the configured planner backend).

Alternatively you can directly pass the API keys into the engine_params argument while instantating the agent.

7. Open Router

```
export OPENROUTER_API_KEY=<YOUR_API_KEY>
export OPEN_ROUTER_ENDPOINT_URL="https://openrouter.ai/api/v1"
```

```python
from gui_agents.s2_5.agents.agent_s import AgentS2_5

engine_params = {
    "engine_type": 'openai', # Allowed Values: 'openai', 'anthropic', 'gemini', 'azure_openai', 'vllm', 'open_router'
    "model": 'gpt-5-2025-08-07', # Allowed Values: Any Vision and Language Model from the supported APIs
}
agent = AgentS2_5(
    engine_params,
    grounding_agent,
    platform=current_platform,
)
```

To use the underlying Multimodal Agent (LMMAgent) which wraps LLMs with message handling functionality, you can use the following code snippet:

```python
from gui_agents.s2_5.core.mllm import LMMAgent

engine_params = {
    "engine_type": 'openai', # Allowed Values: 'openai', 'anthropic', 'gemini', 'azure_openai', 'vllm', 'open_router'
    "model": 'gpt-5-2025-08-07', # Allowed Values: Any Vision and Language Model from the supported APIs
    }
agent = LMMAgent(
    engine_params=engine_params,
)
```

The `AgentS2_5` also utilizes this `LMMAgent` internally.
