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

Alternatively you can directly pass the API keys into the engine_params argument while instantating the agent.

6. Open Router

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