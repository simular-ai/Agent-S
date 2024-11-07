We support the following APIs for MLLM inference: OpenAI, Anthropic, Azure OpenAI, and vLLM for Local Models. To use these APIs, you need to set the corresponding environment variables:

1. OpenAI

```python
export OPENAI_API_KEY=<YOUR_API_KEY>
```

2. Anthropic

```python
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
```

3. OpenAI on Azure

```python
export AZURE_OPENAI_API_BASE=<DEPLOYMENT_NAME>
export AZURE_OPENAI_API_KEY=<YOUR_API_KEY>
```

4. vLLM for Local Models

```python
export vLLM_ENDPOINT_URL=<YOUR_DEPLOYMENT_URL>
```

Alternatively you can directly pass the API keys into the engine_params argument while instantating the agent.

```python
from agent_s.GraphSearchAgent import GraphSearchAgent
engine_params = {
    "engine_type": 'anthropic', # Allowed Values: 'openai', 'anthropic', 'azure_openai', 'vllm'
    "model": 'claude-3-5-sonnet-20240620', # Allowed Values: Any Vision and Language Model from the supported APIs
}
agent = GraphSearchAgent(
    engine_params,
    grounding_agent,
    platform=current_platform,
    action_space="pyautogui",
    observation_type="mixed",
    search_engine="LLM"
)
```

To use the underlying Multimodal Agent (LMMAgent) which wraps LLMs with message handling functionality, you can use the following code snippet:

```python
engine_params = {
    "engine_type": 'anthropic', # Allowed Values: 'openai', 'anthropic', 'azure_openai', 'vllm'
    "model": 'claude-3-5-sonnet-20240620', # Allowed Values: Any Vision and Language Model from the supported APIs
    }
from agent_s.MultimodalAgent import LMMAgent
agent = LMMAgent(
    engine_params = engine_params,
)
```

The GraphSearchAgent also utilizes this LMMAgent internally.