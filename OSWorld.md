## Deplying Agent-S in OSWorld
To use the GUIAgent with OSWorld, first follow the setup instructions at: https://github.com/xlang-ai/OSWorld.git

After completing the setup instructions, import the GraphSearchAgent into the run.py file in OSWorld. The GraphSearchAgent is the parent agent used in the Agent S framework. To understand the architecture of this GraphSearchAgent, refer to [Agent S Architecture](images/agent_s_architecture.pdf).

```
from agent_s.aci.OSWorldACI import OSWorldACI
from agent_s.core.AgentS import GraphSearchAgent
```

Replace the PromptAgent on line 138 in the test() method with the Graph Search Agent. Specify engine params and instantiate the agent as shown:

```
if args.model.startswith("claude"):
  engine_type = "anthropic"
elif args.model.startswith("gpt"):
  engine_type = "openai"
else:
  engine_type = "vllm"

engine_params = {
  "engine_type": engine_type,
  "model": args.model,
}

grounding_agent = OSWorldACI(vm_version=args.vm_version)
agent = GraphSearchAgent(
  engine_params,
  grounding_agent,
  platform='ubuntu',
  action_space="pyautogui",
  observation_type="mixed",
  search_engine="Perplexica"
)
```
We support all multimodal models from OpenAI, Anthropic, and vLLM. For more information, refer to [models.md](models.md).

We have set the latest Agent S to use the latest Ubuntu VM image from OSWorld. However, our experiments are based on the older version of the VM. To reproduce the results, set the vm_version argument to 'old' while instantiating the agent.

