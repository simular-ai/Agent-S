## Deplying Agent-S in OSWorld
To use the GUIAgent with OSWorld, first follow the setup instructions at: https://github.com/xlang-ai/OSWorld.git

After completing the setup instructions, import the GraphSearchAgent into the run.py file in OSWorld. The GraphSearchAgent is the parent agent used in the Agent S framework. To understand the architecture of this GraphSearchAgent, refer to [Agent S Architecture](images/agent_s_architecture.pdf).

```
from agent_s.GraphSearchAgent import GraphSearchAgent
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

agent = GraphSearchAgent(
    engine_params=engine_params,
    experiment_type='osworld',
    platform="ubuntu",
    max_tokens=args.max_tokens,
    top_p=args.top_p,
    temperature=args.temperature,
    action_space=args.action_space,
    observation_type=args.observation_type,
    max_trajectory_length=args.max_trajectory_length,
    vm_version=args.vm_version,
)
```
The permissible values for the model argument are `gpt-4o`, `gpt-4o-mini` for OpenAI models, and `claude-3-5-sonnet-20240620` for Anthropic models. 