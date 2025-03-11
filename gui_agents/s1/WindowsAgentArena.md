## Deploying Agent-S in WindowsAgentArena
> ⚠️ **Warning**: The refactored code has not be fully tested on WindowsAgentArena. To reproduce the results on WindowsAgentArena, please use commit 496a9fa of this repository.

1. To use the Agent S with WindowsAgentArena, follows the setup instructions at: https://github.com/microsoft/WindowsAgentArena.git. **Please use the development mode while preparing the image and running the client as instructed in https://github.com/microsoft/WindowsAgentArena/blob/main/docs/Development-Tips.md.** 

2. To deploy our agent in the WindowsAgentArena, copy the agent_s folder in this repository to  `WindowsAgentArena/src/win-arena-container/client/mm_agents`. 

3. Change the name of the GraphSearchAgent.py file to agent.py to conform to the WindowsAgentArena Setup. 

4. Copy the ocr_server.py file to client/folder `WindowsAgentArena/src/win-arena-container/client` folder

```
cd WindowsAgentArena/src/win-arena-container/client
cp mm_agents/agent_s/ocr_server.py .
```

5. Update the `start_client.sh` file in `WindowsAgentArena/src/win-arena-container` by adding the following line before Running the agent on line 75. 

```
python ocr_server.py &
```

6. In the `src/win-arena-container/client/run.py` file import Agent S
```
from mm_agents.agent_s.agent import GraphSearchAgent
```

7. In the `src/win-arena-container/client/run.py` file, instantiate Agent S by adding the following lines after line 187 where the if condition for NAVI agent ends 

```python
elif cfg_args["agent_name"] == "agent_s":
  if cfg_args["som_origin"] in ["a11y"]:
    som_config = None
  elif cfg_args["som_origin"] in ["oss", "mixed-oss"]:
    som_config = {
      "pipeline": ["webparse", "groundingdino", "ocr"],
      "groundingdino": {
        "prompts": ["icon", "image"]
      },
      "ocr": {
        "class_name": "TesseractOCR"
      },
      "webparse": {
        "cdp_url": f"http://{args.emulator_ip}:9222"
      }
    }
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
    experiment_type='windowsAgentArena',
    temperature=args.temperature
  )
```

8. Run Agent S on WindowsAgentArena by changing the following parameters in the `scripts/run-local.sh` file

```
agent="agent_s"
model="gpt-4o"
```