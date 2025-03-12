# Deplying Agent-S in OSWorld

## Step 1: Environment Setup

Assuming you've followed the guide in the [README.md](README.md), your repository structure should look like:

```
parent/
  └── Agent-S/
```

The next step is to follow the set up instructions for OSWorld: https://github.com/xlang-ai/OSWorld.git.

To easily run Agent-S on OSWorld locally, We recommend moving your OSWorld local repository to the parent directory of Agent-S.

```
parent/
  ├── Agent-S/
  └── OSWorld/
```

We suggest creating a separate conda environment for each repository to avoid dependency conflicts. 

## Step 2: Modifying OSWorld `run.py`

After completing the setup instructions, import the GraphSearchAgent into the run.py file in OSWorld. The GraphSearchAgent is the parent agent used in the Agent S framework. To understand the architecture of this GraphSearchAgent, refer to [Agent S Architecture](images/agent_s_architecture.pdf).

```
from gui_agents.s1.aci.LinuxOSACI import LinuxACI
from gui_agents.s1.core.AgentS import GraphSearchAgent
```

Replace the PromptAgent on line 138 in the test() method with the Graph Search Agent. Specify engine params and instantiate the agent as shown:

```
parser.add_argument("--vm_version", type=str, default="new")

...

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

grounding_agent = LinuxACI(vm_version=args.vm_version)
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


# Step 3: Best Practices

At this point, you will have set up the Agent-S and OSWorld environments and the VMWare Workstation Pro application. Below, we'll list some best practices, and common problems and their fixes.

---

```
from desktop_env.desktop_env import DesktopEnv

example = {
    "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
    "instruction": "I want to install Spotify on my current system. Could you please help me?",
    "config": [
        {
            "type": "execute",
            "parameters": {
                "command": [
                    "python",
                    "-c",
                    "import pyautogui; import time; pyautogui.click(960, 540); time.sleep(0.5);"
                ]
            }
        }
    ],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "which spotify"
        },
        "expected": {
            "type": "rule",
            "rules": {
                "include": ["spotify"],
                "exclude": ["not found"]
            }
        }
    }
}

env = DesktopEnv(action_space="pyautogui")

obs = env.reset(task_config=example)
obs, reward, done, info = env.step("pyautogui.rightClick()")
```

The code above will boot up a VM and restart it. If, for whatever reason, running the starter code below leads to an infinitely long run time, cancel out of the VM.
You should then see:

```
parent/
  Agent-S/
  OSWorld/
    vmware_vm_data/
      Ubuntu0/
        *.lck
        *.vmem
        ...
      ...
      UbuntuX/
```

If you happen to have any `*.lck` folder in your VM's folder, be sure to delete them. Every time you are powering on the VM from creating a new `DesktopEnv` instance, you need to 
delete the `*.lck` folders first. If your VM is already powered on, and your session (in a Jupyter Notebook, for example) crashes, you can keep the `*.lck` files and just re-instantiate the `DesktopEnv` instance. I'd also suggest using just a single VM (as a VM takes up a lot of space!). 

---

If even after rerunning the code and deleting the `*.lck` files don't work, then you should try passing in the `path_to_vm` explicitly to the `DesktopEnv` class. 

```
env = DesktopEnv(action_space="pyautogui", headless=False, require_terminal=True, path_to_vm=<absolute_path>)
```

Pass the absolute path to your VM's (Ubuntu0) `.vmx` file. This file is located here:


```
parent/
  Agent-S/
  OSWorld/
    vmware_vm_data/
      Ubuntu0/
        *.lck
        *.vmem
        ...
        *.vmx
      ...
      UbuntuX/
```

📌 **Note**: If you are testing on the `os` domain, there is an [issue](https://github.com/asweigart/pyautogui/issues/198#issuecomment-1465268536) with `pyautogui`. A *hacky* way to solve this is to, inside the VM, locate where the `pyautogui` module is installed and open the `__init__.py` located under the `pyautogui` folder and remove the "<" in the `set(...)` within the following function: 

```
def isShiftCharacter(character):
    """
    Returns True if the ``character`` is a keyboard key that would require the shift key to be held down, such as
    uppercase letters or the symbols on the keyboard's number row.
    """
    # NOTE TODO - This will be different for non-qwerty keyboards.
    return character.isupper() or character in set('~!@#$%^&*()_+{}|:"<>?')
```

📌 **Note**: If in case, your VM encounters an issue with "The root file system on <path> requires a manual fsck", reset the VM to the previous snapshot. 

With these changes, you should be able to get up and running with VMWare, DesktopEnv, and OSWorld! 😊