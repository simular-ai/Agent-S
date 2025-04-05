# Deplying Agent S2 in OSWorld

# Step 1: Set up Agent S2

Follow the [README.md](https://github.com/simular-ai/Agent-S/blob/main/README.md) to set up Agent S2.

# Step 2: Copying Over Run Files

If you haven't already, please follow the [OSWorld environment setup](https://github.com/xlang-ai/OSWorld/blob/main/README.md). We've provided the relevant OSWorld run files for evaluation in this `osworld_setup` folder. Please copy this over to your OSWorld folder.

# Best Practices

At this point, you will have set up the Agent S2, the OSWorld environment, and the VMWare Workstation Pro application set up. Below, we'll list some best practices, and common problems and their fixes.

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

Note, this code is just for demonstrating how the OSWorld `DesktopEnv` is instantiated. If you're running OSWorld, this process is already part of their code base. The code above will boot up a VM and restart it. If, for whatever reason, running the starter code (or running OSWorld experiments) leads to an infinitely long run time, cancel out of the VM.
You should then see:

```
parent/
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
delete the `*.lck` folders first. If your VM is already powered on, and your session (in a Jupyter Notebook, for example) crashes, you can keep the `*.lck` files and just re-instantiate the `DesktopEnv` instance. I'd also suggest using just a single VM (as a VM takes up a lot of space!). Also, be sure to shut down the VM when you've finished using it. Deleting the `*.lck` files should be done after every time you power off the VM (though it seems to not be an issue from testing).

---

If even after rerunning the code and deleting the `*.lck` files don't work, then you should try passing in the `path_to_vm` explicitly to the `DesktopEnv` class. 

```
env = DesktopEnv(action_space="pyautogui", headless=False, require_terminal=True, path_to_vm=<absolute_path>)
```

Pass the absolute path to your VM's (Ubuntu0) `.vmx` file. This file is located here:


```
parent/
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

📌 **Note**: OSWorld scripts will create the `DesktopEnv` instance which will create a VM for you with a specific snapshot (`snapshot_name` parameter in `DesktopEnv`). If you wish to create a new snapshot of the VM and use that for your experiments, be sure to specify the name of this snapshot where `DesktopEnv` is instantiated.

With these changes, you should be able to get up and running with VMWare, DesktopEnv, and OSWorld! 😊