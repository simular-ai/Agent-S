<h1 align="center">
  <img src="images/agent_s.png" alt="Logo" style="vertical-align:middle" width="60"> Agent S2:
  <small>An Open, Modular, and Scalable Framework for Computer Use Agents</small>
</h1>

<p align="center">&nbsp;
  🌐 <a href="https://www.simular.ai/agent-s2">[S2 blog]</a>&nbsp;
  📄 [S2 Paper] (Coming Soon)&nbsp;
  🎥 <a href="https://www.youtube.com/watch?v=wUGVQl7c0eg">[S2 Video]</a>
  🗨️ <a href="https://discord.gg/E2XfsK9fPV">[Discord]</a>&nbsp;
</p>

<p align="center">&nbsp;
  🌐 <a href="https://www.simular.ai/agent-s">[S1 blog]</a>&nbsp;
    🎥 <a href="https://www.youtube.com/watch?v=OBDE3Knte0g">[S1 Video]</a>
  📄 <a href="https://arxiv.org/abs/2410.08164">[S1 Paper]</a>&nbsp;
</p>

## 🥳 Updates
- [x] **2025/03/12**: Released Agent S2 along with v0.2.0 of [gui-agents](https://github.com/simular-ai/Agent-S), the new state-of-the-art on OSWorld, outperforming OpenAI's CUA and Anthropic's Claude 3.7 Sonnet!
- [x] **2025/01/22**: The [Agent S paper](https://arxiv.org/abs/2410.08164) is accepted to ICLR 2025!
- [x] **2025/01/21**: Released v0.1.2 of [gui-agents](https://github.com/simular-ai/Agent-S) library, with support for Linux and Windows!
- [x] **2024/12/05**: Released v0.1.0 of [gui-agents](https://github.com/simular-ai/Agent-S) library, allowing you to use Agent-S for Mac, OSWorld, and WindowsAgentArena with ease!
- [x] **2024/10/10**: Released [Agent S paper](https://arxiv.org/abs/2410.08164) and codebase!

## Table of Contents

1. [💡 Introduction](#-introduction)
2. [🎯 Current Results](#-current-results)
3. [🛠️ Installation](#%EF%B8%8F-installation) 
4. [🚀 Usage](#-usage)
5. [🤝 Acknowledgements](#-acknowledgements)
6. [💬 Citation](#-citation)

## 💡 Introduction

<p align="center">
    <img src="./images/agent_s2_teaser.png" width="800">
</p>

Welcome to **Agent S**, an open-source framework designed to enable autonomous interaction with computers through Agent-Computer Interface. Our mission is to build intelligent GUI agents that can learn from past experiences and perform complex tasks autonomously on your computer. 

Whether you're interested in AI, automation, or contributing to cutting-edge agent-based systems, we're excited to have you here!

## 🎯 Current Results

<p align="center">
    <img src="./images/agent_s2_osworld_result.png" width="600">
    <br>
    Results of Agent S2's Successful Rate (%) on the OSWorld full test set of all 369 test examples using Image input.
</p>

| Benchmark | Agent S2 | Previous SOTA | Δ improve |
|-----------|----------|--------------|-----------|
| OSWorld (15 step) | 27.0% | 22.7% (ByteDance UI-TARS) | +4.3% |
| OSWorld (50 step) | 34.5% | 32.6% (OpenAI CUA) | +1.9% |
| AndroidWorld | 60.0% | 46.8% (ByteDance UI-TARS) | +13.2% |

## 🛠️ Installation & Setup

> ❗**Warning**❗: If you are on a Linux machine, creating a `conda` environment will interfere with `pyatspi`. As of now, there's no clean solution for this issue. Proceed through the installation without using `conda` or any virtual environment.

> ⚠️**Disclaimer**⚠️: To leverage the full potential of Agent S2, we utilize [UI-TARS](https://github.com/bytedance/UI-TARS) as a grounding model (7B-DPO or 72B-DPO for better performance). They can be hosted locally, on Hugging Face Inference Endpoints, or SageMaker. Our code supports Hugging Face Inference Endpoints and SageMaker. Fortunately, running Agent S2 does not require this model. Check out [Hugging Face Inference Endpoints](https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints) for more information on how to set up and query this endpoint.

Clone the repository:
```
git clone https://github.com/simular-ai/Agent-S.git
```

Install the gui-agents package:
```
pip install gui-agents
```

Set your LLM API Keys and other environment variables. You can do this by adding the following line to your .bashrc (Linux), or .zshrc (MacOS) file. 

```
export OPENAI_API_KEY=<YOUR_API_KEY>
```

Alternatively, you can set the environment variable in your Python script:

```
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

We also support Azure OpenAI, Anthropic, and vLLM inference. For more information refer to [models.md](models.md).

### Setup Retrieval from Web using Perplexica
Agent S works best with web-knowledge retrieval. To enable this feature, you need to setup Perplexica: 

1. Ensure Docker Desktop is installed and running on your system.

2. Navigate to the directory containing the project files.

   ```bash
    cd Perplexica
    git submodule update --init
   ```

3. Rename the `sample.config.toml` file to `config.toml`. For Docker setups, you need only fill in the following fields:

   - `OPENAI`: Your OpenAI API key. **You only need to fill this if you wish to use OpenAI's models**.
   - `OLLAMA`: Your Ollama API URL. You should enter it as `http://host.docker.internal:PORT_NUMBER`. If you installed Ollama on port 11434, use `http://host.docker.internal:11434`. For other ports, adjust accordingly. **You need to fill this if you wish to use Ollama's models instead of OpenAI's**.
   - `GROQ`: Your Groq API key. **You only need to fill this if you wish to use Groq's hosted models**.
   - `ANTHROPIC`: Your Anthropic API key. **You only need to fill this if you wish to use Anthropic models**.

     **Note**: You can change these after starting Perplexica from the settings dialog.

   - `SIMILARITY_MEASURE`: The similarity measure to use (This is filled by default; you can leave it as is if you are unsure about it.)

4. Ensure you are in the directory containing the `docker-compose.yaml` file and execute:

   ```bash
   docker compose up -d
   ```

5. Our implementation of Agent S incorporates the Perplexica API to integrate a search engine capability, which allows for a more convenient and responsive user experience. If you want to tailor the API to your settings and specific requirements, you may modify the URL and the message of request parameters in  `agent_s/query_perplexica.py`. For a comprehensive guide on configuring the Perplexica API, please refer to [Perplexica Search API Documentation](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md)

For a more detailed setup and usage guide, please refer to the [Perplexica Repository](https://github.com/ItzCrazyKns/Perplexica.git).

> ❗**Warning**❗: The agent will directly run python code to control your computer. Please use with care.

## 🚀 Usage

### CLI

Run agent_s on your computer using:  
```
agent_s2 --model gpt-4o
```
This will show a user query prompt where you can enter your query and interact with Agent S2. You can use any model from the list of supported models in [models.md](models.md).

### `gui_agents` SDK

```
import pyautogui
import io
from gui_agents.s2.agents.agent_s import GraphSearchAgent
import platform

current_platform = "ubuntu"  # "macos"

grounding_agent = OSWorldACI(
    platform=current_platform,
    endpoint_provider="huggingface",
    endpoint_url="<endpoint_url>/v1/",  # Check this for more help: https://huggingface.co/docs/inference-endpoints/guides/test_endpoint
)

engine_params = {
    "engine_type": "openai",
    "model": "gpt-4o",
}

agent = GraphSearchAgent(
  engine_params,
  grounding_agent,
  platform="ubuntu",  # "macos"
  action_space="pyautogui",
  observation_type="mixed",
  search_engine="Perplexica"
)

# Get screenshot.
screenshot = pyautogui.screenshot()
buffered = io.BytesIO() 
screenshot.save(buffered, format="PNG")
screenshot_bytes = buffered.getvalue()

obs = {
  "screenshot": screenshot_bytes,
}

instruction = "Close VS Code"
info, action = agent.predict(instruction=instruction, observation=obs)

exec(action[0])
```

Refer to `gui_agents/s2/cli_app.py` for more details on how the inference loop works.

### OSWorld

To deploy Agent S2 in OSWorld, follow the [OSWorld Deployment instructions](OSWorld.md).

## 🤝 Acknowledgements

We extend our sincere thanks to Tianbao Xie for developing OSWorld. We also appreciate the engaging discussions with Tianbao Xie, Yujia Qin, and Shihao Liang regarding UI-TARS. Their insights and collaborative spirit helped us navigate challenges and refine our work!

## 💬 Citation
```
@misc{agashe2024agentsopenagentic,
      title={Agent S: An Open Agentic Framework that Uses Computers Like a Human}, 
      author={Saaket Agashe and Jiuzhou Han and Shuyu Gan and Jiachen Yang and Ang Li and Xin Eric Wang},
      year={2024},
      eprint={2410.08164},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.08164}, 
}
```

