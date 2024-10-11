<h1>
  <img src="images/agent_s.png" alt="Logo" style="vertical-align:middle" width="60"> Agent S: An Open Agentic Framework that Uses Computers Like a Human
</h1>

<p align="center">
  <a href="https://www.simular.ai/agent-s">🌐 [Website]</a>
  <a href="">📄 [Paper]</a>
</p>

<p align="center">
<a href=https://saa1605.github.io> Saaket Agashe</a>, <a href=https://jiuzhouh.github.io/ >Jiuzhou Han </a>, <a href=https://scholar.google.com/citations?user=nfRYJJsAAAAJ&hl=zh-CN>Shuyu Gan</a>, <a href=https://sites.google.com/view/jiachen-yang/>Jiachen Yang</a>, <a href=https://angli.ai/>Ang Li</a>, <a href=https://eric-xw.github.io/>Xin Eric Wang</a>

</p>

## 💡 Introduction

<p align="center">
    <img src="./images/teaser.png" width="800">
</p>

<p>
Agent S is a new agentic framework designed to enable computers to be used as intuitively as a human would. We introduce an Experience-Augmented Hierarchical Planning method. This method utilizes Online Web Knowledge for up-to-date information on frequently changing software and websites, along with Narrative Memory to leverage high-level experiences from past interactions. By breaking complex tasks into manageable subtasks and using Episodic Memory for step-by-step guidance, Agent S continuously refines its actions and learns from experience, achieving adaptable and effective task planning.
</p>

## 🎯 Results

<p align="center">
    <img src="./images/results.png" width="800">
    <br>
    Results of Successful Rate (%) on the OSWorld full test set of all 369 test examples using Image + Accessibility Tree input.
</p>


## 🛠️ Installation

Clone the Agent S Repository
```
git clone https://github.com/simular-ai/GUI-agent.git
```

We recommend using Anaconda or Miniconda to create a virtual environment and install the required dependencies. We used Python 3.9 for development and experiments.
```
conda create -n agent_s python=3.9
conda activate agent_s
```

Install the agent_s package and dependencies
```
pip install -e .
```

### Setup Retrieval from Web using Perplexica

1. Ensure Docker is installed and running on your system.
2. Clone the Perplexica repository:

   ```bash
   git clone https://github.com/ItzCrazyKns/Perplexica.git
   ```

3. After cloning, navigate to the directory containing the project files.

4. Rename the `sample.config.toml` file to `config.toml`. For Docker setups, you need only fill in the following fields:

   - `OPENAI`: Your OpenAI API key. **You only need to fill this if you wish to use OpenAI's models**.
   - `OLLAMA`: Your Ollama API URL. You should enter it as `http://host.docker.internal:PORT_NUMBER`. If you installed Ollama on port 11434, use `http://host.docker.internal:11434`. For other ports, adjust accordingly. **You need to fill this if you wish to use Ollama's models instead of OpenAI's**.
   - `GROQ`: Your Groq API key. **You only need to fill this if you wish to use Groq's hosted models**.
   - `ANTHROPIC`: Your Anthropic API key. **You only need to fill this if you wish to use Anthropic models**.

     **Note**: You can change these after starting Perplexica from the settings dialog.

   - `SIMILARITY_MEASURE`: The similarity measure to use (This is filled by default; you can leave it as is if you are unsure about it.)

5. Ensure you are in the directory containing the `docker-compose.yaml` file and execute:

   ```bash
   docker compose up -d
   ```
For a more detailed setup and usage guide, refer to the [Perplexica Repository](https://github.com/ItzCrazyKns/Perplexica.git)

### Setup Paddle-OCR Server

Run the ocr_server.py file code to use OCR-based bounding boxes.

```
cd agent_s
python ocr_server.py
```

Switch to a new terminal where you will run Agent S. Set the OCR_SERVER_ADDRESS environment variable as shown below. For a better experience, add the following line directly to your .bashrc (Linux), or .zshrc (MacOS) file.

```
export OCR_SERVER_ADDRESS=http://localhost:8000/ocr/
```

You can change the server address by editing the address in [agent_s/ocr_server.py](agent_s/ocr_server.py) file

## 🚀 Usage

### OSWorld

To deploy Agent S in OSWorld, follow the [OSWorld Deployment instructions](OSWorld.md).

### WindowsAgentArena

To deploy Agent S in WindowsAgentArena, follow the [WindowsAgentArena Deployment instructions](WindowsAgentArena.md).

### Run Locally on your Own Computer

We support running Agent S directly on your own system through [OpenACI](https://github.com/simular-ai/OpenACI). To run Agent S on your own system run: 
```
python openaci/cli_app.py --agent agent_s --model <MODEL>
```

This will show a user query prompt where you can enter your query and interact with Agent S. 

## 💬 Citation
```
@misc{AgentS,
  title={Agent S: An Open Agentic Framework that Uses Computers Like a Human},
  author={Saaket Agashe*, Jiuzhou Han*, Shuyu Gan, Jiachen Yang, Ang Li, Xin Eric Wang},
  year={2024},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

