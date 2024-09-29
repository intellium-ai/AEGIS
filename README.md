# Intellium AEGIS Project
<div style="text-align:center">
<img src="./tower_defense.png" width="200" height="200" alt="Image description">
</div>

Welcome to [Intellium AI's](https://www.intellium.ai) AEGIS Project!

This repository contains an adapted version of the [ARCD PrimAITE](https://github.com/Autonomous-Resilient-Cyber-Defence/PrimAITE) Reinforcement Learning environment which includes the following changes / adaptations:
- Bug Fixes.
- Environment observability framework in Streamlit.
- Simplified integration of customised agents, in particular, Large Language Models.
- All practical work conducted as part of Task D2008d "Enhancing Situational Awareness of Language-based Blue Agents through Graph Neural Prompting" for the [Alan Turing Institute](https://www.turing.ac.uk).

## Prerequisites and Disclaimer:
- Ubuntu 22.04.4 LTS.
- A machine with 4 GPUs, preferably 4x NVidia RTX 3090 (24GB) cards.
- Ensure you have Python 3.10.12 installed.
- Ensure you have GPU hardware on your machine with CUDA support.


We developed using this configuration and therefore have been able to validate the execution of our code for this setup. While you do not have to use the same hardware, OS or Python version, we will not be able to offer support for setup or debug problems if we are unable to replicate issues that you may experience as a result.

## Setup Instructions:
- Clone this repository locally.
- Install the PrimAITE environment with additional AEGIS dependencies:
    - `python3 -m venv aegis`
    - `source aegis/bin/activate`
    - `pip install -e '.[dev]'`
    - Check if torch can interface with your GPU(s) using `import torch; print(torch.cuda.is_available())`.

## Working with CUDA
If you do not have the same number of GPUs, we advise you to reconfigure the DEFAULT_DEVICE_MAP we have setup in 
[llm.py](/src/primaite/agents/aegis/modules/llm.py) accordingly.

## Usage
We recommend you familiarise yourself with the original PrimAITE [README](https://github.com/Autonomous-Resilient-Cyber-Defence/PrimAITE/blob/dev/README.md) if you are not already familiar with the environment.


### DEMO Notebooks
We have provided some demonstration notebooks in [DEMO Notebooks](./DEMO%20Notebooks/) which can be used as a reference point for seeing how various components we have developed work. These include:
1. [PrimAITE Network Laydown Generation](/DEMO%20Notebooks/network_generation.ipynb).
2. [Use of Vanilla LLM Agent](/DEMO%20Notebooks/2_vanilla_llm_agent.ipynb).
3. [End-to-end training](/DEMO%20Notebooks/3_e2e_gllm_agent_training.ipynb) of the GLLM agent.
4. [Graph Large Language Model (GLLM) Pre-training](/DEMO%20Notebooks/4_pre_train_gllm.ipynb).
5. [Contrastive Learning](/DEMO%20Notebooks/5_contrastive_learning_training.ipynb) experimentation.
6. [A2C GNN Agent training](/DEMO%20Notebooks/6_train_gnn_agent.ipynb).
7. Benchmark [SB3 A2C Agent training](/DEMO%20Notebooks/7_train_sb3_agent.ipynb).
8. Plotting training logs for GLLM [E2E](/DEMO%20Notebooks/8_plot_e2e_training_logs.ipynb) and [pre-training](/DEMO%20Notebooks/8_plot_gllm_training_logs.ipynb) (for transparency).


### Module Reference
- Our implementation for the Vanilla LLM agent can be found in the agents folder [here](/src/primaite/agents/vanilla_llm/).

- The GNN agent can be found in the agents folder [here](/src/primaite/agents/gnn_agent.py).

- All AEGIS related works, such as the modules used in GLLM pre-training and e2e training can be found within the [aegis folder](/src/primaite/agents/aegis/).


## Observability Framework
To use the observability framework, you will need to follow these steps in a terminal from the repo main directory:
- `cd app/`
- `streamlit run main.py`
- Open the URL shown in the output in your browser.
## Notes
- For use of the Vanilla LLM agent, we added support for the Fireworks AI API to allow for plug and play. You will need to specify your API key in the LLMAgent initialization as requested [here](src/primaite/agents/vanilla_llm/agent.py).
- Some notebooks / scripts contain relative filepaths which will not work for you. You will need to adjust these accordingly. This includes:
    - Removing any references to checkpoints which are not in the repository and starting trainings from scratch.
    - Replacing references to training / evaluation datasets not included in the repository with new datasets which you generate yourself.

<b>We hope you enjoy.</b>