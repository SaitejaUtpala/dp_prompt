## EMNLP 2023 Findings: Locally Differentially Private Document Generation Using Zero Shot Prompting [Paper Link](https://arxiv.org/abs/2310.16111)


<img src="dp_prompt.png" alt="Image Alt Text" width="50%">




### Overview

This is the repository for the paper "Locally Differentially Private Document Generation Using Zero Shot Prompting," presented at EMNLP 2023 Findings. This paper introduces a method for generating differentially private text documents using zero-shot prompting techniques.

### Repository Contents


- **attacks**: Directory containing Python scripts for attacks.

- **data**: Directory for storing the data related to the project, including clean data and ChatGPT paraphrased data.

- **mechanisms**: Directory for Python scripts related to mechanisms for document, sentence, and word-level tasks.

- **requirements.txt**: File listing the required packages and dependencies for the project.

- [**dp_prompt.ipynb**](dp_prompt.ipynb): Demo notebook showing an experiment snapshot comparing DP-Prompt vs Madlib.


### Prerequisites

- Python 3.9+
- Dependencies listed in `requirements.txt`
- A suitable environment or container (e.g., virtual environment)

### Installation

To set up the project on your local machine, follow these steps:


1. git clone git@github.com:SaitejaUtpala/dp_prompt.git
2. cd dp_prompt
3. pip install -r requirements.txt


### Demo Notebook

Demo notebook showing experiment snapshot comparing DP-Prompt vs Madlib  [clicking here](dp_prompt.ipynb).