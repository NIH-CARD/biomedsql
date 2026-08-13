# BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases

[![arXiv](https://img.shields.io/badge/arXiv-2505.20321-b31b1b)](https://arxiv.org/abs/2505.20321)
[![Website](https://img.shields.io/badge/website-biomedsql-blue)](https://datatecnica.github.io/biomedbench-suite/biomedsql)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-BiomedSQL-yellow)](https://huggingface.co/datasets/NIH-CARD/BiomedSQL)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

Accepted as a conference paper at COLM 2026!

See the full paper on [arXiv](https://arxiv.org/abs/2505.20321).

![Alt text](assets/text-to-sql-workflow.png)

<p align="center"><em>Overview of text-to-SQL workflow used to evaluate LLMs on BiomedSQL.</em></p>

## Requirements

We use [uv](https://docs.astral.sh/uv/) for dependency management. To install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all dependencies:

```bash
uv sync
```

## Environment Setup

BiomedSQL requires the extensive use of both opened a closed source LLMs. The following services are needed to run the full set of experiments:

* AzureOpenAI (with endpoints for GPT-4o, GPT-o3-mini, and GPT-5.2)
* AzureAI (with an endpoint for Meta-Llama-405B-Instruct)
* Gemini (for access to Gemini-2.0-Flash and Gemini-3-Pro)
* OpenAI (for access to the general ```completions()``` API for use in the Schema Indexing interaction paradigm)
* Anthtropic (for access to Claude-3.7-Sonnet and Claude-4.5-Opus)
* HuggingFace (for access to gated Qwen-2.5-Coder-14B-Instruct, Qwen-2.5-Coder-32B-Instruct, and Llama-70B-Instruct repositories)

See ```config/sample.env``` for a complete list of specific information needed from each provider. Once complete, please move this file to ```config/.env``` for seamless use in the current experiment setup.

## Benchmark Dataset

Our benchmark dataset and associated database tabular data can be found on [HuggingFace](https://huggingface.co/datasets/NIH-CARD/BiomedSQL).

## BigQuery Database Creation

To create the BigQuery database from the tabular data hosted on HuggingFace, run:

```bash
uv run python create_database.py
```

## LLM Experiments

To run the isolated SQL generation experiments for BiomedSQL, run:

```bash
uv run python run_llm_experiments.py
```

> **Note:** If you do not have GPUs available, comment out any models with `provider: huggingface` under the `experiment_models` section of `config/llm_config.yaml` before running.

Currently we use the following open-source models and detail the following compute requirements to run our experiment pipeline as-is:
* meta-llama/Llama-3.1-70B-Instruct (three NVIDIA 80GB A100 GPUs)
* Qwen/Qwen2.5-Coder-32B-Instruct (two NVIDIA 80GB A100 GPUs)
* Qwen/Qwen2.5-Coder-14B-Instruct (two NVIDIA 80GB A100 GPUs)

We understand that GPU access may differ from user to user, so in order to run our experiments without the need for GPUs, please comment out any models specified with ```provider: huggingface``` under the ```experiment_models``` section of ```config/llm_config.yaml```.

## Interaction Paradigm Experiments

To run the interaction paradigm experiments for BiomedSQL, run:

```bash
uv run python run_interaction_experiments.py
```

## Generate Results

To generate figures and tables after the experiments are finished, run:

```bash
uv run python results.py
```

Tables will show up in ```results/``` and plots will show up in ```results/plots/```.

## Results

On BiomedSQL, Gemini-3-Pro is the top-performing base model. However, our custom-built text-to-SQL system (BMSQL) sees better performance when paired with GPT-o3-mini. Both fall short of expert baselines.

| Model Name             | Execution Accuracy    | Response Quality Rate |
| ---------------------- | --------------------- | --------------------- |
| Expert Baseline        |        90.0%          |          95.0%        |
| ---------------------- | --------------------- | --------------------- |
| Gemini-3-Pro           |        58.1%          |          81.8%        |
| BMSQL-GPT-o3-mini      |        62.6%          |          83.2%        |


## License and Contributing

This respository is under the PolyForm Noncommercial License (Version 1.0.0). 

To contribute, simply clone the repository and open a pull request. For any bugs or other fixes, feel free to open an issue.

## Relevant Citation
```
@article{koretsky2025biomedsql,
      title  = {BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases}, 
      author = {Mathew J. Koretsky and Maya Willey and Owen Bianchi and Chelsea X. Alvarado and Tanay Nayak and Nicole Kuznetsov and Sungwon Kim and Mike A. Nalls and Daniel Khashabi and Faraz Faghri},
      year   = {2026},
      eprint = {2505.20321},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL},
      url    = {https://arxiv.org/abs/2505.20321},
      code   = {https://github.com/NIH-CARD/biomedsql},
}
```
