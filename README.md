# Zero-shot Benchmarking
This repository provides tools to easily create benchmarks and run automatic evaluation for language modelling tasks across domains and modalities using language models end-to-end. 
Check out our [paper]().

- [Installation](#installation)
- [Run an existing benchmark](#run-an-existing-benchmark)
- [Create a new benchmark](#create-a-new-benchmark)
  - [With a supported task](#with-a-supported-task)
  - [With a new task](#with-a-new-task)
- [Cite our work](#cite-our-work)

## Installation
```bash
git clone https://github.com/deep-spin/zsb.git
cd zsb
python -m venv venv
source venv/bin/activate
poetry install
```
> Tested with python==3.10 and poetry==1.6.1

## Run an existing benchmark
We provide benchmarks for general capabilities on 4 languages (English, French, Chinese, and Korean), translation, and vision language general capabilities in English (check the [data](https://github.com/deep-spin/zsb/data) folder).
All models supported in [litellm](https://github.com/BerriAI/litellm) (e.g., Open AI, Anthropic, Together) or [vllm](https://github.com/vllm-project/vllm) (e.g., most HF models) can be used for data creation, response generation, and evaluation.

For example, to get responses from `google/gemma-2-9b-it` for our English general capabilities benchmark, run:

```bash
python zsb/scripts/generate_answers.py --model_name google/gemma-2-9b-it --model_type vllm --prompts_path data/general_capabilities_english.jsonl --output_path example_answers.jsonl
```

Then, to evaluate with `claude-3-5-sonnet-20241022` as a judge, run:

```bash
python zsb/scripts/generate_da_eval.py --task general_purpose_chat_english --model_name claude-3-5-sonnet-20241022 --model_type litellm --answers_path example_answers.jsonl --output_path example_judgments.jsonl
```
> The scores for each instance will be in the `judgement` entry of each row in the output file.

## Create a new benchmark

### With a supported task

You can also create a new benchmark for the tasks we support. For example, to create 10 instances of Chinese general capabilities with `Qwen/Qwen2.5-72B-Instruct`, run:

```bash
python zsb/scripts/generate_prompts.py --task general_purpose_chat_korean --n_prompts 10 --model_name Qwen/Qwen2.5-72B-Instruct --model_type vllm --output_path example_dataset.jsonl --seed 42 --model_args "{'proper_model_args':{'tensor_parallel_size':4},'sampling_params':{'temperature':0,'max_tokens':8192}}"
```

To list all existing tasks, run:

```bash
python zsb/tasks/list.py
```

### With a new task

Check out our guide to create new tasks under the [tasks](https://github.com/deep-spin/zsb/zsb/tasks) folder.

## Cite our work

```bibtex
```