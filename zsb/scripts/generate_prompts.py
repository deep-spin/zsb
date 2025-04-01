import random

import pandas as pd
import tqdm
from jsonargparse import CLI

from zsb.models import Model, instantiate_model
from zsb.tasks import available_tasks
from zsb.tasks.base import Task
from zsb.utils import get_all_possible_combinations


def generate_user_prompts_batched(
    task_obj: Task,
    combinations: list[dict[str, str]],
    model: Model,
    n_prompts: int,
) -> list[str]:
    system_prompt = task_obj.meta_prompt["system_prompt"]
    model.system_prompt = system_prompt
    generating = True
    failed_combinations = 0
    outputs = []
    remaining_combinatinos = n_prompts - len(outputs)
    progress_bar = tqdm.tqdm(total=n_prompts, desc="Generating prompts...")
    while generating:
        start_i = failed_combinations + len(outputs)
        end_i = start_i + remaining_combinatinos
        batch_combinations = combinations[start_i:end_i]
        batch_prompts = [
            task_obj.meta_prompt["user_prompt"].substitute(**c)
            for c in batch_combinations
        ]
        partial_outputs = model.batch_generate(batch_prompts)
        for i, o in enumerate(partial_outputs):
            parsed_output = task_obj.parse_meta_prompt_output(o)
            if not parsed_output:
                failed_combinations += 1
                continue
            else:
                parsed_output["metadata"] = batch_combinations[i]
                outputs.append(parsed_output)
                progress_bar.update(1)
        if len(outputs) == n_prompts:
            print(
                f"Generated {n_prompts} prompts, having failed {failed_combinations} combinations."
            )
            generating = False
            break
        else:
            remaining_combinatinos = n_prompts - len(outputs)
            print(
                f"Failed to parse output for {failed_combinations} combinations. Retrying {remaining_combinatinos} combinations."
            )

    return outputs


def generate_user_prompts_unbatched(
    task_obj: Task,
    combinations: list[dict[str, str]],
    model: Model,
    n_prompts: int,
) -> list[str]:
    outputs = []
    generating = True
    # create a progress bar to update for later
    i = 0
    progress_bar = tqdm.tqdm(total=n_prompts, desc="Generating prompts...")
    # CHANGE TO SAMPLE AT EACH STEP OF FOR LOOP TO REPEAT IF FAILS
    while generating:
        combination = combinations[i]
        system_prompt = task_obj.meta_prompt["system_prompt"]
        model.system_prompt = system_prompt
        user_prompt = task_obj.meta_prompt["user_prompt"].substitute(**combination)
        model_output = model.generate(user_prompt)
        parsed_output = task_obj.parse_meta_prompt_output(model_output)
        if not parsed_output:
            print(f"Failed to parse output for query. Retrying another combination.")
            continue
        else:
            parsed_output["metadata"] = combination
            outputs.append(parsed_output)
            progress_bar.update(1)
            i += 1
        if len(outputs) == n_prompts:
            generating = False
            break
    return outputs


def main(
    task: str,
    n_prompts: int,
    model_name: str,
    model_type: str,
    output_path: str,
    seed: int = 124,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    random.seed(seed)
    # Instantiate task, generate all possible combinations of attributes, and shuffle.
    print(f"Generating task combinations for {task}.")
    task_obj = available_tasks[task]
    task_combinations = get_all_possible_combinations(task_obj.task_attributes)
    print(f"Generated {len(task_combinations)} unique task attribute combinations.")
    # Instantiate model that will be used to generate user prompts.
    model_args["proper_model_args"]["model"] = model_name
    model = instantiate_model(model_type, model_args)
    if model.model_type() == "vllm":
        random.shuffle(task_combinations)
        simulated_user_prompts = generate_user_prompts_batched(
            task_obj, task_combinations, model, n_prompts
        )
    else:
        shuffled_combinations = random.sample(task_combinations, len(task_combinations))
        simulated_user_prompts = generate_user_prompts_unbatched(
            task_obj, shuffled_combinations, model, n_prompts
        )
    # save prompts for later
    pd.DataFrame.from_dict(simulated_user_prompts).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return simulated_user_prompts


if __name__ == "__main__":
    CLI([main], as_positional=False)
