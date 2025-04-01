import random

import pandas as pd
import tqdm
from datasets import load_dataset
from jsonargparse import CLI
from PIL.Image import Image

from zsb.models import Model, instantiate_model
from zsb.tasks import available_tasks
from zsb.tasks.base import Task
from zsb.utils import load_image


def generate_user_prompts_batched(
    task_obj: Task,
    images: list[Image],
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
        batch_image_strs = images[start_i:end_i]
        batch_insts = [
            task_obj.meta_prompt["user_prompt"].substitute()
            for _ in range(len(batch_image_strs))
        ]
        partial_outputs = model.batch_generate(batch_insts, batch_image_strs)
        for i, o in enumerate(partial_outputs):
            parsed_output = task_obj.parse_meta_prompt_output(o)
            if not parsed_output:
                failed_combinations += 1
                continue
            else:
                parsed_output["metadata"] = {"image_str": batch_image_strs[i]}
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
    image_strs: list[str],
    model: Model,
    n_prompts: int,
) -> list[str]:
    system_prompt = task_obj.meta_prompt["system_prompt"]
    model.system_prompt = system_prompt
    outputs = []
    generating = True
    # create a progress bar to update for later
    i = 0
    progress_bar = tqdm.tqdm(total=n_prompts, desc="Generating prompts...")
    # CHANGE TO SAMPLE AT EACH STEP OF FOR LOOP TO REPEAT IF FAILS
    while generating:
        image_str = image_strs[i]
        user_prompt = task_obj.meta_prompt["user_prompt"].substitute()
        model_output = model.generate(user_prompt, image_str)
        parsed_output = task_obj.parse_meta_prompt_output(model_output)
        if not parsed_output:
            print(f"Failed to parse output for query. Retrying another combination.")
            continue
        else:
            parsed_output["metadata"] = {"image_str": image_str}
            outputs.append(parsed_output)
            progress_bar.update(1)
            i += 1
        if len(outputs) == n_prompts:
            generating = False
            break
    return outputs


def main(
    task: str,
    dataset_path: str,
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
    dataset = load_dataset(dataset_path, split="validation")  # .shuffle(seed=seed)
    images: list[str] = [load_image(i) for i in dataset["image"]]
    # instantiate task object
    task_obj = available_tasks[task]
    # Instantiate model that will be used to generate user prompts.
    model_args["proper_model_args"]["model"] = model_name
    model = instantiate_model(model_type, model_args)
    if model.model_type() == "vllm":
        simulated_user_prompts = generate_user_prompts_batched(
            task_obj, images, model, n_prompts
        )
    else:
        simulated_user_prompts = generate_user_prompts_unbatched(
            task_obj, images, model, n_prompts
        )
    # save prompts for later
    pd.DataFrame.from_dict(simulated_user_prompts).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return simulated_user_prompts


if __name__ == "__main__":
    CLI([main], as_positional=False)
