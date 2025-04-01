import pandas as pd
from datasets import load_dataset
from jsonargparse import CLI

from zsb.models import instantiate_model
from zsb.tasks.multilingual_general_purpose_chat import GeneralPurposeChatEnglish
from zsb.utils import read_lines


def main(
    dataset_path: str,
    dataset_type: str,
    model_name: str,
    model_type: str,
    lang: str,
    answers_path: str,
    output_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load answers and task
    if dataset_type == "jsonl":
        prompts = pd.read_json(dataset_path, lines=True)["prompt"]
    elif dataset_type == "hf":
        prompts = load_dataset(dataset_path, lang)["test"]["prompt"]
    answers = read_lines(answers_path, unescape_newline=True)
    assert len(prompts) == len(answers)
    task_obj = GeneralPurposeChatEnglish()
    judge_prompt = task_obj.da_judge_prompt["user_prompt"]
    system_prompt = task_obj.da_judge_prompt["system_prompt"]
    # load judge
    model_args["proper_model_args"]["model"] = model_name
    model_args["system_prompt"] = system_prompt
    model = instantiate_model(model_type, model_args)
    prompts = [
        judge_prompt.substitute(prompt=prompt, answer=answer)
        for prompt, answer in zip(prompts, answers)
    ]
    # generate judgements
    judgements = model.batch_generate(prompts)
    parsed_results, parsed_feedbacks = zip(
        *[task_obj.parse_da_prompt_output(j) for j in judgements]
    )
    final_data = pd.DataFrame()
    final_data["judgement"] = list(parsed_results)
    final_data["feedback"] = list(parsed_feedbacks)

    # save
    pd.DataFrame.from_dict(final_data).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return final_data


if __name__ == "__main__":
    CLI([main], as_positional=False)
