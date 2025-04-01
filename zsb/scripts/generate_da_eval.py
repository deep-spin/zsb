import pandas as pd
from jsonargparse import CLI

from zsb.models import instantiate_model
from zsb.tasks import available_tasks


def main(
    task: str,
    model_name: str,
    model_type: str,
    answers_path: str,
    output_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
    use_ref: bool = False,
):
    # load answers and task
    answers = pd.read_json(answers_path, lines=True)
    task_obj = available_tasks[task]
    if use_ref:
        judge_prompt = task_obj.ref_da_judge_prompt["user_prompt"]
    else:
        judge_prompt = task_obj.da_judge_prompt["user_prompt"]
    system_prompt = task_obj.da_judge_prompt["system_prompt"]
    # load judge
    model_args["proper_model_args"]["model"] = model_name
    model_args["system_prompt"] = system_prompt
    model = instantiate_model(model_type, model_args)
    prompts = [
        judge_prompt.substitute(prompt=prompt, answer=answer, reference=reference)
        for prompt, answer, reference in zip(
            answers["prompt"].tolist(),
            answers["answer"].tolist(),
            answers["reference"].tolist(),
        )
    ]
    # generate judgements
    judgements = model.batch_generate(prompts)
    parsed_results, parsed_feedbacks = zip(
        *[task_obj.parse_da_prompt_output(j) for j in judgements]
    )
    final_data = answers.to_dict(orient="list")
    final_data["judgement"] = list(parsed_results)
    final_data["feedback"] = list(parsed_feedbacks)

    # save
    pd.DataFrame.from_dict(final_data).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return final_data


if __name__ == "__main__":
    CLI([main], as_positional=False)
