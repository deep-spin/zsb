import random

import pandas as pd
from jsonargparse import CLI

from zsb.models import instantiate_model
from zsb.tasks import available_tasks


def main(
    task: str,
    model_name: str,
    model_type: str,
    answers_path_A: str,
    answers_path_B: str,
    output_path: str,
    random_seed: int = 42,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load answers and task object
    answers_A = pd.read_json(answers_path_A, lines=True)
    answers_B = pd.read_json(answers_path_B, lines=True)
    task_obj = available_tasks[task]
    judge_prompt = task_obj.relative_judge_prompt["user_prompt"]
    system_prompt = task_obj.relative_judge_prompt["system_prompt"]
    # load model
    model_args["proper_model_args"]["model"] = model_name
    model_args["system_prompt"] = system_prompt
    model = instantiate_model(model_type, model_args)
    # shuffle orders of A and B for each query (models may be biased to prefer one of them)
    prompts = []
    a_places = []
    for (i, q_a), (j, q_b) in zip(answers_A.iterrows(), answers_B.iterrows()):
        random.seed(random_seed * i)
        order = [0, 1]
        random.shuffle(order)
        meta_user_prompt = q_a["prompt"]
        a_answer = q_a["answer"]
        b_answer = q_b["answer"]
        answers = [a_answer, b_answer]
        formatted_prompt = judge_prompt.substitute(
            prompt=meta_user_prompt,
            answer_A=answers[order[0]],
            answer_B=answers[order[1]],
        )
        prompts.append(formatted_prompt)
        a_places.append(order[0])
    # generate answers
    judgements = model.batch_generate(prompts)
    parsed_results = []
    parsed_feedbacks = []
    for i, (j, a_place) in enumerate(zip(judgements, a_places)):
        # get result
        result, feedback = task_obj.parse_relative_prompt_output(j, a_place)
        parsed_results.append(result)
        parsed_feedbacks.append(feedback)
    # save data
    final_data = answers_A.to_dict(orient="list")
    final_data["answer_A"] = answers_A["answer"].tolist()
    final_data["answer_B"] = answers_B["answer"].tolist()
    final_data.pop("answer")
    final_data["judgement"] = parsed_results
    final_data["feedback"] = parsed_feedbacks
    final_data["real_a_place"] = a_places

    # save
    pd.DataFrame.from_dict(final_data).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return final_data


if __name__ == "__main__":
    CLI([main], as_positional=False)
