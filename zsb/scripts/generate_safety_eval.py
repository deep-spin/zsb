import pandas as pd
from jsonargparse import CLI

from zsb.models import instantiate_model
from zsb.utils import SAFETY_TEMPALTE, parse_safety_output


def main(
    model_name: str,
    model_type: str,
    data_path: str,
    output_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load answers and task
    data = pd.read_json(data_path, lines=True)
    # load judge
    model_args["proper_model_args"]["model"] = model_name
    model_args["system_prompt"] = None
    model = instantiate_model(model_type, model_args)
    prompts = [
        SAFETY_TEMPALTE.substitute(prompt=prompt) for prompt in data["prompt"].tolist()
    ]
    # generate judgements
    judgements = model.batch_generate(prompts)
    parsed_results, parsed_feedbacks = zip(
        *[parse_safety_output(j) for j in judgements]
    )
    final_data = data.to_dict(orient="list")
    final_data["judgement"] = list(parsed_results)
    final_data["feedback"] = list(parsed_feedbacks)

    # save
    pd.DataFrame.from_dict(final_data).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return final_data


if __name__ == "__main__":
    CLI([main], as_positional=False)
