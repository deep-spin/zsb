import pandas as pd
from jsonargparse import CLI

from zsb.models import instantiate_model


def main(
    model_name: str,
    model_type: str,
    prompts_path: str,
    output_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load queries
    queries = pd.read_json(prompts_path, lines=True)
    # load model
    model_args["proper_model_args"]["model"] = model_name
    model = instantiate_model(model_type, model_args)
    # generate answers
    answers = model.batch_generate(queries["prompt"].tolist())
    queries["answer"] = answers
    # save prompts for later
    pd.DataFrame.from_dict(queries).to_json(
        output_path, lines=True, orient="records", force_ascii=False
    )

    return queries


if __name__ == "__main__":
    CLI([main], as_positional=False)
