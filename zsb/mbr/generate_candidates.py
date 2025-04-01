import pandas as pd
from datasets import load_dataset
from jsonargparse import CLI

from zsb.models import instantiate_model
from zsb.utils import write_lines


def main(
    dataset_path: str,
    dataset_type: str,
    model_name: str,
    model_type: str,
    lang: str,
    n_candidates: int,
    output_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load queries
    if dataset_type == "jsonl":
        queries = pd.read_json(dataset_path, lines=True)["prompt"]
    elif dataset_type == "hf":
        queries = load_dataset(dataset_path, lang)["test"]["prompt"]
    queries = [q for q in queries for _ in range(n_candidates)]
    # load model
    model_args["proper_model_args"]["model"] = model_name
    model = instantiate_model(model_type, model_args)
    # generate answers
    answers = model.batch_generate(queries)
    write_lines(output_path, answers, escape_newline=True)

    return queries


if __name__ == "__main__":
    CLI([main], as_positional=False)
