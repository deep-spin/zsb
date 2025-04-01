import pandas as pd
from datasets import load_dataset
from jsonargparse import CLI

from zsb.mbr.utils import run_mbr_matrix
from zsb.models import instantiate_model
from zsb.utils import read_lines, write_lines


def main(
    dataset_path: str,
    dataset_type: str,
    model_name: str,
    model_type: str,
    lang: str,
    n_candidates: int,
    mbr_model_type: str,
    bigger_is_better: bool,
    candidates_path: str,
    best_candidates_output_path: str,
    utilities_output_path: str,
    full_scores_path: str,
    model_args: dict = {
        "proper_model_args": {},
        "sampling_params": {"temperature": 0, "max_tokens": 8192},
    },
):
    # load data
    if dataset_type == "jsonl":
        prompts = pd.read_json(dataset_path, lines=True)["prompt"]
    elif dataset_type == "hf":
        prompts = load_dataset(dataset_path, lang)["test"]["prompt"]
    candidates = read_lines(candidates_path)
    # load model
    model_args["proper_model_args"]["model"] = model_name
    model = instantiate_model(model_type, model_args)
    # generate answers
    best_candidates, best_utilities, expected_utilities = run_mbr_matrix(
        candidates,
        prompts,
        n_candidates,
        model,
        mbr_model_type,
        bigger_is_better,
    )
    # save artifacts
    write_lines(best_candidates_output_path, best_candidates, escape_newline=True)
    write_lines(utilities_output_path, [str(u) for u in best_utilities])
    write_lines(full_scores_path, [str(u) for u in expected_utilities])

    return best_candidates, best_utilities, expected_utilities


if __name__ == "__main__":
    CLI([main], as_positional=False)
