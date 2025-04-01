import random

import numpy as np

from zsb import utils
from zsb.models import Model


def data_to_instruction(cand: str, ref: str, src: str, mbr_model_type: str) -> str:
    if mbr_model_type == "prometheus":
        template = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{src}

###Response to evaluate:
{cand}

###Reference Answer (Score 5):
{ref}

###Score Rubrics:
[How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?]
Score 1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
Score 2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
Score 3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
Score 4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
Score 5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

###Feedback: """

    instruction = template.format(cand=cand, ref=ref, src=src)

    return instruction


def parse_scores(unparsed_string: str, mbr_model_type: str) -> int:
    if mbr_model_type == "prometheus":
        score = unparsed_string.split("[RESULT]")[-1].strip()

    try:
        score = int(score)
    except ValueError:
        print(
            f"Failed to parse in output.\nSetting score to random value between 1 and 5 instead."
        )
        score = random.randint(1, 5)

    return score


def prep_mbr_data(candidates, sources, n_candidates):
    # convert list of strings to list of lists where each list contains n_candidates strings
    grouped_candidates = [
        candidates[i : i + n_candidates]
        for i in range(0, len(candidates), n_candidates)
    ]
    # create a list of duplicated sources, candidates, and references
    cands = []
    refs = []
    dup_srcs = []
    for i, samples in enumerate(grouped_candidates):
        indices = list(range(n_candidates))
        for cand in samples:
            for ref_id in indices:
                cands.append(cand)
                refs.append(samples[ref_id])
                dup_srcs.append(sources[i])
    return cands, refs, dup_srcs


def run_mbr_matrix(
    candidates: list[str],
    sources: list[str],
    n_candidates: int,
    mbr_model: Model,
    mbr_model_type: str,
    bigger_is_better: bool,
):
    cands, refs, dup_srcs = prep_mbr_data(candidates, sources, n_candidates)

    # get metric for all instances (100 candidates * 100 references * N length)
    instructions = [
        data_to_instruction(cand, ref, src, mbr_model_type)
        for cand, ref, src in zip(cands, refs, dup_srcs)
    ]
    unparsed_scores = mbr_model.batch_generate(instructions)
    scores = [parse_scores(u, mbr_model_type) for u in unparsed_scores]

    # get expected utilities for each candidate (100 candidates * N length)
    expected_utilities = []
    for i in range(0, len(scores), n_candidates):
        expected_utilities.append(np.mean(scores[i : i + n_candidates]))

    # for each source, select the candidate with the highest expected utility (N length)
    if bigger_is_better:
        _argbest_fn = np.argmax
        _best_fn = max
    else:
        _argbest_fn = np.argmin
        _best_fn = min
    best_utilities = []
    best_candidates = []
    for i in range(0, len(expected_utilities), n_candidates):
        best_utilities.append(_best_fn(expected_utilities[i : i + n_candidates]))
        best_candidate_index = _argbest_fn(expected_utilities[i : i + n_candidates])
        best_candidates.append(candidates[i + best_candidate_index])

    return best_candidates, best_utilities, expected_utilities
