import itertools
import json
import random
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import AUDIENCES, LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.base import Task


@dataclass
class TranslationWHardRules(Task):
    name: str = "translation_w_hard_rules"
    description: str = "Translation with some rules to follow."
    task_attributes: dict[str, list[str] | dict[str, str | Callable]] = field(
        default_factory=lambda: {
            "source_language": [""],
            "target_language": [""],
            "topic": TOPICS,
            "subtopic": {
                "_depends_on": "topic",
                "callable": lambda topic: SUBTOPICS[topic],
            },
            "style": STYLES,
            "audience": AUDIENCES,
            "source_length": LENGTHS,
            "n_rules": [2, 3, 4],
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """As an expert prompt engineer, create a detailed prompt for a language model to perform the following task: translation of a source text, given a set of ${n_rules} rules. The source text should abide by the followin parameters:
- Source language: ${source_language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Style: ${style}
- Source length: ${source_length}

The translation should be in ${target_language}, and your generated prompt must specify a set of ${n_rules} rules. 

IMPORTANT: These rules must be objectively verifiable and should be clearly stated in the prompt. The language model should be instructed to follow these rules when translating the source text. An example of a verifiable rule is "Convert dates to the format DD/MM/YYYY."; an example of an unverifiable rule is "Make the translation sound more professional.". Keep in mind that the rules should make sense in the context of the source text and the target language.

IMPORTANT: Make sure that the source you create has elements that correspond to the rules you set.

The prompt should be structured as follows:
[INSERT THE DETAILED PROMPT HERE]
[INSERT THE RULES HERE ENCLOSED BY <START OF RULES> <END OF RULES>, AND LISTED WITH A "-".]
[INSERT THE SOURCE TEXT TO BE TRANSLTED HERE ENCLOSED BY <START OF SOURCE> <END OF SOURCE>.]
[REMIND THE MODEL TO RETURN ONLY THE TRANSLATION.]

To demonstrate the expected output, also provide a reference translation following the requested requirements at the end.

IMPORTANT: Your response should be structured as follows:

<START OF PROMPT>
[INSERT ONLY THE PROMPT HERE. NOTHING ELSE.]
<END OF PROMPT>

<START OF REFERENCE>
[INSERT ONLY THE REFERENCE TRANSLATION. NOTHING ELSE.]
<END OF REFERENCE>"""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(""""""),
        }
    )
    relative_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(""""""),
        }
    )

    @staticmethod
    def parse_meta_prompt_output(output: str) -> dict[str, str] | bool:
        output_dict = {"PROMPT": None, "SOURCE": None, "REFERENCE": None, "RULES": None}
        for part in output_dict.keys():
            part_re_match = re.search(
                rf"<START OF {part}>\n(.*?)\n<END OF {part}>", output, re.DOTALL
            )
            if part_re_match is None:
                return False
            else:
                output_dict[part] = part_re_match.group(1).strip()
        return output_dict

    @staticmethod
    def parse_da_prompt_output(output: str) -> tuple[int, str]:
        try:
            dict_judgement = json.loads(output)
            judgement = int(dict_judgement["result"])
        except:
            judgement = 1
        # get feedback
        try:
            dict_judgement = json.loads(output)
            feedback = dict_judgement["feedback"]
        except:
            feedback = None
        return judgement, feedback

    @staticmethod
    def parse_relative_prompt_output(output: str, a_place: int) -> tuple[str, str]:
        # get result
        try:
            dict_judgement = json.loads(output)
            judgement = dict_judgement["result"]
        except:
            judgement = random.choice(["A", "B"])
        # get feedback
        try:
            dict_judgement = json.loads(output)
            feedback = dict_judgement["feedback"]
        except:
            feedback = None
        # switch up if necessary
        if a_place == 1:
            if judgement == "A":
                judgement = "B"
            else:
                judgement = "A"
        return judgement, feedback


@dataclass
class TranslationWHardRules_EN_PTPT(TranslationWHardRules):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_en_ptpt"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["European Portuguese"]
