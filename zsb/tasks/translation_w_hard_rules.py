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
    description: str = (
        "Generate a translation instruction for a source text with hard rules to follow."
    )
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

To demonstrate the expected output, also provide a reference translation following the requested requirements at the end.

IMPORTANT: Your response should be structured as follows:

<START OF PROMPT>
[INSERT ONLY THE PROMPT HERE COMBINING SOURCE, RULES, AND AN INSTRUCTION. REMIND THE MODEL TO RETURN ONLY THE TRANSLATION. NOTHING ELSE.]
<END OF PROMPT>

<START OF REFERENCE>
[INSERT ONLY THE REFERENCE TRANSLATION. NOTHING ELSE.]
<END OF REFERENCE>

ABIDE STRICTLY BY THE REQUESTED FORMAT."""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating translation quality. You will be presented with:

- A text, prompting a model for a translation of a source
- A translation to evaluate

Rate the translation on a scale of 1-6 based on these key criteria:

- Accuracy and fidelity to source
- Grammar and language correctness
- Natural flow and readability
- Terminology consistency
- Completeness of translation
- Technical precision

Scoring Rubric:
6 - Outstanding

- Perfect accuracy with source meaning
- Flawless grammar and language use
- Reads completely naturally in target language
- Consistent and precise terminology
- Complete translation with no omissions
- Excellent technical accuracy

5 - Excellent

- Very accurate rendering of source
- Strong grammar with minimal issues
- Natural-sounding translation
- Good terminology consistency
- Nearly complete coverage
- Strong technical accuracy

4 - Good

- Generally accurate translation
- Mostly correct grammar
- Readable with some awkward passages
- Generally consistent terminology
- Minor omissions only
- Adequate technical accuracy

3 - Fair

- Some accuracy issues
- Notable grammar problems
- Often unnatural phrasing
- Inconsistent terminology
- Several omissions
- Technical inaccuracies present

2 - Poor

- Significant accuracy issues
- Frequent grammar errors
- Unnatural throughout
- Poor terminology consistency
- Major omissions
- Many technical errors

1 - Inadequate

- Fails to convey source meaning
- Severe grammar issues
- Incomprehensible in target language
- No terminology consistency
- Incomplete translation
- Technical meaning lost

Provide your evaluation in this JSON format:

{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

<START OF SOURCE TEXT>
${prompt}
<END OF SOURCE TEXT>

<START OF TRANSLATION>
${answer}
<END OF TRANSLATION>
                                    
You may proceed to evaluate the translation. Focus on evaluating the quality of the translation, not on whether it follows the rules in the prompt. Ensure the output is valid JSON, without additional formatting or explanations."""
            ),
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
        part_strings = [
            "PROMPT",
            "REFERENCE",
        ]
        output_dict = {"prompt": None, "reference": None}
        for part, key in zip(part_strings, output_dict.keys()):
            part_re_match = re.search(
                rf"<START OF {part}>\n(.*?)\n<END OF {part}>", output, re.DOTALL
            )
            if part_re_match is None:
                return False
            else:
                output_dict[key] = part_re_match.group(1).strip()
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


@dataclass
class TranslationWHardRules_EN_ESLA(TranslationWHardRules):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_en_esla"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Spanish (Latin America)"]


@dataclass
class TranslationWHardRules_EN_ZH(TranslationWHardRules):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_en_zh"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Chinese (Simplified)"]
