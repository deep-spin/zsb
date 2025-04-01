import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import AUDIENCES, LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.base import Task


@dataclass
class EndToEndMTEval(Task):
    name: str = "end_to_end_mt_eval"
    description: str = "End to End MT Evaluation with Examples."
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
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are a multilingual content creator and translation expert. Your task is to generate a comprehensive translation exercise package based on the given attributes. Follow these instructions carefully:

1. Review the following input variables:
- Source language: ${source_language}
- Target language: ${target_language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Source Length: ${source_length}
- Audience: ${audience}
- Style: ${style}

2. Generate a source text:
Create an original text in the source language, adhering to the specified topic, subtopic, and length. The text should be coherent, informative, and suitable for translation.

3. Create a translation instruction:
Formulate a clear and specific instruction for translating the source text, taking into account the given attributes. The instruction should guide the translator on how to approach the translation task.

4. Generate a reference translation:
Produce a high-quality, fluent translation of the source text in the target language. This translation should serve as a reference for evaluating other translations.

5. Develop scoring rubrics:
Create one to three scoring factors to evaluate translations. These rubrics should be in English, clear, specific, and relevant to the translation task.

6. Generate example translations with scores and feedback:
Create five example translations of varying quality, ranging from score 1 (worst) to score 5 (best). The score 5 translation example should be different from the reference. 
For each example:
- Give a description in English of what the score represents.
- Provide the translation in the target language.
- Give comprehensive feedback in English on why the translation received the score, referencing the scoring rubrics. Be precise about the issues in the translation; avoid being vague. Phrase the feedback in various ways, avoiding repeating the same phrasing.

Format your output as follows:

<START OF SOURCE>
[INSERT THE SOURCE TEXT HERE]
<END OF SOURCE>

<START OF TRANSLATION INSTRUCTION>
[INSERT THE TRANSLATION INSTRUCTION HERE]
<END OF TRANSLATION INSTRUCTION>

<START OF REFERENCE TRANSLATION>
[INSERT THE REFERENCE TRANSLATION HERE]
<END OF REFERENCE TRANSLATION>

<START OF SCORING RUBRICS>
[INSERT SCORING RUBRICS IN ENGLISH SEPARATED BY A ;]
<END OF SCORING RUBRICS>

<START OF SCORE 1 DESCRIPTION>
[INSERT SCORE 1 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 1 DESCRIPTION>

<START OF SCORE 1 TRANSLATION>
[INSERT SCORE 1 TRANSLATION HERE]
<END OF SCORE 1 TRANSLATION>

<START OF SCORE 1 TRANSLATION FEEDBACK>
[INSERT SCORE 1 FEEDBACK IN ENGLISH HERE]
<END OF SCORE 1 TRANSLATION FEEDBACK>

<START OF SCORE 2 DESCRIPTION>
[INSERT SCORE 2 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 2 DESCRIPTION>

<START OF SCORE 2 TRANSLATION>
[INSERT SCORE 2 TRANSLATION HERE]
<END OF SCORE 2 TRANSLATION>

<START OF SCORE 2 TRANSLATION FEEDBACK>
[INSERT SCORE 2 FEEDBACK IN ENGLISH HERE]
<END OF SCORE 2 TRANSLATION FEEDBACK>

<START OF SCORE 3 DESCRIPTION>
[INSERT SCORE 3 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 3 DESCRIPTION>

<START OF SCORE 3 TRANSLATION>
[INSERT SCORE 3 TRANSLATION HERE]
<END OF SCORE 3 TRANSLATION>

<START OF SCORE 3 TRANSLATION FEEDBACK>
[INSERT SCORE 3 FEEDBACK IN ENGLISH HERE]
<END OF SCORE 3 TRANSLATION FEEDBACK>

<START OF SCORE 4 DESCRIPTION>
[INSERT SCORE 4 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 4 DESCRIPTION>

<START OF SCORE 4 TRANSLATION>
[INSERT SCORE 4 TRANSLATION HERE]
<END OF SCORE 4 TRANSLATION>

<START OF SCORE 4 TRANSLATION FEEDBACK>
[INSERT SCORE 4 FEEDBACK IN ENGLISH HERE]
<END OF SCORE 4 TRANSLATION FEEDBACK>

<START OF SCORE 5 DESCRIPTION>
[INSERT SCORE 5 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 5 DESCRIPTION>

<START OF SCORE 5 TRANSLATION>
[INSERT SCORE 5 TRANSLATION HERE]
<END OF SCORE 5 TRANSLATION>

<START OF SCORE 5 TRANSLATION FEEDBACK>
[INSERT SCORE 5 FEEDBACK IN ENGLISH HERE]
<END OF SCORE 5 TRANSLATION FEEDBACK>

Ensure that your response is comprehensive, coherent, and follows all the instructions provided above. 
IMPORTANT: BAD TRANSLATIONS CAN BE THE SAME SIZE AS GOOD ONES PROVIDED THEY HAVE INACCURACIES IN OTHER ASPECTS.
IMPORTANT: ABIDE STRICTLY BY THE REQUESTED FORMAT AND GENERATED UNTIL THE END OF THE REQUESTED OUTPUT."""
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
        # parse first part of the output
        part_strings = [
            "SOURCE",
            "TRANSLATION INSTRUCTION",
            "REFERENCE TRANSLATION",
            "SCORING RUBRICS",
            "SCORE 1 DESCRIPTION",
            "SCORE 1 TRANSLATION",
            "SCORE 1 TRANSLATION FEEDBACK",
            "SCORE 2 DESCRIPTION",
            "SCORE 2 TRANSLATION",
            "SCORE 2 TRANSLATION FEEDBACK",
            "SCORE 3 DESCRIPTION",
            "SCORE 3 TRANSLATION",
            "SCORE 3 TRANSLATION FEEDBACK",
            "SCORE 4 DESCRIPTION",
            "SCORE 4 TRANSLATION",
            "SCORE 4 TRANSLATION FEEDBACK",
            "SCORE 5 DESCRIPTION",
            "SCORE 5 TRANSLATION",
            "SCORE 5 TRANSLATION FEEDBACK",
        ]
        output_dict = {
            "source": None,
            "translation_instruction": None,
            "reference": None,
            "scoring_rubrics": None,
            "score_1_description": None,
            "score_1_translation": None,
            "score_1_feedback": None,
            "score_2_description": None,
            "score_2_translation": None,
            "score_2_feedback": None,
            "score_3_description": None,
            "score_3_translation": None,
            "score_3_feedback": None,
            "score_4_description": None,
            "score_4_translation": None,
            "score_4_feedback": None,
            "score_5_description": None,
            "score_5_translation": None,
            "score_5_feedback": None,
        }
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
        pass

    @staticmethod
    def parse_relative_prompt_output(output: str, a_place: int) -> tuple[str, str]:
        pass


@dataclass
class EndToEndMTEvalEN_JA(EndToEndMTEval):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_en_ja"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Japanese"]


@dataclass
class EndToEndMTEvalEN_PTPT(EndToEndMTEval):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_en_ptpt"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["European Portuguese"]
