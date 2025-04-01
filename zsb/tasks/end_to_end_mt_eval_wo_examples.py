import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import AUDIENCES, LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.base import Task


@dataclass
class EndToEndMTEvalWOExamples(Task):
    name: str = "end_to_end_mt_eval_wo_examples"
    description: str = "End to End MT Evaluation without Examples."
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

6. Generate descriptions of scores, ranging from score 1 (worst) to score 5 (best), which will later be used as guidelines to score translations. Give a description in English of what each score represents.

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

<START OF SCORE 2 DESCRIPTION>
[INSERT SCORE 2 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 2 DESCRIPTION>

<START OF SCORE 3 DESCRIPTION>
[INSERT SCORE 3 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 3 DESCRIPTION>

<START OF SCORE 4 DESCRIPTION>
[INSERT SCORE 4 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 4 DESCRIPTION>

<START OF SCORE 5 DESCRIPTION>
[INSERT SCORE 5 DESCRIPTION IN ENGLISH HERE]
<END OF SCORE 5 DESCRIPTION>

Ensure that your response is comprehensive, coherent, and follows all the instructions provided above.
IMPORTANT: ABIDE STRICTLY BY THE REQUESTED FORMAT AND KEEP GENERATING UNTIL THE END OF THE REQUESTED OUTPUT."""
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
            "SCORE 2 DESCRIPTION",
            "SCORE 3 DESCRIPTION",
            "SCORE 4 DESCRIPTION",
            "SCORE 5 DESCRIPTION",
        ]
        output_dict = {
            "source": None,
            "translation_instruction": None,
            "reference": None,
            "scoring_rubrics": None,
            "score_1_description": None,
            "score_2_description": None,
            "score_3_description": None,
            "score_4_description": None,
            "score_5_description": None,
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
class EndToEndMTEvalWOExamplesEN_JA(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_ja"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Japanese"]


@dataclass
class EndToEndMTEvalWOExamplesEN_DE(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_de"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["German"]


@dataclass
class EndToEndMTEvalWOExamplesEN_ES(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_es"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Spanish"]


@dataclass
class EndToEndMTEvalWOExamplesEN_RU(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_ru"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Russian"]


@dataclass
class EndToEndMTEvalWOExamplesEN_UK(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_uk"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Ukrainian"]


@dataclass
class EndToEndMTEvalWOExamplesEN_IS(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_is"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Icelandic"]


@dataclass
class EndToEndMTEvalWOExamplesEN_HI(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_hi"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Hindi"]


@dataclass
class EndToEndMTEvalWOExamplesEN_ZH(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_zh"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Chinese"]


@dataclass
class EndToEndMTEvalWOExamplesEN_CS(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_cs"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Czech"]


@dataclass
class EndToEndMTEvalWOExamplesEN_KO(EndToEndMTEvalWOExamples):
    def __post_init__(self):
        self.name = "end_to_end_mt_eval_wo_examples_en_ko"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Korean"]
