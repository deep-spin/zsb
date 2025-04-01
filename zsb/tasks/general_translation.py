import json
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.base import Task


@dataclass
class GeneralTranslation(Task):
    name: str = "general_translation"
    description: str = "Translation."
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
            "source_length": LENGTHS,
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are a multilingual content creator and translation expert. Your task is to generate a comprehensive translation exercise based on the given attributes. Follow these instructions carefully:

1. Review the following input variables:
- Source language: ${source_language}
- Target language: ${target_language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Source Length: ${source_length}
- Style: ${style}

2. Generate a source text:
Create an original text in the source language, adhering to the specified topic, subtopic, and length. The text should be coherent, informative, and suitable for translation.

3. Generate a reference translation:
Produce a high-quality, fluent translation of the source text in the target language. This translation should serve as a reference for evaluating other translations. IT IS CRUCIAL THAT THE REFERENCE TRANSLATION SOUNDS NATURAL IN THE TARGET LANGUAGE.

Format your output as follows:

<START OF SOURCE>
[INSERT THE SOURCE TEXT HERE]
<END OF SOURCE>

<START OF REFERENCE TRANSLATION>
[INSERT THE REFERENCE TRANSLATION HERE]
<END OF REFERENCE TRANSLATION>

Ensure that your response is comprehensive, coherent, and follows all the instructions provided above. Abide strictly by the requested format and generated until the end of the requested output."""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating translation quality. You will be presented with:

- An original text
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
                                    
You may proceed to evaluate the translation. Ensure the output is valid JSON, without additional formatting or explanations."""
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
        # parse first part of the output
        part_strings = [
            "SOURCE",
            "REFERENCE TRANSLATION",
        ]
        output_dict = {
            "source": None,
            "reference": None,
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
        pass


@dataclass
class GeneralTranslationEN_DE(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_de"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["German"]


@dataclass
class GeneralTranslationEN_ZH(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_zh"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Chinese"]


@dataclass
class GeneralTranslationCS_UK(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_cs_uk"
        self.task_attributes["source_language"] = ["Czech"]
        self.task_attributes["target_language"] = ["Ukrainian"]


@dataclass
class GeneralTranslationJA_ZH(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_ja_zh"
        self.task_attributes["source_language"] = ["Japanese"]
        self.task_attributes["target_language"] = ["Chinese"]


@dataclass
class GeneralTranslationEN_ES(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_es"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Spanish"]


@dataclass
class GeneralTranslationEN_CS(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_cs"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Czech"]


@dataclass
class GeneralTranslationEN_RU(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_ru"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Russian"]


@dataclass
class GeneralTranslationEN_UK(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_uk"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Ukrainian"]


@dataclass
class GeneralTranslationEN_HI(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_hi"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Hindi"]


@dataclass
class GeneralTranslationEN_JA(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_ja"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Japanese"]


@dataclass
class GeneralTranslationEN_IS(GeneralTranslation):
    def __post_init__(self):
        self.name = "general_translation_en_is"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Icelandic"]
