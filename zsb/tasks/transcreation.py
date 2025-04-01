import json
import random
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import AUDIENCES, LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.base import Task


@dataclass
class Transcreation(Task):
    name: str = "transcreation"
    description: str = (
        "Transcreation (translation + cultural adaptation) of a source text into a target language."
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
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """As an expert prompt engineer, create a detailed prompt for a language model to perform transcreation of a source text. The transcreation should be adapted for the following specifications:
- Source language: ${source_language}
- Target language: ${target_language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Style: ${style}
- Target audience: ${audience}
- Source length: ${source_length}

The prompt should:
1. Specify the target audience and style requirements.
2. Emphasize cultural adaptation and maintaining the original message's intent.
3. Include the source text to be transcreated. IMPORTANT: THE SOURCE TEXT SHOULD BE CONTRASTIVE WITH THE TRANSCREATION REQUIREMENTS.
4. Ask the model to return only the transcreated text.

The prompt should be structured as follows:
[INSERT THE DETAILED PROMPT HERE]
[INSERT THE SOURCE TEXT TO BE TRANSCREATED HERE ENCLOSED BY <START OF SOURCE> <END OF SOURCE>.]
[REMIND THE MODEL TO RETURN ONLY THE TRANSCREATION.]

To demonstrate the expected output, also provide a reference transcreation following the requested requirements at the end.

IMPORTANT: Your response should be structured as follows:

<START OF PROMPT>
[INSERT ONLY THE PROMPT HERE. NOTHING ELSE.]
<END OF PROMPT>

<START OF REFERENCE>
[INSERT ONLY THE REFERENCE TRANSCREATION. NOTHING ELSE.]
<END OF REFERENCE>"""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating transcreation quality. You will be presented with:

- An original prompt requesting transcreation
- A response to evaluate

Rate the response on a scale of 1-6 based on these key criteria:

- Cultural relevance and adaptation
- Preservation of the original message's intent
- Creativity and originality
- Emotional resonance with the target audience
- Language fluency and naturalness
- Brand voice consistency (if applicable)

Scoring Rubric:
6 - Outstanding
- Masterfully adapts content for target culture while perfectly preserving original intent
- Highly creative and original approach that resonates deeply with target audience
- Exceptional language fluency that feels native to target culture
- Flawlessly maintains brand voice while culturally adapting
- Goes above and beyond in cultural nuance and emotional impact

5 - Excellent
- Strong cultural adaptation with clear preservation of original message
- Creative solutions that effectively connect with target audience
- Very natural language use with cultural authenticity
- Strong brand voice consistency
- Notable attention to cultural nuances

4 - Good
- Adequate cultural adaptation while maintaining core message
- Some creative elements that generally work for target audience
- Generally natural language with occasional minor issues
- Mostly consistent brand voice
- Basic cultural considerations addressed

3 - Fair
- Partial cultural adaptation with some loss of original intent
- Limited creativity or generic solutions
- Language that sometimes feels unnatural
- Inconsistent brand voice
- Missing important cultural elements

2 - Poor
- Minimal cultural adaptation
- Lacks creativity and originality
- Frequently unnatural language
- Weak brand voice consistency
- Cultural disconnects present

1 - Inadequate
- Failed to adapt for target culture
- No creative input or consideration
- Unnatural language throughout
- Lost brand voice entirely
- Cultural inappropriateness or insensitivity

Provide your evaluation in this JSON format:

{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

Remember that good transcreation goes beyond literal translation to create culturally appropriate content that maintains the original message's impact while adapting it for the target audience.

<START OF USER PROMPT>
${prompt}
<END OF USER PROMPT>

<START OF ANSWER>
${answer}
<END OF ANSWER>

You may proceed to evaluate the responses. Ensure the output is valid JSON, without additional formatting or explanations."""
            ),
        }
    )
    relative_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating transcreation quality. You will be presented with:

- An original prompt requesting transcreation
- Two different responses (Response A and Response B)

Evaluate both responses based on these key criteria:

- Cultural relevance and adaptation
- Preservation of the original message's intent
- Creativity and originality
- Emotional resonance with the target audience
- Language fluency and naturalness
- Brand voice consistency (if applicable)

Provide your evaluation in this JSON format:

{"feedback": "<write feedback about the strengths and weaknesses of each answer and compare them>", "result": "<write "A" or "B">"}

Remember that good transcreation goes beyond literal translation to create culturally appropriate content that maintains the original message's impact while adapting it for the target audience.

<START OF USER PROMPT>
${prompt}
<END OF USER PROMPT>

<START OF ANSWER A>
${answer_A}
<END OF ANSWER A>

<START OF ANSWER B>
${answer_B}
<END OF ANSWER B>

You may proceed to evaluate the responses. Ensure the output is valid JSON, without additional formatting or explanations."""
            ),
        }
    )

    @staticmethod
    def parse_meta_prompt_output(output: str) -> dict[str, str] | bool:
        part_strings = ["PROMPT", "SOURCE", "REFERENCE"]
        output_dict = {"prompt": None, "source": None, "reference": None}
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
class TranscreationEN_PTPT(Transcreation):
    def __post_init__(self):
        self.name = "transcreation_en_ptpt"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["European Portuguese"]
