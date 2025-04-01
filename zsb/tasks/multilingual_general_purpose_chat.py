import json
import random
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import (
    AUDIENCES,
    DIFFICULTIES,
    LENGTHS,
    STYLES,
    SUBTOPICS,
    TOPICS,
    WRITING_PROFICIENCIES,
)
from zsb.tasks.base import Task


@dataclass
class GeneralPurposeChat(Task):
    name: str = "general_purpose_chat"
    description: str = "General capabilities in some language."
    task_attributes: dict[str, list[str] | dict[str, str | Callable]] = field(
        default_factory=lambda: {
            "language": [""],
            "topic": TOPICS,
            "subtopic": {
                "_depends_on": "topic",
                "callable": lambda topic: SUBTOPICS[topic],
            },
            "difficulty": DIFFICULTIES,
            "style": STYLES,
            "writer": AUDIENCES,
            "writing_proficiency": WRITING_PROFICIENCIES,
            "length": LENGTHS,
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are tasked with creating a diverse and engaging prompt for a chatbot arena. This prompt will be used to test and compare the capabilities of different language models. Your goal is to generate a question or prompt that will challenge these models and showcase their strengths or weaknesses. Also, generate a reference answer to your prompt that will serve as a benchmark for evaluating the models' responses.

Here are the input variables you will use to craft your prompt:
- Language: ${language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Difficulty: ${difficulty}
- Style: ${style}
- Writer: ${writer}
- Writing proficiency: ${writing_proficiency}
- Prompt length: ${length}

Guidelines for creating the prompt:

1. Abide strictly by the input variables provided.

2. Ensure that your prompt is open-ended enough to allow for varied and interesting responses from different language models.

3. Avoid prompts that are overly specific to a particular AI model's capabilities or training data.

4. Create a prompt that is engaging and thought-provoking, encouraging creative or analytical thinking.

Generate output in the following format:

<START OF PROMPT>
[Your generated prompt here. IMPORTANT: include only the prompt.]
<END OF PROMPT>

<START OF REFERENCE>
[A reference answer to your prompt. IMPORTANT: include only the reference answer.]
<END OF REFERENCE>

Ensure the generated prompt is in the requested language. Remember to abide strictly by the provided input variables and the requested format."""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating response quality for prompts on a variety of topics. You will be presented with:

- An original prompt
- A response to evaluate

Rate the response on a scale of 1-6 based on these key criteria:

- Accuracy and factual correctness
- Comprehensiveness of the answer
- Clarity and organization
- Relevance to the prompt
- Supporting evidence/examples (if applicable)
- Objectivity and balance (if applicable)

Scoring Rubric:
6 - Outstanding

- Exceptionally accurate and well-researched information
- Comprehensive coverage with valuable additional context
- Perfectly structured and crystal clear explanation
- Directly addresses all aspects of the prompt
- Strong supporting evidence and examples
- Maintains perfect objectivity where needed

5 - Excellent

- Highly accurate information
- Thorough coverage of the topic
- Very clear and well-organized
- Closely aligned with the prompt
- Good supporting evidence
- Strong objectivity

4 - Good

- Generally accurate information
- Adequate coverage of main points
- Clear organization with minor issues
- Mostly relevant to the prompt
- Basic supporting evidence
- Reasonable objectivity

3 - Fair

- Some inaccuracies present
- Incomplete coverage
- Organization needs improvement
- Partially misses prompt intent
- Limited supporting evidence
- Some bias present

2 - Poor

- Multiple inaccuracies
- Superficial coverage
- Unclear organization
- Largely misses prompt intent
- Minimal supporting evidence
- Clear bias present

1 - Inadequate

- Fundamentally incorrect information
- Severely incomplete
- Disorganized or incomprehensible
- Fails to address prompt
- No supporting evidence
- Extremely biased or misleading

Provide your evaluation in this JSON format:

{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

Remember that a good response should be accurate, comprehensive, clear, and directly address the prompt while providing appropriate supporting evidence and maintaining objectivity where needed. Remember that a long response is not necessarily good. If the response is not in the language of the prompt, give it a score of 1.

<START OF PROMPT>
${prompt}
<END OF PROMPT>

<START OF ANSWER>
${answer}
<END OF ANSWER>

You may proceed to evaluate the response. Ensure the output is valid JSON, without additional formatting or explanations.
"""
            ),
        }
    )
    ref_da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating response quality for prompts on a variety of topics. You will be presented with:

- An original prompt
- A reference response.
- A response to evaluate

Given the reference, rate the response on a scale of 1-6 based on these key criteria:

- Accuracy and factual correctness
- Comprehensiveness of the answer
- Clarity and organization
- Relevance to the prompt
- Supporting evidence/examples (if applicable)
- Objectivity and balance (if applicable)

Scoring Rubric:
6 - Outstanding

- Exceptionally accurate and well-researched information
- Comprehensive coverage with valuable additional context
- Perfectly structured and crystal clear explanation
- Directly addresses all aspects of the prompt
- Strong supporting evidence and examples
- Maintains perfect objectivity where needed

5 - Excellent

- Highly accurate information
- Thorough coverage of the topic
- Very clear and well-organized
- Closely aligned with the prompt
- Good supporting evidence
- Strong objectivity

4 - Good

- Generally accurate information
- Adequate coverage of main points
- Clear organization with minor issues
- Mostly relevant to the prompt
- Basic supporting evidence
- Reasonable objectivity

3 - Fair

- Some inaccuracies present
- Incomplete coverage
- Organization needs improvement
- Partially misses prompt intent
- Limited supporting evidence
- Some bias present

2 - Poor

- Multiple inaccuracies
- Superficial coverage
- Unclear organization
- Largely misses prompt intent
- Minimal supporting evidence
- Clear bias present

1 - Inadequate

- Fundamentally incorrect information
- Severely incomplete
- Disorganized or incomprehensible
- Fails to address prompt
- No supporting evidence
- Extremely biased or misleading

Provide your evaluation in this JSON format:

{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

Remember that a good response should be accurate, comprehensive, clear, and directly address the prompt while providing appropriate supporting evidence and maintaining objectivity where needed. Remember that a long response is not necessarily good. If the response is not in the language of the prompt, give it a score of 1.

<START OF PROMPT>
${prompt}
<END OF PROMPT>

<START OF REFERENCE>
${reference}
<END OF REFERENCE>

<START OF ANSWER>
${answer}
<END OF ANSWER>

You may proceed to evaluate the response. Ensure the output is valid JSON, without additional formatting or explanations.
"""
            ),
        }
    )
    relative_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating response quality for prompts on a variety of topics. You will be presented with:

- An original prompt
- Two different responses (Response A and Response B)

Evaluate both responses based on these key criteria:

- Accuracy and factual correctness
- Comprehensiveness of the answer
- Clarity and organization
- Relevance to the prompt
- Supporting evidence/examples (if applicable)
- Objectivity and balance (if applicable)

Provide your evaluation in this JSON format:

{"feedback": "<write feedback about the strengths and weaknesses of each answer and compare them>", "result": "<write "A" or "B">"}

Remember that a good response should be accurate, comprehensive, clear, and directly address the prompt while providing appropriate supporting evidence and maintaining objectivity where needed. If one response is in a language that is different from the one in the prompt, automatically consider the other answer better.

<START OF PROMPT>
${prompt}
<END OF PROMPT>

<START OF ANSWER A>
${answer_A}
<END OF ANSWER A>

<START OF ANSWER B>
${answer_B}
<END OF ANSWER B>

You may proceed to evaluate the response. Ensure the output is valid JSON, without additional formatting or explanations.
"""
            ),
        }
    )

    @staticmethod
    def parse_meta_prompt_output(output: str) -> dict[str, str] | bool:
        part_strings = ["PROMPT", "REFERENCE"]
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
class GeneralPurposeChatEnglish(GeneralPurposeChat):
    def __post_init__(self):
        self.name = "general_purpose_chat_english"
        self.task_attributes["language"] = ["English"]


@dataclass
class GeneralPurposeChatChineseS(GeneralPurposeChat):
    def __post_init__(self):
        self.name = "general_purpose_chat_chinese_s"
        self.task_attributes["language"] = ["Chinese (Simplified)"]


@dataclass
class GeneralPurposeChatFrench(GeneralPurposeChat):
    def __post_init__(self):
        self.name = "general_purpose_chat_french"
        self.task_attributes["language"] = ["French"]


@dataclass
class GeneralPurposeChatKorean(GeneralPurposeChat):
    def __post_init__(self):
        self.name = "general_purpose_chat_korean"
        self.task_attributes["language"] = ["Korean"]
