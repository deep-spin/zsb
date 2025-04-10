import json
import random
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.tasks.base import Task


@dataclass
class MVLMGeneralPurposeChat(Task):
    name: str = "m_vlm_general_purpose_chat"
    description: str = "General purpose chat task for vision language models"
    task_attributes: dict[str, list[str] | dict[str, str | Callable]] = field(
        default_factory=lambda: {
            "language": [""],
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are tasked with creating a diverse and engaging prompt for a vision language model chatbot arena in ${language}, given an input image. This prompt will be used to test and compare the capabilities of different vision language models when they analyze images. Your goal is to generate a question or prompt that will challenge these models and showcase their strengths or weaknesses in image understanding, analysis, and reasoning. Also, generate a reference answer that will serve as a benchmark for evaluating the models' responses.

Guidelines for creating the prompt:
- Ensure the prompt is relevant to the attached image.
- Create a prompt that requires both visual analysis and reasoning capabilities.
- Do not describe the image in the prompt; let the model analyze it.

Ensure your prompt encourages the model to:
- Describe specific details in the image.
- Make connections between different elements.
- Draw conclusions or insights.
- Provide explanations or interpretations.
- Consider context and implications.


Avoid prompts that:
- Can be answered without looking at the image.
- Focus only on simple object detection.
- Are biased toward specific vision models.


Make prompts that test various vision-language capabilities such as:
- Spatial reasoning
- Object relationships
- Visual detail recognition
- Context understanding
- Abstract concept interpretation
- Temporal reasoning from static images
- Cultural or symbolic understanding



Generate output in the following format:

<START OF PROMPT>
[Your generated prompt in ${language} here. This should be a question or instruction that will be paired with the image for the model to analyze.]
<END OF PROMPT>

<START OF REFERENCE>
[A reference answer, and only the answer, to the generated prompt.]
<END OF REFERENCE>

Remember to abide strictly by the provided input variables and the requested format."""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating how well vision language models analyze and respond to image-based prompts. You will be presented with:

- The original prompt given to the model
- The model's response

Rate the response on a scale of 1-6 based on these key criteria:

- Visual Detail Recognition: How well does the response identify and describe specific elements in the image?
- Reasoning & Analysis: Does the response make meaningful connections and draw logical conclusions?
- Prompt Adherence: How well does the response address all aspects of the prompt?
- Accuracy: Is the information provided accurate based on what's visible in the image?
- Depth of Understanding: Does the response demonstrate deeper insight beyond surface-level observation?

Scoring Rubric:

6 - Outstanding

- Exceptional attention to visual details
- Sophisticated reasoning and connections
- Addresses all prompt aspects comprehensively
- Perfect accuracy in image interpretation
- Deep insights beyond obvious observations

5 - Excellent

- Strong attention to visual details
- Clear reasoning and meaningful connections
- Addresses most prompt aspects well
- High accuracy in image interpretation
- Good contextual understanding

4 - Good

- Adequate visual detail recognition
- Basic but logical reasoning
- Addresses main prompt points
- Generally accurate interpretation
- Some contextual understanding

3 - Fair

- Missing some key visual details
- Limited reasoning or connections
- Partially addresses the prompt
- Some inaccuracies in interpretation
- Superficial understanding

2 - Poor

- Misses many visual details
- Weak or illogical reasoning
- Largely misses prompt intent
- Multiple inaccuracies
- Very surface-level analysis

1 - Inadequate

- Fails to identify basic visual elements
- No meaningful reasoning
- Doesn't address prompt
- Fundamentally incorrect
- No real understanding demonstrated

Provide your evaluation in this JSON format:
{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

Base your evaluation primarily on how well the response demonstrates visual understanding and reasoning capabilities compared to the reference answer. Consider both what is said and what should have been noticed but wasn't. If the response is not in the language of the prompt, give it a score of 1.

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
    relative_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating how well vision language models analyze and respond to image-based prompts. You will be presented with:

The original prompt given to the model
Two different responses (Response A and Response B)

Evaluate both responses based on these key criteria:

Visual Detail Recognition: How well does each response identify and describe specific elements in the image?
Reasoning & Analysis: Do the responses make meaningful connections and draw logical conclusions?
Prompt Adherence: How well does each response address all aspects of the prompt?
Accuracy: Is the information provided accurate based on what's visible in the image?
Depth of Understanding: Do the responses demonstrate deeper insight beyond surface-level observation?

Consider both what is said and what should have been noticed but wasn't. A strong response should demonstrate:

Exceptional attention to visual details
Sophisticated reasoning and connections
Comprehensive addressing of prompt aspects
Accurate image interpretation
Deep insights beyond obvious observations

Provide your evaluation in this JSON format:
{"feedback": "<write feedback comparing the visual understanding and reasoning capabilities of each response>", "result": "<write "A" or "B">"}

If one response is in a language that is different from the one in the prompt, automatically consider the other answer better.

<START OF PROMPT>
${prompt}
<END OF PROMPT>

<START OF ANSWER A>
${answer_A}
<END OF ANSWER A>

<START OF ANSWER B>
${answer_B}
<END OF ANSWER B>

You may proceed to evaluate the responses. Ensure the output is valid JSON, without additional formatting or explanations.
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
class MVLMGeneralPurposeChatPortuguese(MVLMGeneralPurposeChat):
    def __post_init__(self):
        self.name = "m_vlm_general_purpose_chat_portuguese"
        self.task_attributes["language"] = ["Portuguese (Portugal)"]


@dataclass
class MVLMGeneralPurposeChatChinese(MVLMGeneralPurposeChat):
    def __post_init__(self):
        self.name = "m_vlm_general_purpose_chat_chinese_s"
        self.task_attributes["language"] = ["Chinese (Simplified)"]
