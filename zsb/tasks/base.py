from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template
from typing import Callable


@dataclass
class Task(ABC):
    """
    An abstract base class representing a task with configurable attributes and prompt templates.

    This class serves as a foundation for implementing specific task types, providing structure
    for task definitions, prompt management, and output parsing.

    Attributes:
        name (str): The name of the task.
        description (str): A detailed description of what the task entails.
        task_attributes (dict[str, list[str] | dict[str, str | Callable]]): Dictionary defining
            task attributes and their possible values. Can include both independent attributes
            (list[str]) and dependent attributes (dictionary with a "_depends_on" key pointing to an existing
            independent attribute, and a "callable" key pointing to a function that accesses a dictionary with keys corresponding
            to values from the independent attribute, and values are lists of possible attributes.
        meta_prompt (dict[str, str | Template]): Dictionary containing system_prompt and user_prompt keys
            and either a fixed string or a Template that takes task attributes. This prompt is used to query LLMs
            to generate task-specific prompts (e.g., if the task is tranlation, a simple meta-prompt would be
            "Generate a translation instruction for a source text and a reference translation").
        da_judge_prompt (dict[str, Template]): Same structure as meta prompt, but this prompt is used to query the model to evaluate
            an answer to the task, given a prompt (generated from the meta prompt). The output must be a numerical score and an explanation.
            Must have placeholders for "prompt" and "answer".
        relative_judge_prompt (dict[str, Template]): Same structure as the da judge prompt, but the goal is to compare two answers to the task.
            The output must be a preferred option (A or B) and an explanation.
            Must have placeholders for "prompt", "answer_A", and "answer_B".

    Methods:
        parse_meta_prompt_output(output: str) -> dict[str, str]:
            Abstract method to parse the output from meta prompts.
            Args:
                output (str): Raw output string from the meta prompt.
            Returns:
                dict[str, str] | False: Parsed key-value pairs from the output
                    with the format {"prompt": str, "source": str, "reference": str}.
                    The prompt will include the source (it should be ready to be passed to another model).
                    If parsing fails, returns false. This is useful for retrying the prompt generation.

        parse_da_prompt_output(output: str) -> tuple[int, str]:
            Abstract method to parse direct assessment prompt outputs.
            Args:
                output (str): Raw output string from the direct assessment prompt.
            Returns:
                tuple[int, str]: A tuple containing the numerical score and explanation.

        parse_relative_prompt_output(output: str, a_place: int) -> tuple[str, str]:
            Abstract method to parse relative judgment prompt outputs.
            Args:
                output (str): Raw output string from the relative judgment prompt.
                a_place (int): The index of the answer that corresponds to option A.
            Returns:
                tuple[str, str]: A tuple containing the preferred option and explanation.

        get_all_possible_combinations() -> dict[str, str]:
            Generates all possible combinations of task attributes, handling both independent
            and dependent attributes.
            Returns:
                dict[str, str]: Dictionary containing all possible attribute combinations.
    """

    name: str
    description: str
    task_attributes: dict[str, list[str] | dict[str, str | Callable]]
    meta_prompt: dict[str, str | Template]
    da_judge_prompt: dict[str, Template]
    relative_judge_prompt: dict[str, Template]

    @staticmethod
    @abstractmethod
    def parse_meta_prompt_output(output: str) -> dict[str, str] | bool:
        pass

    @staticmethod
    @abstractmethod
    def parse_da_prompt_output(output: str) -> tuple[int, str]:
        pass

    @staticmethod
    @abstractmethod
    def parse_relative_prompt_output(output: str, a_place: int) -> tuple[str, str]:
        pass
