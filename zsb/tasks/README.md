# Creating a new ZSB
To create a new task for benchmarking, you must create a file where a task class is defined, and import it in `__init__.py`.

The task class requires creating a **meta prompt** for data generation, a **judgement prompt** (DA, relative, or both) for evaluation, and parsing functions.

Refer to `multilingual_general_purpose_chat.py` or any other task file for useful examples.

The file you create must define a task object with the following attributes and methods:

```python
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.tasks.base import Task

@dataclass
class Example(Task):
    name: str = "example"
    description: str = (
        "Example Task"
    )
    task_attributes: dict[str, list[str] | dict[str, str | Callable]] = field(
        default_factory=lambda: {
            "some_attribute": some_list
        }
    )
    meta_prompt: dict[str, str | Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """some prompt"""
            ),
        }
    )
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """some prompt"""
            ),
        }
    )
    ref_da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """some prompt"""
            ),
        }
    )
    relative_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """some prompt"""
            ),
        }
    )

    @staticmethod
    def parse_meta_prompt_output(output: str) -> dict[str, str] | bool:
        pass

    @staticmethod
    def parse_da_prompt_output(output: str) -> tuple[int, str]:
        pass

    @staticmethod
    def parse_relative_prompt_output(output: str, a_place: int) -> tuple[str, str]:
        pass
```

`task_attributes` will be used in the meta prompt for data generation. 
Create one in the dictionary, and place its key somewhere in the meta prompt like `${some_attribute}`. 
The value must be a list of strings, or, alternatively, if it depends on some other attribute (like subtopic depends on topic), a dictionary with the form 

```python
{
    "_depends_on": "some_attribute", 
    "callable": lambda some_attribute: OTHER_ATTRIBUTE_LIST[some_attribute]
}
```

When creating data, values will be sampled from these attribute lists to create varied and unique data instances.

Then, a parsing function must be defined in `parse_meta_prompt_output`, which must return a dictionary where both keys and values are strings.

The judgment prompts can be defined arbitrarily, provided their output can be parsed somehow into whatever their parsing functions (`parse_da_prompt_output` or `parse_relative_prompt_output`) expect. 
The parsing functions must return a tuple where the first element is an integer and the second a string.

`name` and `description` are arbitrary. The former is used in scripts.

Experiment by defining a new task, importing it in `__init__.py` and adding it to `available_tasks`, like:

```python
from zsb.tasks.example import Example
...
available_tasks: dict[str, Task] = {
    ...
    Example().name: Example(),
}
```

and then, in this folder, running:

```bash
python list.py
```

Your task `name` should appear in the output of the script.