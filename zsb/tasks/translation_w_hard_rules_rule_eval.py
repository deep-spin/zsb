import itertools
import json
import random
import re
from dataclasses import dataclass, field
from string import Template
from typing import Callable

from zsb.attributes import AUDIENCES, LENGTHS, STYLES, SUBTOPICS, TOPICS
from zsb.tasks.translation_w_hard_rules import TranslationWHardRules


@dataclass
class TranslationWHardRulesRE(TranslationWHardRules):
    name: str = "translation_w_hard_rules_re"
    da_judge_prompt: dict[str, Template] = field(
        default_factory=lambda: {
            "system_prompt": None,
            "user_prompt": Template(
                """You are an expert judge evaluating translation quality. You will be presented with:

- A text, prompting a model for a translation of a source following some rules
- A translation to evaluate

Rate the translation on a scale of 1-6 based on how well it follows the specified rules and instructions in the prompt, regardless of overall translation quality, according to the following criteria:

- Rule Adherence: Does the translation follow all explicit rules stated in the prompt?
- Instruction Compliance: Are specific formatting, style, or technical instructions followed?
- Constraint Observance: Are any limitations or restrictions properly respected?
- Specification Accuracy: Does the output match the exact specifications requested?
- Requirement Fulfillment: Are all mandatory elements present as instructed?

Scoring Rubric:
6 - Perfect Compliance

- Follows every single rule and instruction precisely
- No deviations from any specified constraints
- All requirements fully met as requested
- Complete adherence to formatting/style directives
- Perfect execution of all procedural instructions
- Zero rule violations of any kind

5 - Excellent Compliance

- Follows nearly all rules with only trivial deviations
- Minor lapses that don't affect core requirements
- Strong adherence to most constraints and directives
- Formatting/style mostly correct
- Very few rule violations, all inconsequential

4 - Good Compliance
- Follows most important rules correctly
- Some minor rule violations that don't undermine main objectives
- Generally respects constraints and limitations
- Adequate adherence to formatting requirements
- Few significant rule violations

3 - Fair Compliance
- Follows some rules but misses several others
- Notable violations of stated constraints
- Inconsistent adherence to instructions
- Some formatting/style requirements ignored
- Multiple rule violations affecting compliance

2 - Poor Compliance
- Fails to follow many stated rules
- Significant violations of constraints and limitations
- Poor adherence to specific instructions
- Formatting/style requirements largely ignored
- Frequent and notable rule violations

1 - No Compliance
- Ignores most or all stated rules
- Complete disregard for constraints and limitations
- Fails to follow basic instructions
- No attention to specified requirements
- Systematic rule violations throughout

Provide your evaluation in this JSON format:

{"feedback": "<detailed explanation of the score based on the criteria>", "result": "<only a number from 1 to 6>"}

<START OF SOURCE TEXT>
${prompt}
<END OF SOURCE TEXT>

<START OF TRANSLATION>
${answer}
<END OF TRANSLATION>
                                    
You may proceed to evaluate the translation. Focus on evaluating the extent to which the translation follows the rules in the prompt, not its quality. Ensure the output is valid JSON, without additional formatting or explanations."""
            ),
        }
    )


@dataclass
class TranslationWHardRulesRE_EN_PTPT(TranslationWHardRulesRE):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_re_en_ptpt"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["European Portuguese"]


@dataclass
class TranslationWHardRulesRE_EN_ESLA(TranslationWHardRulesRE):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_re_en_esla"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Spanish (Latin America)"]


@dataclass
class TranslationWHardRulesRE_EN_ZH(TranslationWHardRulesRE):
    def __post_init__(self):
        self.name = "translation_w_hard_rules_re_en_zh"
        self.task_attributes["source_language"] = ["English"]
        self.task_attributes["target_language"] = ["Chinese (Simplified)"]
