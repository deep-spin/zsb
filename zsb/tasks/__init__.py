from zsb.tasks.base import Task
from zsb.tasks.end_to_end_mt_eval import EndToEndMTEvalEN_JA, EndToEndMTEvalEN_PTPT
from zsb.tasks.end_to_end_mt_eval_wo_examples import (
    EndToEndMTEvalWOExamplesEN_CS,
    EndToEndMTEvalWOExamplesEN_DE,
    EndToEndMTEvalWOExamplesEN_ES,
    EndToEndMTEvalWOExamplesEN_HI,
    EndToEndMTEvalWOExamplesEN_IS,
    EndToEndMTEvalWOExamplesEN_JA,
    EndToEndMTEvalWOExamplesEN_KO,
    EndToEndMTEvalWOExamplesEN_RU,
    EndToEndMTEvalWOExamplesEN_UK,
    EndToEndMTEvalWOExamplesEN_ZH,
)
from zsb.tasks.general_translation import (
    GeneralTranslationCS_UK,
    GeneralTranslationEN_CS,
    GeneralTranslationEN_DE,
    GeneralTranslationEN_ES,
    GeneralTranslationEN_HI,
    GeneralTranslationEN_IS,
    GeneralTranslationEN_JA,
    GeneralTranslationEN_RU,
    GeneralTranslationEN_UK,
    GeneralTranslationEN_ZH,
    GeneralTranslationJA_ZH,
)
from zsb.tasks.multilingual_general_purpose_chat import (
    GeneralPurposeChatChineseS,
    GeneralPurposeChatEnglish,
    GeneralPurposeChatFrench,
    GeneralPurposeChatKorean,
)
from zsb.tasks.multilingual_general_purpose_chat_rubric_1 import (
    GeneralPurposeChatChineseS_rubric_1,
    GeneralPurposeChatEnglish_rubric_1,
    GeneralPurposeChatFrench_rubric_1,
    GeneralPurposeChatKorean_rubric_1,
)
from zsb.tasks.multilingual_general_purpose_chat_rubric_2 import (
    GeneralPurposeChatChineseS_rubric_2,
    GeneralPurposeChatEnglish_rubric_2,
    GeneralPurposeChatFrench_rubric_2,
    GeneralPurposeChatKorean_rubric_2,
)
from zsb.tasks.multilingual_general_purpose_chat_topic_only import (
    GeneralPurposeChatEnglishTopicOnly,
)
from zsb.tasks.multilingual_general_purpose_chat_topic_subtopic import (
    GeneralPurposeChatEnglishTopicSubtopic,
)
from zsb.tasks.multilingual_vlm_general_purpose_chat import (
    MVLMGeneralPurposeChatChinese,
    MVLMGeneralPurposeChatPortuguese,
)
from zsb.tasks.transcreation import TranscreationEN_PTPT
from zsb.tasks.translation_w_hard_rules import (
    TranslationWHardRules_EN_ESLA,
    TranslationWHardRules_EN_PTPT,
    TranslationWHardRules_EN_ZH,
)
from zsb.tasks.translation_w_hard_rules_rule_eval import (
    TranslationWHardRulesRE_EN_ESLA,
    TranslationWHardRulesRE_EN_PTPT,
    TranslationWHardRulesRE_EN_ZH,
)
from zsb.tasks.vlm_general_purpose_chat import VLMGeneralPurposeChatEnglish

available_tasks: dict[str, Task] = {
    GeneralPurposeChatEnglish().name: GeneralPurposeChatEnglish(),
    GeneralPurposeChatChineseS().name: GeneralPurposeChatChineseS(),
    GeneralPurposeChatFrench().name: GeneralPurposeChatFrench(),
    GeneralPurposeChatKorean().name: GeneralPurposeChatKorean(),
    TranscreationEN_PTPT().name: TranscreationEN_PTPT(),
    EndToEndMTEvalEN_JA().name: EndToEndMTEvalEN_JA(),
    EndToEndMTEvalEN_PTPT().name: EndToEndMTEvalEN_PTPT(),
    EndToEndMTEvalWOExamplesEN_JA().name: EndToEndMTEvalWOExamplesEN_JA(),
    EndToEndMTEvalWOExamplesEN_DE().name: EndToEndMTEvalWOExamplesEN_DE(),
    EndToEndMTEvalWOExamplesEN_ES().name: EndToEndMTEvalWOExamplesEN_ES(),
    EndToEndMTEvalWOExamplesEN_RU().name: EndToEndMTEvalWOExamplesEN_RU(),
    EndToEndMTEvalWOExamplesEN_UK().name: EndToEndMTEvalWOExamplesEN_UK(),
    EndToEndMTEvalWOExamplesEN_IS().name: EndToEndMTEvalWOExamplesEN_IS(),
    EndToEndMTEvalWOExamplesEN_HI().name: EndToEndMTEvalWOExamplesEN_HI(),
    EndToEndMTEvalWOExamplesEN_ZH().name: EndToEndMTEvalWOExamplesEN_ZH(),
    EndToEndMTEvalWOExamplesEN_CS().name: EndToEndMTEvalWOExamplesEN_CS(),
    EndToEndMTEvalWOExamplesEN_KO().name: EndToEndMTEvalWOExamplesEN_KO(),
    GeneralTranslationEN_DE().name: GeneralTranslationEN_DE(),
    GeneralTranslationEN_ZH().name: GeneralTranslationEN_ZH(),
    GeneralTranslationCS_UK().name: GeneralTranslationCS_UK(),
    GeneralTranslationJA_ZH().name: GeneralTranslationJA_ZH(),
    GeneralTranslationEN_ES().name: GeneralTranslationEN_ES(),
    GeneralTranslationEN_CS().name: GeneralTranslationEN_CS(),
    GeneralTranslationEN_RU().name: GeneralTranslationEN_RU(),
    GeneralTranslationEN_UK().name: GeneralTranslationEN_UK(),
    GeneralTranslationEN_HI().name: GeneralTranslationEN_HI(),
    GeneralTranslationEN_JA().name: GeneralTranslationEN_JA(),
    GeneralTranslationEN_IS().name: GeneralTranslationEN_IS(),
    TranslationWHardRules_EN_PTPT().name: TranslationWHardRules_EN_PTPT(),
    TranslationWHardRules_EN_ESLA().name: TranslationWHardRules_EN_ESLA(),
    TranslationWHardRules_EN_ZH().name: TranslationWHardRules_EN_ZH(),
    TranslationWHardRulesRE_EN_PTPT().name: TranslationWHardRulesRE_EN_PTPT(),
    TranslationWHardRulesRE_EN_ESLA().name: TranslationWHardRulesRE_EN_ESLA(),
    TranslationWHardRulesRE_EN_ZH().name: TranslationWHardRulesRE_EN_ZH(),
    VLMGeneralPurposeChatEnglish().name: VLMGeneralPurposeChatEnglish(),
    MVLMGeneralPurposeChatPortuguese().name: MVLMGeneralPurposeChatPortuguese(),
    MVLMGeneralPurposeChatChinese().name: MVLMGeneralPurposeChatChinese(),
    # ablations for paper
    GeneralPurposeChatEnglishTopicOnly().name: GeneralPurposeChatEnglishTopicOnly(),
    GeneralPurposeChatEnglishTopicSubtopic().name: GeneralPurposeChatEnglishTopicSubtopic(),
    # ablations for reviews
    GeneralPurposeChatEnglish_rubric_1().name: GeneralPurposeChatEnglish_rubric_1(),
    GeneralPurposeChatEnglish_rubric_2().name: GeneralPurposeChatEnglish_rubric_2(),
    GeneralPurposeChatChineseS_rubric_1().name: GeneralPurposeChatChineseS_rubric_1(),
    GeneralPurposeChatChineseS_rubric_2().name: GeneralPurposeChatChineseS_rubric_2(),
    GeneralPurposeChatFrench_rubric_1().name: GeneralPurposeChatFrench_rubric_1(),
    GeneralPurposeChatFrench_rubric_2().name: GeneralPurposeChatFrench_rubric_2(),
    GeneralPurposeChatKorean_rubric_1().name: GeneralPurposeChatKorean_rubric_1(),
    GeneralPurposeChatKorean_rubric_2().name: GeneralPurposeChatKorean_rubric_2(),
}
