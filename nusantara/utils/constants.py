from enum import Enum
from types import SimpleNamespace
from collections import defaultdict
from nusantara.utils.schemas import (
    kb_features, qa_features, text2text_features, text_features, 
    text_multi_features, pairs_features, seq_label_features, 
    ssp_features, speech_text_features, image_text_features
)

METADATA: dict = {
    "_LOCAL": bool,
    "_LANGUAGES": str,
    "_DISPLAYNAME": str,
}

NusantaraValues = SimpleNamespace(NULL="<NUSA_NULL_STR>")

# Default View Name
DEFAULT_SOURCE_VIEW_NAME = "source"
DEFAULT_NUSANTARA_VIEW_NAME = "nusantara"

class Tasks(Enum):
    DEPENDENCY_PARSING = "DEP"
    WORD_SENSE_DISAMBIGUATION = "WSD"
    KEYWORD_EXTRACTION = "KE"
    COREFERENCE_RESOLUTION = "COREF"

    # Single Text Classification
    SENTIMENT_ANALYSIS = "SA"
    ASPECT_BASED_SENTIMENT_ANALYSIS = "ABSA"
    EMOTION_CLASSIFICATION = "EC"

    # Single Text Sequence Labeling
    POS_TAGGING = "POS"
    KEYWORD_TAGGING = "KT"
    NAMED_ENTITY_RECOGNITION = "NER"
    SENTENCE_ORDERING = "SO"

    # Pair Text Classification
    QUESTION_ANSWERING = "QA"
    TEXTUAL_ENTAILMENT = "TE"
    SEMANTIC_SIMILARITY = "STS"
    NEXT_SENTENCE_PREDICTION = "NSP"
    SHORT_ANSWER_GRADING = "SAG"

    # Single Text Generation
    MACHINE_TRANSLATION = "MT"
    PARAPHRASING = "PARA"
    SUMMARIZATION = "SUM"

    # Multi Text Generation
    DIALOGUE_SYSTEM = "DS"

    # Self Supervised Pretraining
    SELF_SUPERVISED_PRETRAINING = "SSP"

    # SpeechText
    SPEECH_RECOGNITION = "ASR"
    SPEECH_TO_TEXT_TRANSLATION = "STTT"
    TEXT_TO_SPEECH = "TTS"

    # ImageText
    IMAGE_CAPTIONING = "IC"
    STYLIZED_IMAGE_CAPTIONING = "SIC"
    VISUALLY_GROUNDED_REASONING = "VGR"
    
TASK_TO_SCHEMA = {
    Tasks.DEPENDENCY_PARSING: "KB",
    Tasks.WORD_SENSE_DISAMBIGUATION: "KB",
    Tasks.KEYWORD_EXTRACTION: "SEQ_LABEL",
    Tasks.COREFERENCE_RESOLUTION: "KB",
    Tasks.DIALOGUE_SYSTEM: "KB",
    Tasks.NAMED_ENTITY_RECOGNITION: "SEQ_LABEL",
    Tasks.POS_TAGGING: "SEQ_LABEL",
    Tasks.KEYWORD_TAGGING: "SEQ_LABEL",
    Tasks.SENTENCE_ORDERING: "SEQ_LABEL",
    Tasks.QUESTION_ANSWERING: "QA",
    Tasks.TEXTUAL_ENTAILMENT: "PAIRS",
    Tasks.SEMANTIC_SIMILARITY: "PAIRS_SCORE",
    Tasks.NEXT_SENTENCE_PREDICTION: "PAIRS",
    Tasks.SHORT_ANSWER_GRADING: "PAIRS_SCORE",
    Tasks.PARAPHRASING: "T2T",
    Tasks.MACHINE_TRANSLATION: "T2T",
    Tasks.SUMMARIZATION: "T2T",
    Tasks.SENTIMENT_ANALYSIS: "TEXT",
    Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS: "TEXT_MULTI",
    Tasks.EMOTION_CLASSIFICATION: "TEXT",
    Tasks.SELF_SUPERVISED_PRETRAINING: "SSP",
    Tasks.SPEECH_RECOGNITION: "SPTEXT",
    Tasks.SPEECH_TO_TEXT_TRANSLATION: "SPTEXT",
    Tasks.TEXT_TO_SPEECH: "SPTEXT",
    Tasks.IMAGE_CAPTIONING: "IMTEXT",
    Tasks.STYLIZED_IMAGE_CAPTIONING: "IMTEXT",
    Tasks.VISUALLY_GROUNDED_REASONING: "IMTEXT",
}

SCHEMA_TO_TASKS = defaultdict(set)
for task, schema in TASK_TO_SCHEMA.items():
    SCHEMA_TO_TASKS[schema].add(task)
SCHEMA_TO_TASKS = dict(SCHEMA_TO_TASKS)

VALID_TASKS = set(TASK_TO_SCHEMA.keys())
VALID_SCHEMAS = set(TASK_TO_SCHEMA.values())

SCHEMA_TO_FEATURES = {
    "KB": kb_features,
    "QA": qa_features,
    "T2T": text2text_features,
    "TEXT": text_features(),
    "TEXT_MULTI": text_multi_features(),
    "PAIRS": pairs_features(),
    "PAIRS_SCORE": pairs_features_score(),
    "SEQ_LABEL": seq_label_features(),
    "SSP": ssp_features,
    "SPTEXT": speech_text_features,
    "IMTEXT": image_text_features()
}

TASK_TO_FEATURES = {
    Tasks.NAMED_ENTITY_RECOGNITION: {"entities"},
    Tasks.DEPENDENCY_PARSING: {"relations", "entities"},
    Tasks.COREFERENCE_RESOLUTION: {"entities", "coreferences"},
    # Tasks.NAMED_ENTITY_DISAMBIGUATION: {"entities", "normalized"},
    # Tasks.EVENT_EXTRACTION: {"events"}
}