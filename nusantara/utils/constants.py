from enum import Enum
from enum import Enum
from types import SimpleNamespace

METADATA = ["_LOCAL", "_LANGUAGES"]

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
    NAMED_ENTITY_RECOGNITION = "NER"
    SENTENCE_ORDERING = "SO"

    # Pair Text Classification
    QUESTION_ANSWERING = "QA"
    TEXTUAL_ENTAILMENT = "TE"
    SEMANTIC_SIMILARITY = "STS"

    # Single Text Generation
    MACHINE_TRANSLATION = "MT"
    PARAPHRASING = "PARA"
    SUMMARIZATION = "SUM"

    # Multi Text Generation
    DIALOGUE_SYSTEM = "DS"

    # Self Supervised Pretraining
    SELF_SUPERVISED_PRETRAINING = "SSP"

    # Speech Recognition
    SPEECH_RECOGNITION = "ASR"


# TASK_TO_SCHEMA = {
#     Tasks.NAMED_ENTITY_RECOGNITION: "KB",
#     Tasks.DEPENDENCY_PARSING: "KB",
#     Tasks.RELATION_EXTRACTION: "KB",
#     Tasks.COREFERENCE_RESOLUTION: "KB",
#     Tasks.QUESTION_ANSWERING: "QA",
#     Tasks.TEXTUAL_ENTAILMENT: "TE",
#     Tasks.SEMANTIC_SIMILARITY: "PAIRS",
#     Tasks.PARAPHRASING: "T2T",
#     Tasks.MACHINE_TRANSLATION: "T2T",
#     Tasks.SUMMARIZATION: "T2T",
#     Tasks.SENTIMENT_ANALYSIS: "TEXT",
# }

# SCHEMA_TO_TASKS = defaultdict(set)
# for task, schema in TASK_TO_SCHEMA.items():
#     SCHEMA_TO_TASKS[schema].add(task)
# SCHEMA_TO_TASKS = dict(SCHEMA_TO_TASKS)

# VALID_TASKS = set(TASK_TO_SCHEMA.keys())
# VALID_SCHEMAS = set(TASK_TO_SCHEMA.values())

# SCHEMA_TO_FEATURES = {
#     "KB": kb_features,
#     "QA": qa_features,
#     "TE": entailment_features,
#     "T2T": text2text_features,
#     "TEXT": text_features,
#     "PAIRS": pairs_features,
# }
