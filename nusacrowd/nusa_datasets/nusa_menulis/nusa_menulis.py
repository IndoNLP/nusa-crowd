from pathlib import Path
from typing import List

import datasets
import json

import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "nusa_menulis"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "abs", "btk", "bew", "bhp", "jav", "mad", "mak", "min", "mui", "rej", "sun"]
_LOCAL = False
_CITATION = """\

"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/IndoNLP"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
	"btk": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_batak.csv",
	"bew": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_betawi.csv",
	"bug": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_buginese.csv",
	"jav": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_javanese.csv",
	"mad": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_madurese.csv",
	"mak": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_makassarese.csv",
	"min": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_minangkabau.csv",
	"mui": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_palembang.csv",
	"rej": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_rejang_lebong.csv",
	"sun": "https://huggingface.co/datasets/indonlp/nusa_menulis/raw/main/menulis_sundanese.csv",
}

CODE2LANG = {
    "btk": "Batak",
    "bew": "Betawi",
    "bug": "Buginese",
    "jav": "Javanese",
    "mad": "Madurese",
    "mak": "Makassarese",
    "min": "Minangkabau",
    "mui": "Palembang / Musi",
    "rej": "Rejang (Lebong)",
    "sun": "Sundanese"
}

_SUPPORTED_TASKS = [
    Tasks.SELF_SUPERVISED_PRETRAINING,
    Tasks.TOPIC_CLASSIFICATION,
    Tasks.EMOTION_CLASSIFICATION,
    Tasks.PARAGRAPH_CLASSIFICATION,
]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"

class NusaMenulis(datasets.GeneratorBasedBuilder):
    TOPIC_LIST = [
        "Business", "Culture & Heritage", "Food & Beverages", "History", "Leisures", "Politics", 
        "Religion", "Science", "Slice of Life", "Social Media", "Sports", "Technology"
    ]
    EMOTION_LIST = [
         "Emotion: Angry", "Emotion: Fear", "Emotion: Happy", "Emotion: Sad",
         "Emotion: Disgusted", "Emotion: Shame", "Emotion: Surprise",
    ]
    PARAGRAPH_LIST = ['narrative', 'descriptive', 'argumentative', 'expository', 'persuasive']
    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"nusa_menulis_{lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"NusaMenulis {CODE2LANG[lang]} source schema",
            schema="source",
            subset_id=f"nusa_menulis_{lang}",
        ) for lang in CODE2LANG.keys()
    ] + [
        NusantaraConfig(
            name=f"nusa_menulis_{lang}_topic_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaMenulis {CODE2LANG[lang]} topic classification nusantara schema",
            schema="nusantara_text",
            subset_id=f"nusa_menulis_topic_{lang}",
        ) for lang in CODE2LANG.keys()
    ] +  [
        NusantaraConfig(
            name=f"nusa_menulis_{lang}_emotion_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaMenulis {CODE2LANG[lang]} emotion classification nusantara schema",
            schema="nusantara_text",
            subset_id=f"nusa_menulis_emotion_{lang}",
        ) for lang in CODE2LANG.keys()
    ] + [
        NusantaraConfig(
            name=f"nusa_menulis_{lang}_paragraph_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaMenulis {CODE2LANG[lang]} paragraph classification nusantara schema",
            schema="nusantara_text",
            subset_id=f"nusa_menulis_paragraph_{lang}",
        ) for lang in CODE2LANG.keys()
    ] + [
        NusantaraConfig(
            name=f"nusa_menulis_{lang}_nusantara_ssp",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaMenulis {CODE2LANG[lang]} self-supervised pre-training nusantara schema",
            schema="nusantara_ssp",
            subset_id=f"nusa_menulis_ssp_{lang}",
        ) for lang in CODE2LANG.keys()
    ]

    DEFAULT_CONFIG_NAME = "nusa_menulis_jav_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    'id': datasets.Value("string"),
                    'lang': datasets.Value("string"),
                    'topic': datasets.Value("string"),
                    'paragraph_type': datasets.Value("string"),
                    'paragraph': datasets.Value("string"),
                    'num_tokens': datasets.Value("string")
                }
            )
        elif self.config.schema == "nusantara_text":
            # Support/Neutral/Contradict topic is omitted
            if 'topic' in self.config.name:
                features = schemas.text_features(self.TOPIC_LIST)
            elif 'emotion' in self.config.name:
                features = schemas.text_features(self.EMOTION_LIST)
            else: # 'paragraph' in self.config.name
                features = schemas.text_features(self.PARAGRAPH_LIST)
        else: # self.config.schema == "nusantara_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = self.config.name.split("_")[2]
        url = _URLs[lang]
        path = dl_manager.download(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path
                },
            ),
        ]

    def _generate_examples(self, filepath: List[Path]):
        df = pd.read_csv(filepath)

        if self.config.schema == "source":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": row.id,
                    "lang": row.lang,
                    "topic": row.topic,
                    "paragraph_type": row.paragraph_type,
                    "paragraph": row.paragraph,
                    "num_tokens": row.num_tokens
                }
                yield row.id, ex
        elif self.config.schema == "nusantara_text":
            # Support/Neutral/Contradict topic is omitted
            if 'topic' in self.config.name:
                df = df.loc[df['topic'].isin(self.TOPIC_LIST),:]
                for idx, row in enumerate(df.itertuples()):
                    ex = {
                        "id": str(row.id), 
                        "text": row.paragraph, 
                        "label": row.topic
                    }
                    yield row.id, ex
            elif 'emotion' in self.config.name:
                df = df.loc[df['topic'].isin(self.EMOTION_LIST),:]
                for idx, row in enumerate(df.itertuples()):
                    ex = {
                        "id": str(row.id), 
                        "text": row.paragraph, 
                        "label": row.topic
                    }
                    yield row.id, ex
            else: # 'paragraph' in self.config.name
                for idx, row in enumerate(df.itertuples()):
                    ex = {
                        "id": str(row.id), 
                        "text": row.paragraph, 
                        "label": row.paragraph_type
                    }
                    yield row.id, ex                
        elif self.config.schema == "nusantara_ssp":        
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(row.id), 
                    "text": row.paragraph
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")