from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME
from nusantara.utils.common_parser import load_conll_data

_DATASETNAME = "emot"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ['ind'] # We follow ISO639-3 langauge code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\

"""

_DESCRIPTION = """\    
EmoT is an emotion classification dataset collected from the social media platform Twitter. The dataset consists of around 4000 Indonesian colloquial language tweets, covering five different emotion labels: anger, fear, happiness, love, and sadness.
EmoT dataset is splitted into 3 sets with 3521 train, 440 validation, 442 test data.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/valid_preprocess.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/test_preprocess.csv"
}

_SUPPORTED_TASKS = [
    Tasks.EMOTION_CLASSIFICATION
]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"

class BaPOSDataset(datasets.GeneratorBasedBuilder):
    """BaPOS is a POS tagging dataset contains about 10,000 sentences, collected from the PAN Localization Project tagged with 23 POS tag classes."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="bapos_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="BaPOS source schema",
            schema="source",
            subset_id="bapos",
        ),
        NusantaraConfig(
            name="bapos_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="BaPOS Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="bapos",
        )
    ]

    DEFAULT_CONFIG_NAME = "bapos_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "pos_tag": [datasets.Value("string")]
                }
            )
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs['train']))
        validation_tsv_path = Path(dl_manager.download_and_extract(_URLs['validation']))
        test_tsv_path = Path(dl_manager.download_and_extract(_URLs['test']))
        data_files = {
            "train": train_tsv_path,
            "validation": validation_tsv_path,
            "test": test_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        conll_dataset = load_conll_data(filepath) # [{'sentence': [T1, T2, ..., Tn], 'labels': [L1, L2, ..., Ln]}]

        if self.config.schema == "source":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "index": str(i),
                    "tokens": row['sentence'],
                    "pos_tag": row['label']
                }
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "id": str(i),
                    "tokens": row['sentence'],
                    "labels": row['label']
                }
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")