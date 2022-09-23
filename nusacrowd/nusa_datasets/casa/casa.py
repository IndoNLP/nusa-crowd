from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """
@INPROCEEDINGS{8629181,
    author={Ilmania, Arfinda and Abdurrahman and Cahyawijaya, Samuel and Purwarianti, Ayu},
    booktitle={2018 International Conference on Asian Language Processing (IALP)},
    title={Aspect Detection and Sentiment Classification Using Deep Neural Network for Indonesian Aspect-Based Sentiment Analysis},
    year={2018},
    volume={},
    number={},
    pages={62-67},
    doi={10.1109/IALP.2018.8629181
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "casa"

_DESCRIPTION = """
CASA: An aspect-based sentiment analysis dataset consisting of around a thousand car reviews collected from multiple Indonesian online automobile platforms (Ilmania et al., 2018).
The dataset covers six aspects of car quality.
We define the task to be a multi-label classification task,
where each label represents a sentiment for a single aspect with three possible values: positive, negative, and neutral.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/train_preprocess.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/valid_preprocess.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/test_preprocess.csv",
}

_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class CASA(datasets.GeneratorBasedBuilder):
    """CASA is an aspect based sentiment analysis dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="casa_source",
            version=SOURCE_VERSION,
            description="CASA source schema",
            schema="source",
            subset_id="casa",
        ),
        NusantaraConfig(
            name="casa_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="CASA Nusantara schema",
            schema="nusantara_text_multi",
            subset_id="casa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "casa_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("int64"),
                    "sentence": datasets.Value("string"),
                    "fuel": datasets.Value("string"),
                    "machine": datasets.Value("string"),
                    "others": datasets.Value("string"),
                    "part": datasets.Value("string"),
                    "price": datasets.Value("string"),
                    "service": datasets.Value("string"),
                }
            )

        elif self.config.schema == "nusantara_text_multi":
            features = schemas.text_multi_features(["positive", "neutral", "negative"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"]))
        validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"]))
        test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"]))

        data_dir = {
            "train": train_csv_path,
            "validation": validation_csv_path,
            "test": test_csv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()
        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {"index": row.index, "sentence": row.sentence, "fuel": row.fuel, "machine": row.machine, "others": row.others, "part": row.part, "price": row.price, "service": row.service}
                yield row.index, entry

        elif self.config.schema == "nusantara_text_multi":
            for row in df.itertuples():
                entry = {
                    "id": str(row.index),
                    "text": row.sentence,
                    "labels": [label for label in row[3:]],
                }
                yield row.index, entry
