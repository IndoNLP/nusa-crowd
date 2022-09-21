from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """
@inproceedings{azhar2019multi,
  title={Multi-label Aspect Categorization with Convolutional Neural Networks and Extreme Gradient Boosting},
  author={A. N. Azhar, M. L. Khodra, and A. P. Sutiono}
  booktitle={Proceedings of the 2019 International Conference on Electrical Engineering and Informatics (ICEEI)},
  pages={35--40},
  year={2019}
}
"""


_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "hoasa"

_DESCRIPTION = """
HoASA: An aspect-based sentiment analysis dataset consisting of hotel reviews collected from the hotel aggregator platform, AiryRooms.
The dataset covers ten different aspects of hotel quality. Similar to the CASA dataset, each review is labeled with a single sentiment label for each aspect.
There are four possible sentiment classes for each sentiment label:
positive, negative, neutral, and positive-negative.
The positivenegative label is given to a review that contains multiple sentiments of the same aspect but for different objects (e.g., cleanliness of bed and toilet).
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/train_preprocess.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/valid_preprocess.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/test_preprocess.csv",
}

_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class HoASA(datasets.GeneratorBasedBuilder):
    """HoASA is an aspect based sentiment analysis dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="hoasa_source",
            version=SOURCE_VERSION,
            description="HoASA source schema",
            schema="source",
            subset_id="hoasa",
        ),
        NusantaraConfig(
            name="hoasa_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="HoASA Nusantara schema",
            schema="nusantara_text_multi",
            subset_id="hoasa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "hoasa_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("int64"),
                    "review": datasets.Value("string"),
                    "ac": datasets.Value("string"),
                    "air_panas": datasets.Value("string"),
                    "bau": datasets.Value("string"),
                    "general": datasets.Value("string"),
                    "kebersihan": datasets.Value("string"),
                    "linen": datasets.Value("string"),
                    "service": datasets.Value("string"),
                    "sunrise_meal": datasets.Value("string"),
                    "tv": datasets.Value("string"),
                    "wifi": datasets.Value("string"),
                }
            )

        elif self.config.schema == "nusantara_text_multi":
            features = schemas.text_multi_features(["pos", "neut", "neg", "neg_pos"])

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
                entry = {
                    "index": row.index,
                    "review": row.review,
                    "ac": row.ac,
                    "air_panas": row.air_panas,
                    "bau": row.bau,
                    "general": row.general,
                    "kebersihan": row.kebersihan,
                    "linen": row.linen,
                    "service": row.service,
                    "sunrise_meal": row.sunrise_meal,
                    "tv": row.tv,
                    "wifi": row.wifi,
                }
                yield row.index, entry

        elif self.config.schema == "nusantara_text_multi":
            for row in df.itertuples():
                entry = {
                    "id": str(row.index),
                    "text": row.review,
                    "labels": [label for label in row[3:]],
                }
                yield row.index, entry
