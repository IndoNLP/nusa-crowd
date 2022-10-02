import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas
import csv

_CITATION = """\
@article{nurlaila2018classification,
  title={CLASSIFICATION OF CUSTOMERS EMOTION USING NA{\"I}VE BAYES CLASSIFIER (Case Study: Natasha Skin Care)},
  author={Nurlaila, Afifah and Wiranto, Wiranto and Saptono, Ristu},
  journal={ITSMART: Jurnal Teknologi dan Informasi},
  volume={6},
  number={2},
  pages={92--97},
  year={2018}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "sentiment_nathasa_review"

_DESCRIPTION = """\
Customer Review (Natasha Skincare) is a customers emotion dataset, with amounted to 19,253 samples with the division for each class is 804 joy, 43 surprise, 154 anger, 61 fear, 287 sad, 167 disgust, and 17736 no-emotions.
"""

_HOMEPAGE = "https://jurnal.uns.ac.id/itsmart/article/viewFile/17328/15082"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?id=1D1pHX7CxrI-eIl2bAvIp1bWQeucyUGw0",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class SentimentNathasaReview(datasets.GeneratorBasedBuilder):
    """Customer Review (Natasha Skincare) is a customers emotion dataset, with amounted to 19,253 samples with the division for each class is 804 joy, 43 surprise, 154 anger, 61 fear, 287 sad, 167 disgust, and 17736 no-emotions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="sentiment_nathasa_review_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="sentiment_nathasa_review source schema",
            schema="source",
            subset_id="sentiment_nathasa_review",
        ),
        NusantaraConfig(
            name="sentiment_nathasa_review_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="sentiment_nathasa_review Nusantara schema",
            schema="nusantara_text",
            subset_id="sentiment_nathasa_review",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sentiment_nathasa_review_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                    "id": datasets.Value("string"),
                    "usr": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
               }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(['NOEMOTION', 'SURPRISE', 'SAD', 'JOY', 'FEAR', 'DISGUST', 'ANGER'])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = Path(dl_manager.download(_URLS[_DATASETNAME]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            with open(filepath, "r") as F:
                csvreader = csv.reader(F)
                for row in csvreader:
                    try:
                        row_data = eval(row[0].replace(';',','))[0]
                    except:
                        continue
                    if split == "train" and row_data[3] == "DATA LATIH":
                        ex = {
                            "id": row_data[0],
                            "usr": row_data[1],
                            "text": row_data[4],
                            "label": row_data[2],
                        }
                        yield row_data[0], ex
                    elif split == "test" and row_data[3] == "DATA UJI":
                        ex = {
                            "id": row_data[0],
                            "usr": row_data[1],
                            "text": row_data[4],
                            "label": row_data[2],
                        }
                        yield row_data[0], ex

        elif self.config.schema == "nusantara_text":
            with open(filepath, "r") as F:
                csvreader = csv.reader(F)
                for row in csvreader:
                    try:
                        row_data = eval(row[0].replace(';',','))[0]
                    except:
                        continue
                    if split == "train" and row_data[3] == "DATA LATIH":
                        ex = {
                            "id": row_data[0],
                            "text": row_data[4],
                            "label": row_data[2],
                        }
                        yield row_data[0], ex
                    elif split == "test" and row_data[3] == "DATA UJI":
                        ex = {
                            "id": row_data[0],
                            "text": row_data[4],
                            "label": row_data[2],
                        }
                        yield row_data[0], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
