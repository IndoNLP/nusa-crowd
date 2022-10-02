import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas
import pandas as pd

_CITATION = """\
@inproceedings{wongso2021causal,
  title={Causal and masked language modeling of Javanese language using transformer-based architectures},
  author={Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
  booktitle={2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
  pages={1--7},
  year={2021},
  organization={IEEE}
}
"""

_DATASETNAME = "imdb_jv"

_DESCRIPTION = """\
Javanese Imdb Movie Reviews Dataset is a Javanese version of the IMDb Movie Reviews dataset by translating the original English dataset to Javanese.
"""

_HOMEPAGE = "https://huggingface.co/datasets/w11wo/imdb-javanese"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/w11wo/imdb-javanese/resolve/main/javanese_imdb_csv.zip",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class IMDbJv(datasets.GeneratorBasedBuilder):
    """Javanese Imdb Movie Reviews Dataset is a Javanese version of the IMDb Movie Reviews dataset by translating the original English dataset to Javanese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="imdb_jv_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="imdb_jv source schema",
            schema="source",
            subset_id="imdb_jv",
        ),
        NusantaraConfig(
            name="imdb_jv_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="imdb_jv Nusantara schema",
            schema="nusantara_text",
            subset_id="imdb_jv",
        ),
    ]

    DEFAULT_CONFIG_NAME = "imdb_jv_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string")
               }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(['1', '0', '-1'])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        data_files = {
            "train": "javanese_imdb_train.csv",
            "unsupervised": "javanese_imdb_unsup.csv",
            "test": "javanese_imdb_test.csv",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name="unsupervised",
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["unsupervised"]),
                    "split": "unsupervised",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["test"]),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            data = pd.read_csv(filepath)
            length = len(data['label'])
            for id in range(length):
                ex = {
                    "id": str(id),
                    "text": data['text'][id],
                    "label": data['label'][id],
                }
                yield id, ex

        elif self.config.schema == "nusantara_text":
            data = pd.read_csv(filepath)
            length = len(data['label'])
            for id in range(length):
                ex = {
                    "id": str(id),
                    "text": data['text'][id],
                    "label": data['label'][id],
                }
                yield id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
