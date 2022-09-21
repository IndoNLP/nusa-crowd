from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "netifier"

_DESCRIPTION = """\
Netifier dataset is a collection of scraped posts on famous social media sites in Indonesia,
such as Instagram, Twitter, and Kaskus aimed to do multi-label toxicity classification.
The dataset consists of 7,773 texts. The author manually labelled ~7k samples into 4 categories:
pornography, hate speech, racism, and radicalism.
"""

_HOMEPAGE = "https://github.com/ahmadizzan/netifier"
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
_URLS = {_DATASETNAME: {"train": "https://raw.githubusercontent.com/ahmadizzan/netifier/master/data/processed/train.csv", "test": "https://raw.githubusercontent.com/ahmadizzan/netifier/master/data/processed/test.csv"}}
_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class Netifier(datasets.GeneratorBasedBuilder):
    """Netifier dataset is a collection of scraped posts on famous social media sites in Indonesia,
    such as Instagram, Twitter, and Kaskus aimed to do multi-label toxicity classification."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="netifier_source",
            version=SOURCE_VERSION,
            description="Netifier source schema",
            schema="source",
            subset_id="netifier",
        ),
        NusantaraConfig(
            name="netifier_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="Netifier Nusantara schema",
            schema="nusantara_text_multi",
            subset_id="netifier",
        ),
    ]

    DEFAULT_CONFIG_NAME = "netifier_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "pornography": datasets.Value("bool"),
                    "blasphemy_racism_discrimination": datasets.Value("bool"),
                    "radicalism": datasets.Value("bool"),
                    "defamation": datasets.Value("bool"),
                }
            )
        elif self.config.schema == "nusantara_text_multi":
            features = schemas.text_multi_features([0, 1])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        train_data = Path(dl_manager.download(urls["train"]))
        test_data = Path(dl_manager.download(urls["test"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_data,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_data,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset does not have id, using row index as id
        label_cols = ["pornography", "blasphemy_racism_discrimination", "radicalism", "defamation"]
        df = pd.read_csv(filepath, encoding="ISO-8859-1").reset_index()
        df.columns = ["id", "original_text", "text"] + label_cols

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "text": row.text,
                }
                for label in label_cols:
                    ex[label] = getattr(row, label)
                yield row.id, ex

        elif self.config.schema == "nusantara_text_multi":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.text,
                    "labels": [label for label in row[4:]],
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
