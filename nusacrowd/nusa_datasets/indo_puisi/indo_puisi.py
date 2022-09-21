from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "indo_puisi"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_CITATION = """
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DESCRIPTION = """\
Puisi is an Indonesian poetic form. The dataset was collected by scraping various websites. It contains 7223 Indonesian puisi along with the title and author.
"""

_HOMEPAGE = "https://github.com/ilhamfp/puisi-pantun-generator"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/ilhamfp/puisi-pantun-generator/main/data/puisi.csv",
}


class IndoPuisi(datasets.GeneratorBasedBuilder):
    """IndoPuisi contains 7223 Indonesian puisi along with the title and author."""

    BUILDER_CONFIGS = (
        NusantaraConfig(
            name="indo_puisi_source",
            version=_SOURCE_VERSION,
            description="Indo puisi source schema",
            schema="source",
            subset_id="indo_puisi",
        ),
        NusantaraConfig(
            name="indo_puisi_nusantara_ssp",
            version=_NUSANTARA_VERSION,
            description="Indo puisi Nusantara schema",
            schema="nusantara_ssp",
            subset_id="indo_puisi",
        ),
    )

    DEFAULT_CONFIG_NAME = "indo_puisi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "puisi": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "author": datasets.Value("string"),
                    "puisi_with_header": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_ssp":
            features = schemas.self_supervised_pretraining.features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_csv_path = Path(dl_manager.download(_URLS["train"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_csv_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema != "source" and self.config.schema != "nusantara_ssp":
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        df = pd.read_csv(filepath).reset_index()
        if self.config.name == "indo_puisi_source":
            for row in df.itertuples():
                ex = {
                    "id": str(row.index),
                    "puisi": str(row.puisi).rstrip(),
                    "title": row.title,
                    "author": row.author,
                    "puisi_with_header": str(row.puisi_with_header).rstrip(),
                }
                yield row.index, ex

        elif self.config.name == "indo_puisi_nusantara_ssp":
            for row in df.itertuples():
                ex = {"id": str(row.index), "text": str(row.puisi).rstrip()}
                yield row.index, ex
