from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "indo_puisi"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_CITATION = """
"""

_DESCRIPTION = """\
Puisi is an Indonesian poetic form. The dataset was collected by scraping various websites. It contains 7223 Indonesian puisi along with the title and author.
"""

_HOMEPAGE = "https://github.com/ilhamfp/puisi-pantun-generator"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_SUPPORTED_TASKS = [Tasks.SSP]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_URLS = {
    "train": "https://github.com/ilhamfp/puisi-pantun-generator/blob/main/data/puisi.csv",
}


class IndoPuisi(datasets.GeneratorBasedBuilder):
    """IndoPuisi contains 7223 Indonesian puisi along with the title and author."""

    BUILDER_CONFIGS = (
        [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in LANGUAGES_MAP]
    )

    DEFAULT_CONFIG_NAME = "indo_puisi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["negative", "neutral", "positive"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.name == "nusax_senti_source" or self.config.name == "nusax_senti_nusantara_text":
            # Load all 12 languages
            train_csv_path = dl_manager.download_and_extract([_URLS["train"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
            validation_csv_path = dl_manager.download_and_extract([_URLS["validation"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
            test_csv_path = dl_manager.download_and_extract([_URLS["test"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
        else:
            lang = self.config.name[12:15]
            train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"].format(lang=LANGUAGES_MAP[lang])))
            validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"].format(lang=LANGUAGES_MAP[lang])))
            test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"].format(lang=LANGUAGES_MAP[lang])))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_csv_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": validation_csv_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_csv_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema != "source" and self.config.schema != "nusantara_text":
            raise ValueError(f"Invalid config: {self.config.name}")

        if self.config.name == "nusax_senti_source" or self.config.name == "nusax_senti_nusantara_text":
            ldf = []
            for fp in filepath:
                ldf.append(pd.read_csv(fp))
            df = pd.concat(ldf, axis=0, ignore_index=True).reset_index()
            # Have to use index instead of id to avoid duplicated key
            df = df.drop(columns=["id"]).rename(columns={"index": "id"})
        else:
            df = pd.read_csv(filepath).reset_index()

        for row in df.itertuples():
            ex = {"id": str(row.id), "text": row.text, "label": row.label}
            yield row.id, ex
