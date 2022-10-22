from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import pandas as pd


_CITATION = """\
@INPROCEEDINGS{8629151,
author={Aliyah Salsabila, Nikmatun and Ardhito Winatmoko, Yosef and Akbar Septiandri, Ali and Jamal, Ade},
booktitle={2018 International Conference on Asian Language Processing (IALP)},
title={Colloquial Indonesian Lexicon},
year={2018},
volume={},
number={},
pages={226-229},
doi={10.1109/IALP.2018.8629151}}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "kamus_alay"

_DESCRIPTION = """\
Kamus Alay provide a lexicon for text normalization of Indonesian colloquial words.
It contains 3,592 unique colloquial words-also known as “bahasa alay” -and manually annotated them
with the normalized form. We built this lexicon from Instagram comments provided by Septiandri & Wibisono (2017)
"""

_HOMEPAGE = "https://ieeexplore.ieee.org/abstract/document/8629151"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv",
}

_SUPPORTED_TASKS = [Tasks.MORPHOLOGICAL_INFLECTION]

# Dataset does not have versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class KamusAlay(datasets.GeneratorBasedBuilder):
    """Kamus Alay is a dataset of lexicon for text normalization of Indonesian colloquial word"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    label_classes = [
        "abreviasi",
        "afiksasi",
        "akronim",
        "anaptiksis",
        "coinage",
        "elongasi",
        "homofon",
        "metatesis",
        "modifikasi vokal",
        "monoftongisasi",
        "naturalisasi",
        "pungtuasi",
        "reduplikasi",
        "salah ketik",
        "subtitusi",
        "word-value letter",
        "zeroisasi",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="kamus_alay_source",
            version=SOURCE_VERSION,
            description="Kamus Alay source schema",
            schema="source",
            subset_id="kamus_alay",
        ),
        NusantaraConfig(
            name="kamus_alay_nusantara_pairs_multi",
            version=NUSANTARA_VERSION,
            description="Kamus Alay Nusantara schema",
            schema="nusantara_pairs_multi",
            subset_id="kamus_alay",
        ),
    ]

    DEFAULT_CONFIG_NAME = "kamus_alay_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "slang": datasets.Value("string"),
                    "formal": datasets.Value("string"),
                    "in_dictionary": datasets.Value("bool"),
                    "context": datasets.Value("string"),
                    "categories": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "nusantara_pairs_multi":
            features = schemas.pairs_multi_features(self.label_classes)

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

        data_dir = Path(dl_manager.download(urls))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset does not have id, using row index as id
        df = pd.read_csv(filepath, encoding="ISO-8859-1").reset_index()
        df.columns = ["id", "slang", "formal", "is_in_dictionary", "example", "category1", "category2", "category3"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "slang": row.slang,
                    "formal": row.formal,
                    "in_dictionary": row.is_in_dictionary,
                    "context": row.example,
                    "categories": [c for c in (row.category1, row.category2, row.category3) if c != "0"],
                }
                yield row.id, ex

        elif self.config.schema == "nusantara_pairs_multi":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text_1": row.formal,
                    "text_2": row.slang,
                    "label": [c for c in (row.category1, row.category2, row.category3) if c != "0"],
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
