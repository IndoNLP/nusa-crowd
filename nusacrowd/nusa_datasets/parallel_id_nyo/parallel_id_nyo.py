from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (
    DEFAULT_NUSANTARA_VIEW_NAME,
    DEFAULT_SOURCE_VIEW_NAME,
    Tasks,
)

_DATASETNAME = "parallel_id_nyo"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LOCAL = False
_LANGUAGES = ["ind", "abl"]

_CITATION = """\
@article{Abidin_2021,
doi = {10.1088/1742-6596/1751/1/012036},
url = {https://dx.doi.org/10.1088/1742-6596/1751/1/012036},
year = {2021},
month = {jan},
publisher = {IOP Publishing},
volume = {1751},
number = {1},
pages = {012036},
author = {Z Abidin and  Permata and I Ahmad and  Rusliyawati},
title = {Effect of mono corpus quantity on statistical machine translation
Indonesian - Lampung dialect of nyo},
journal = {Journal of Physics: Conference Series},
abstract = {Lampung Province is located on the island of Sumatera. For the
immigrants in Lampung, they have difficulty in
communicating with the indigenous people of Lampung. As an alternative, both
immigrants and the indigenous people of Lampung speak Indonesian.
This research aims to build a language model from Indonesian language and a
 translation model from the Lampung language dialect of nyo, both models will
 be combined in a Moses decoder.
This research focuses on observing the effect of adding mono corpus to the
 experimental statistical machine translation of
 Indonesian - Lampung dialect of nyo.
This research uses 3000 pair parallel corpus in Indonesia language and
Lampung language dialect of nyo as source language
and uses 3000 mono corpus sentences in Lampung language
dialect of nyo as target language. The results showed that the accuracy
value in bilingual evalution under-study score when using 1000 sentences,
2000 sentences, 3000 sentences mono corpus
show the accuracy value of the bilingual evaluation under-study,
respectively, namely 40.97 %, 41.80 % and 45.26 %.}
}
"""

_DESCRIPTION = """\
Dataset that contains Indonesian - Lampung language pairs.

The original data should contains 3000 rows, unfortunately,
not all of the instances in the original data is aligned perfectly.
Thus, this data only have the aligned ones, which only contain 1727 pairs.
"""

_HOMEPAGE = "https://drive.google.com/drive/folders/1oNpybrq5OJ_4Ne0HS5w9eHqnZlZASpmC?usp=sharing"

_LICENSE = "Unknown"

# WARNING: Incomplete data!
_URLs = {
    "train": "https://raw.githubusercontent.com/haryoa/IndoData/main/data_ind_lampung_1729_line.csv"
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"

COL_INDONESIA = "indo"
COL_LAMPUNG = "lampung"


class ParallelIdNyo(datasets.GeneratorBasedBuilder):
    """Dataset that contains Indonesian - Lampung language pairs."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="parallel_id_nyo_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Parallel Id-Nyo source schema",
            schema="source",
            subset_id="parallel_id_nyo",
        ),
        NusantaraConfig(
            name="parallel_id_nyo_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Parallel Id-Nyo Nusantara schema",
            schema="nusantara_t2t",
            subset_id="parallel_id_nyo",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ted_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

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
        path = Path(dl_manager.download_and_extract(_URLs["train"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": path},
            )
        ]

    def _generate_examples(self, filepath: Path):

        df = pd.read_csv(filepath).reset_index()

        if self.config.schema == "source":
            for idx, row in df.iterrows():
                ex = {
                    "id": str(idx),
                    "text": str(row[COL_INDONESIA]).rstrip(),
                    "label": str(row[COL_LAMPUNG]).rstrip(),
                }
                yield idx, ex
        elif self.config.schema == "nusantara_t2t":
            for idx, row in df.iterrows():
                ex = {
                    "id": str(idx),
                    "text_1": str(row[COL_INDONESIA]).rstrip(),
                    "text_2": str(row[COL_LAMPUNG]).rstrip(),
                    "text_1_name": "ind",
                    "text_2_name": "abl",  # code name for lampung Nyo
                }
                yield idx, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


if __name__ == "__main__":
    datasets.load_dataset(__file__)
