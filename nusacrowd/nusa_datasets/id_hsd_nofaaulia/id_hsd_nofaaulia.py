# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_CITATION = """\
@inproceedings{10.1145/3330482.3330491,
author = {Aulia, Nofa and Budi, Indra},
title = {Hate Speech Detection on Indonesian Long Text Documents Using Machine Learning Approach},
year = {2019},
isbn = {9781450361064},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3330482.3330491},
doi = {10.1145/3330482.3330491},
abstract = {Due to the growth of hate speech on social media in recent years, it is important to understand this issue. An automatic hate speech detection system is needed to help to counter this problem. There have been many studies on detecting hate speech in short documents like Twitter data. But to our knowledge, research on long documents is rare, we suppose that the difficulty is increasing due to the possibility of the message of the text may be hidden. In this research, we explore in detecting hate speech on Indonesian long documents using machine learning approach. We build a new Indonesian hate speech dataset from Facebook. The experiment showed that the best performance obtained by Support Vector Machine (SVM) as its classifier algorithm using TF-IDF, char quad-gram, word unigram, and lexicon features that yield f1-score of 85%.},
booktitle = {Proceedings of the 2019 5th International Conference on Computing and Artificial Intelligence},
pages = {164–169},
numpages = {6},
keywords = {machine learning, SVM, long documents, hate speech detection},
location = {Bali, Indonesia},
series = {ICCAI '19}
}
"""
_DATASETNAME = "id_hsd_nofaaulia"
_LANGUAGES = ["ind"]
_LOCAL = False

_DESCRIPTION = """\
There have been many studies on detecting hate speech in short documents like Twitter data. But to our knowledge, research on long documents is rare, we suppose that the difficulty is increasing due to the possibility of the message of the text may be hidden. In this research, we explore in detecting hate speech on Indonesian long documents using machine learning approach. We build a new Indonesian hate speech dataset from Facebook.
"""

_HOMEPAGE = "https://dl.acm.org/doi/10.1145/3330482.3330491"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/nofaulia/hate-speech-detection/main/data/dataset.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"

class IdHSDNofaaulia(datasets.GeneratorBasedBuilder):
    """Indonesian hate speech detection for long article."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_hsd_nofaaulia_source",
            version=SOURCE_VERSION,
            description="id_hsd_nofaaulia source schema",
            schema="source",
            subset_id="id_hsd_nofaaulia",
        ),
        NusantaraConfig(
            name="id_hsd_nofaaulia_nusantara_text",
            version=NUSANTARA_VERSION,
            description="id_hsd_nofaaulia Nusantara schema",
            schema="nusantara_text",
            subset_id="id_hsd_nofaaulia",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_hsd_nofaaulia_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["0", "1"])

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
        data_dir = Path(dl_manager.download_and_extract(urls))

        data_files = {"train": os.path.join(data_dir.parent, "train.csv"), "test": os.path.join(data_dir.parent, "test.csv")}

        df = pd.read_csv(data_dir)
        df = df.dropna(axis=0)

        features = [c for c in df.columns.values if c not in []]
        target = "label"

        # The split follows the implementation below
        # https://github.com/IndoNLP/nusa-crowd/blob/master/nusantara/utils/schemas/pairs.py
        # test_size=0.1, random_state=42
        # tested locally using :
        # scikit-learn 1.1.2
        # python 3.10.4
        x_train, x_test, y_train, y_test = train_test_split(df[features].replace(r'\s+|\\n', ' ', regex=True), df[target], test_size=0.1, random_state=42)

        x_train.to_csv(data_files["train"], index=False)
        x_test.to_csv(data_files["test"], index=False)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset Split: 816 train, 0 validation, 91 test

        df = pd.read_csv(filepath).reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "index": row.Index,
                    "text": row.text,
                    "label": row.label,
                }
                yield row.Index, ex

        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.Index),
                    "text": row.text,
                    "label": row.label,
                }
                yield row.Index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
