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
""" NERGrit Dataset """

from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@misc{Fahmi_NERGRIT_CORPUS_2019,
author = {Fahmi, Husni and Wibisono, Yudi and Kusumawati, Riyanti},
title = {{NERGRIT CORPUS}},
url = {https://github.com/grit-id/nergrit-corpus},
year = {2019}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "nergrit"
_DESCRIPTION = """\
Nergrit Corpus is a dataset collection of Indonesian Named Entity Recognition (NER), Statement Extraction,
and Sentiment Analysis developed by PT Gria Inovasi Teknologi (GRIT).
The Named Entity Recognition contains 18 entities as follow:
    'CRD': Cardinal
    'DAT': Date
    'EVT': Event
    'FAC': Facility
    'GPE': Geopolitical Entity
    'LAW': Law Entity (such as Undang-Undang)
    'LOC': Location
    'MON': Money
    'NOR': Political Organization
    'ORD': Ordinal
    'ORG': Organization
    'PER': Person
    'PRC': Percent
    'PRD': Product
    'QTY': Quantity
    'REG': Religion
    'TIM': Time
    'WOA': Work of Art
    'LAN': Language
"""

_HOMEPAGE = "https://github.com/grit-id/nergrit-corpus"
_LICENSE = "MIT"
_URL = "https://github.com/cahya-wirawan/indonesian-language-models/raw/master/data/nergrit-corpus_20190726_corrected.tgz"
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class NergritDataset(datasets.GeneratorBasedBuilder):
    """Indonesian Named Entity Recognition from https://github.com/grit-id/nergrit-corpus."""

    label_classes = {
        "ner": [
            "B-CRD",
            "B-DAT",
            "B-EVT",
            "B-FAC",
            "B-GPE",
            "B-LAN",
            "B-LAW",
            "B-LOC",
            "B-MON",
            "B-NOR",
            "B-ORD",
            "B-ORG",
            "B-PER",
            "B-PRC",
            "B-PRD",
            "B-QTY",
            "B-REG",
            "B-TIM",
            "B-WOA",
            "I-CRD",
            "I-DAT",
            "I-EVT",
            "I-FAC",
            "I-GPE",
            "I-LAN",
            "I-LAW",
            "I-LOC",
            "I-MON",
            "I-NOR",
            "I-ORD",
            "I-ORG",
            "I-PER",
            "I-PRC",
            "I-PRD",
            "I-QTY",
            "I-REG",
            "I-TIM",
            "I-WOA",
            "O",
        ],
        "sentiment": ["B-POS", "B-NEG", "B-NET", "I-POS", "I-NEG", "I-NET", "O"],
        "statement": ["B-BREL", "B-FREL", "B-STAT", "B-WHO", "I-BREL", "I-FREL", "I-STAT", "I-WHO", "O"],
    }
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"nergrit_{task}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="NERGrit source schema",
            schema="source",
            subset_id=f"nergrit_{task}",
        )
        for task in label_classes
    ]
    BUILDER_CONFIGS += [
        NusantaraConfig(
            name=f"nergrit_{task}_nusantara_seq_label",
            version=datasets.Version(_SOURCE_VERSION),
            description="NERGrit Nusantara schema",
            schema="nusantara_seq_label",
            subset_id=f"nergrit_{task}",
        )
        for task in label_classes
    ]

    DEFAULT_CONFIG_NAME = "nergrit_ner_source"

    def _info(self):
        features = None
        task = self.config.subset_id.split("_")[-1]
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "ner_tag": [datasets.Value("string")]})
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.label_classes[task])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        task = self.config.subset_id.split("_")[-1]
        archive = Path(dl_manager.download_and_extract(_URL))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": archive / f"nergrit-corpus/{task}/data/train_corrected.txt"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": archive / f"nergrit-corpus/{task}/data/test_corrected.txt"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": archive / f"nergrit-corpus/{task}/data/valid_corrected.txt"},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        conll_dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for index, row in enumerate(conll_dataset):
                ex = {"index": str(index), "tokens": row["sentence"], "ner_tag": row["label"]}
                yield index, ex
        elif self.config.schema == "nusantara_seq_label":
            for index, row in enumerate(conll_dataset):
                ex = {"id": str(index), "tokens": row["sentence"], "labels": row["label"]}
                yield index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
