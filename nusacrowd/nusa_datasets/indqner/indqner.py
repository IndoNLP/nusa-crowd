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
""" IndQNER Dataset """

from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@misc{,
author = {Ria Hari Gusmita, Asep Fajar Firmansyah, Khodijah Khuliyah},
title = {{IndQNER: a NER Benchmark Dataset on Indonesian Translation of Quran}},
url = {https://github.com/dice-group/IndQNER},
year = {2022}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]
_DATASETNAME = "IndQNER"
_DESCRIPTION = """\
IndQNER is a NER dataset created by manually annotating the Indonesian translation of Quran text.
The dataset contains 18 named entity categories as follow:
    "Allah": Allah (including synonim of Allah such as Yang maha mengetahui lagi mahabijaksana)
    "Throne": Throne of Allah (such as 'Arasy)
    "Artifact": Artifact (such as Ka'bah, Baitullah)
    "AstronomicalBody": Astronomical body (such as bumi, matahari)
    "Event": Event (such as hari akhir, kiamat)
    "HolyBook": Holy book (such as AlQur'an)
    "Language": Language (such as bahasa Arab
    "Angel": Angel (such as Jibril, Mikail)
    "Person": Person (such as Bani Israil, Fir'aun)
    "Messenger": Messenger (such as Isa, Muhammad, Musa)
    "Prophet": Prophet (such as Adam, Sulaiman)
    "AfterlifeLocation": Afterlife location (such as Jahanam, Jahim, Padang Mahsyar)
    "GeographicalLocation": Geographical location (such as Sinai, negeru Babilonia)
    "Color": Color (such as kuning tua)
    "Religion": Religion (such as Islam, Yahudi, Nasrani)
    "Food": Food (such as manna, salwa)
"""

_HOMEPAGE = "https://github.com/dice-group/IndQNER"
_LICENSE = "Unknown"
_URLs = {
    "train": "https://raw.githubusercontent.com/dice-group/IndQNER/master/datasets/train.txt",
    "validation": "https://raw.githubusercontent.com/dice-group/IndQNER/master/datasets/dev.txt",
    "test": "https://raw.githubusercontent.com/dice-group/IndQNER/master/datasets/test.txt",
}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndqnerDataset(datasets.GeneratorBasedBuilder):
    """IndQNER is an Named Entity Recognition benchmark dataset on a niche domain i.e. Indonesian Translation of Quran."""

    label_classes = [
        "B-Allah",
        "B-Throne",
        "B-Artifact",
        "B-AstronomicalBody",
        "B-Event",
        "B-HolyBook",
        "B-Language",
        "B-Angel",
        "B-Person",
        "B-Messenger",
        "B-Prophet",
        "B-AfterlifeLocation",
        "B-GeographicalLocation",
        "B-Color",
        "B-Religion",
        "B-Food",
        "I-Allah",
        "I-Throne",
        "I-Artifact",
        "I-AstronomicalBody",
        "I-Event",
        "I-HolyBook",
        "I-Language",
        "I-Angel",
        "I-Person",
        "I-Messenger",
        "I-Prophet",
        "I-AfterlifeLocation",
        "I-GeographicalLocation",
        "I-Color",
        "I-Religion",
        "I-Food",
        "O",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indqner_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="NER dataset from Indonesian translation Quran source schema",
            schema="source",
            subset_id="indqner",
        ),
        NusantaraConfig(
            name="indqner_nusantara_seq_label",
            version=datasets.Version(_SOURCE_VERSION),
            description="NER dataset from Indonesian translation Quran Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="indqner",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indqner_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "ner_tag": [datasets.Value("string")]})
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.label_classes)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        validation_tsv_path = Path(dl_manager.download_and_extract(_URLs["validation"]))
        test_tsv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "train": train_tsv_path,
            "validation": validation_tsv_path,
            "test": test_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
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
