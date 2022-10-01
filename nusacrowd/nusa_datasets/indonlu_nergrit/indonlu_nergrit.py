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
""" IndoNLU NERGrit Dataset """

from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
@online{nergrit2019,
  title={NERGrit Corpus},
  author={NERGrit Developers},
  year={2019},
  url={https://github.com/grit-id/nergrit-corpus}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indonlu_nergrit"
_DESCRIPTION = """\
This NER dataset is taken from the Grit-ID repository, and the labels are spans in IOB chunking representation.
The dataset consists of three kinds of named entity tags, PERSON (name of person), PLACE (name of location), and
ORGANIZATION (name of organization).
"""

_HOMEPAGE = "https://github.com/grit-id/nergrit-corpus"
_LICENSE = "MIT"
_URL_ROOT = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit"
_URLs = {
    "train": f"{_URL_ROOT}/train_preprocess.txt",
    "validation": f"{_URL_ROOT}/valid_preprocess.txt",
    "test": f"{_URL_ROOT}/test_preprocess.txt",
}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndonluNergritDataset(datasets.GeneratorBasedBuilder):
    """Indonesian Named Entity Recognition from https://github.com/grit-id/nergrit-corpus."""

    label_classes = ["I-PERSON", "B-ORGANISATION", "I-ORGANISATION", "B-PLACE", "I-PLACE", "O", "B-PERSON"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indonlu_nergrit_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="IndoNLU NERGrit source schema",
            schema="source",
            subset_id="indonlu_nergrit",
        ),
        NusantaraConfig(
            name="indonlu_nergrit_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="IndoNLU NERGrit Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="indonlu_nergrit",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indonlu_nergrit_source"

    def _info(self):
        features = None
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
