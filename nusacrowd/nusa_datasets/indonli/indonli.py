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

"""
IndoNLI is the first human-elicited Natural Language Inference (NLI) dataset for Indonesian.
IndoNLI is annotated by both crowd workers and experts. The expert-annotated data is used exclusively as a test set.
It is designed to provide a challenging test-bed for Indonesian NLI by explicitly incorporating various linguistic
phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning.

The data is split across train, valid, test_lay, and test_expert.

A small subset of test_expert is used as a diasnostic tool. For more info, please visit https://github.com/ir-nlp-csui/indonli

The premise were collected from Indonesian Wikipedia and from other public Indonesian dataset: Indonesian PUD and GSD treebanks provided by the Universal Dependencies 2.5 and IndoSum

The data was produced by humans.

"""

from pathlib import Path
from typing import List

import datasets
import jsonlines

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{mahendra-etal-2021-indonli,
    title = "{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
    author = "Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.821",
    pages = "10511--10527",
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indonli"

_DESCRIPTION = """\
This dataset is designed for Natural Language Inference NLP task.  It is designed to provide a challenging test-bed
for Indonesian NLI by explicitly incorporating various linguistic phenomena such as numerical reasoning, structural
changes, idioms, or temporal and spatial reasoning.
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/indonli"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl",
        "valid": "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/val.jsonl",
        "test": "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test.jsonl",
    }
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.1.0"  # Mentioned in https://github.com/huggingface/datasets/blob/main/datasets/indonli/indonli.py

_NUSANTARA_VERSION = "1.0.0"


class IndoNli(datasets.GeneratorBasedBuilder):
    """IndoNLI, a human-elicited NLI dataset for Indonesian containing ~18k sentence pairs annotated by crowd workers."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indonli_source",
            version=SOURCE_VERSION,
            description="indonli source schema",
            schema="source",
            subset_id="indonli",
        ),
        NusantaraConfig(
            name="indonli_nusantara_pairs",
            version=NUSANTARA_VERSION,
            description="indonli Nusantara schema",
            schema="nusantara_pairs",
            subset_id="indonli",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indonli_source"
    labels = ["c", "e", "n"]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pair_id": datasets.Value("int32"),
                    "premise_id": datasets.Value("int32"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "annotator_type": datasets.Value("string"),
                    "sentence_size": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_pairs":
            features = schemas.pairs_features(self.labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        train_data_path = Path(dl_manager.download_and_extract(urls["train"]))
        valid_data_path = Path(dl_manager.download_and_extract(urls["valid"]))
        test_data_path = Path(dl_manager.download_and_extract(urls["test"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_data_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": valid_data_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path},
            ),
        ]

    def _generate_examples(self, filepath: Path):

        if self.config.schema == "source":
            print(filepath)
            with jsonlines.open(filepath) as f:
                skip = []  # To avoid duplicate IDs
                for example in f.iter():
                    if example["pair_id"] not in skip:
                        skip.append(example["pair_id"])
                        example = {
                            "pair_id": example["pair_id"],
                            "premise_id": example["premise_id"],
                            "premise": example["premise"],
                            "hypothesis": example["hypothesis"],
                            "annotator_type": example["annotator_type"],
                            "sentence_size": example["sentence_size"],
                            "label": example["label"],
                        }
                        yield example["pair_id"], example

        elif self.config.schema == "nusantara_pairs":
            print(filepath)
            with jsonlines.open(filepath) as f:
                skip = []  # To avoid duplicate IDs
                for example in f.iter():
                    if example["pair_id"] not in skip:
                        skip.append(example["pair_id"])
                        nu_eg = {"id": str(example["pair_id"]), "text_1": example["premise"], "text_2": example["hypothesis"], "label": example["label"]}
                        yield example["pair_id"], nu_eg
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
