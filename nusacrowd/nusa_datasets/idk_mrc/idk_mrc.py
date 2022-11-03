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

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@misc{putri2022idk,
    doi = {10.48550/ARXIV.2210.13778},
    url = {https://arxiv.org/abs/2210.13778},
    author = {Putri, Rifki Afina and Oh, Alice},
    title = {IDK-MRC: Unanswerable Questions for Indonesian Machine Reading Comprehension},
    publisher = {arXiv},
    year = {2022}
}

"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_ALL_DATASETS = ["idk_mrc", "trans_squad", "tydiqa", "model_gen", "human_filt"]
_DATASETNAME = _ALL_DATASETS[0]  # idk_mrc
_BASELINES = _ALL_DATASETS[1:]   # trans_squad, tydiqa, model_gen, human_filt

_DESCRIPTION = """\
I(n)dontKnow-MRC (IDK-MRC) is an Indonesian Machine Reading Comprehension dataset that covers
answerable and unanswerable questions. Based on the combination of the existing answerable questions in TyDiQA,
the new unanswerable question in IDK-MRC is generated using a question generation model and human-written question.
Each paragraph in the dataset has a set of answerable and unanswerable questions with the corresponding answer.

Besides IDK-MRC (idk_mrc) dataset, several baseline datasets also provided:
1. Trans SQuAD (trans_squad): machine translated SQuAD 2.0 (Muis and Purwarianti, 2020)
2. TyDiQA (tydiqa): Indonesian answerable questions set from the TyDiQA-GoldP (Clark et al., 2020)
3. Model Gen (model_gen): TyDiQA + the unanswerable questions output from the question generation model
4. Human Filt (human_filt): Model Gen dataset that has been filtered by human annotator
"""

_HOMEPAGE = "https://github.com/rifkiaputri/IDK-MRC"

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    _DATASETNAME: {
        "test": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/idk_mrc/test.json",
        "train": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/idk_mrc/train.json",
        "validation": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/idk_mrc/valid.json",
    },
    "baseline": {
        "test": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/baseline/{name}/test.json",
        "train": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/baseline/{name}/train.json",
        "validation": "https://raw.githubusercontent.com/rifkiaputri/IDK-MRC/master/dataset/baseline/{name}/valid.json",
    },
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(name, schema, version):
    """
    Construct NusantaraConfig with idk_mrc_{schema} format for the main dataset &
    idk_mrc_baseline_{name}_{schema} format for the baseline datasets.
    Suported dataset names: see _ALL_DATASETS
    """
    if schema != "source" and schema != "nusantara_qa":
        raise ValueError(f"Invalid schema: {schema}")

    if name not in _ALL_DATASETS:
        raise ValueError(f"Invalid dataset name: {name}")

    if name == "idk_mrc":
        return NusantaraConfig(
            name="idk_mrc_{schema}".format(schema=schema),
            version=datasets.Version(version),
            description="IDK-MRC with {schema} schema".format(schema=schema),
            schema=schema,
            subset_id="idk_mrc",
        )
    else:
        return NusantaraConfig(
            name="idk_mrc_baseline_{name}_{schema}".format(name=name, schema=schema),
            version=datasets.Version(version),
            description="IDK-MRC baseline ({name}) with {schema} schema".format(name=name, schema=schema),
            schema=schema,
            subset_id="idk_mrc",
        )


class IdkMrc(datasets.GeneratorBasedBuilder):
    """IDK-MRC is an Indonesian MRC dataset that covers answerable and unanswerable questions"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        nusantara_config_constructor(name, schema, version)
        for name in _ALL_DATASETS for schema, version in zip(["source", "nusantara_qa"], [_SOURCE_VERSION, _NUSANTARA_VERSION])
    ]

    DEFAULT_CONFIG_NAME = "idk_mrc_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "qas": [
                        {
                            "id": datasets.Value("string"),
                            "is_impossible": datasets.Value("bool"),
                            "question": datasets.Value("string"),
                            "answers": [
                                {
                                    "text": datasets.Value("string"),
                                    "answer_start": datasets.Value("int64")
                                }
                            ]
                        }
                    ],
                }
            )

        elif self.config.schema == "nusantara_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.name == "idk_mrc_source" or self.config.name == "idk_mrc_nusantara_qa":
            data_name = "idk_mrc"
            train_data_path = dl_manager.download_and_extract(_URLS[_DATASETNAME]["train"])
            validation_data_path = dl_manager.download_and_extract(_URLS[_DATASETNAME]["validation"])
            test_data_path = dl_manager.download_and_extract(_URLS[_DATASETNAME]["test"])
        else:
            try:
                data_name = re.search("baseline_(.+?)_(source|nusantara_qa)", self.config.name).group(1)
            except AttributeError:
                raise ValueError(f"Invalid config name: {self.config.name}")

            if data_name not in _BASELINES:
                raise ValueError(f"Invalid baseline dataset name: {data_name}")

            train_data_path = dl_manager.download_and_extract(_URLS["baseline"]["train"].format(name=data_name))
            validation_data_path = dl_manager.download_and_extract(_URLS["baseline"]["validation"].format(name=data_name))
            test_data_path = dl_manager.download_and_extract(_URLS["baseline"]["test"].format(name=data_name)) if data_name != "trans_squad" else ""

        data_split = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_data_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(validation_data_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(test_data_path),
                },
            ),
        ]

        if data_name == "trans_squad":
            # trans_squad doesn't have test split
            return data_split[:2]

        return data_split

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath) as json_file:
            examples = json.load(json_file)

        if self.config.schema == "source":
            # Dataset doesn't have predefined context ID, use python enumeration.
            # The examples in the source schema are grouped by paragraph context;
            # each context can have multiple questions.
            for key, example in enumerate(examples):
                yield key, example

        elif self.config.schema == "nusantara_qa":
            for key, example in enumerate(examples):
                for qa in example["qas"]:
                    # Use question ID as key
                    yield str(qa["id"]), {
                        "id": qa["id"],
                        "question_id": qa["id"],
                        "document_id": str(key),
                        "question": qa["question"],
                        "type": "extractive",
                        "choices": [],
                        "context": example["context"],
                        "answer": [ans["text"] for ans in qa["answers"]],
                    }
