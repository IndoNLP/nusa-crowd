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

from pathlib import Path
from typing import List

import datasets
import json

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{clark-etal-2020-tydi,
    title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
    author = "Clark, Jonathan H.  and
      Choi, Eunsol  and
      Collins, Michael  and
      Garrette, Dan  and
      Kwiatkowski, Tom  and
      Nikolaev, Vitaly  and
      Palomaki, Jennimaria",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.30",
    doi = "10.1162/tacl_a_00317",
    pages = "454--470",
}

@inproceedings{cahyawijaya-etal-2021-indonlg,
    title = "{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation",
    author = "Cahyawijaya, Samuel  and
      Winata, Genta Indra  and
      Wilie, Bryan  and
      Vincentio, Karissa  and
      Li, Xiaohong  and
      Kuncoro, Adhiguna  and
      Ruder, Sebastian  and
      Lim, Zhi Yuan  and
      Bahar, Syafri  and
      Khodra, Masayu  and
      Purwarianti, Ayu  and
      Fung, Pascale",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.699",
    doi = "10.18653/v1/2021.emnlp-main.699",
    pages = "8875--8898"
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "tydiqa_id"

_DESCRIPTION = """\
TyDiQA dataset is collected from Wikipedia articles with human-annotated question and answer pairs covering 11 languages. 
The question-answer pairs are collected for each language without using translation services.
IndoNLG uses the Indonesian data from the secondary Gold passage task of the original TyDiQA dataset and
randomly split off 15% of the training data and use it as the test set.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlg"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip"
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class TyDiQAIdDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="tydiqa_id_source",
            version=SOURCE_VERSION,
            description="TyDiQA Id source schema",
            schema="source",
            subset_id="tydiqa_id",
        ),
        NusantaraConfig(
            name="tydiqa_id_nusantara_qa",
            version=NUSANTARA_VERSION,
            description="TyDiQA Id Nusantara schema",
            schema="nusantara_qa",
            subset_id="tydiqa_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "tydiqa_id_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "label": datasets.Value("string")
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
        url = _URLS[_DATASETNAME]
        base_path = Path(dl_manager.download_and_extract(url))
        train_data_path = base_path / "IndoNLG_downstream_tasks" / "question_answering" / "train_preprocess.json"
        valid_data_path = base_path / "IndoNLG_downstream_tasks" / "question_answering" / "valid_preprocess.json"
        test_data_path = base_path / "IndoNLG_downstream_tasks" / "question_answering" / "test_preprocess.json"

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
            )
        ]

    def _generate_examples(self, filepath: Path):
        if self.config.schema == "source":
            for example in json.load(open(filepath, 'r')):
                yield example["id"], example
        elif self.config.schema == "nusantara_qa":
            for example in json.load(open(filepath, 'r')):
                yield example["id"], {
                    "id": example['id'],
                    "question_id": example['id'],
                    "document_id": example['id'],
                    "question": example['question'],
                    "type": 'abstractive',
                    "choices": [],
                    "context": example['context'],
                    "answer": [example['label']]
                }
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
