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

from posixpath import split
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "indo4b_plus"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LOCAL = False
_LANGUAGES = ["ind", "sun", "jav"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_CITATION = """\
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
        pages = "8875--8898",
        abstract = "Natural language generation (NLG) benchmarks provide an important avenue to measure progress 
        and develop better NLG systems. Unfortunately, the lack of publicly available NLG benchmarks for low-resource 
        languages poses a challenging barrier for building NLG systems that work well for languages with limited 
        amounts of data. Here we introduce IndoNLG, the first benchmark to measure natural language generation (NLG)
        progress in three low-resource{---}yet widely spoken{---}languages of Indonesia: Indonesian, Javanese, and Sundanese. 
        Altogether, these languages are spoken by more than 100 million native speakers, and hence constitute an important 
        use case of NLG systems today. Concretely, IndoNLG covers six tasks: summarization, question answering, chit-chat, 
        and three different pairs of machine translation (MT) tasks. We collate a clean pretraining corpus of Indonesian, 
        Sundanese, and Javanese datasets, Indo4B-Plus, which is used to pretrain our models: IndoBART and IndoGPT. 
        We show that IndoBART and IndoGPT achieve competitive performance on all tasks{---}despite using only one-fifth
        the parameters of a larger multilingual model, mBART-large (Liu et al., 2020). This finding emphasizes 
        the importance of pretraining on closely related, localized languages to achieve more efficient learning and faster inference 
        at very low-resource languages like Javanese and Sundanese.",
    }
"""

_DESCRIPTION = """\
    Indo4B-Plus is an extension of Indo4B, a large-scale Indonesian self-supervised pre-training corpus. 
    Indo4B-Plus extend Indo4B by adding two low-resource Indonesian local languages to the corpus, i.e., Sundanese and Javanese.
    Indo4B-Plus adds 82,582,025 words (∼2.07%) of Sundanese sentences and 331,041,877 words (∼8.29%) of Javanese
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "CC0"

_LANGUAGES_MAP = {
    "ind": "id",
    "jav": "jv",
    "sun": "su",
}

_URLS = {
    "indo4b": "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt.zip",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class Indo4BPlus(datasets.GeneratorBasedBuilder):
    """Indo4B-Plus is a large-scale Indonesian self-supervised pre-training corpus consists
    of around 4B words, covering three languages, i.e., Indonesian, Sundanese, and Javanese."""

    DEFAULT_CONFIG_NAME = "indo4b_plus_source"

    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo4b_plus_source",
            version=_SOURCE_VERSION,
            description="Indo4B-Plus source schema",
            schema="source",
            subset_id="indo4b_plus",
        ),
        NusantaraConfig(
            name="indo4b_plus_nusantara_ssp",
            version=_NUSANTARA_VERSION,
            description="Indo4B-Plus Nusantara schema",
            schema="nusantara_ssp",
            subset_id="indo4b_plus",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        url = _URLS["indo4b"]
        path = dl_manager.download_and_extract(url) + "/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as f:
            if self.config.schema == "source":
                for counter, row in enumerate(f):
                    if row.strip() != "":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": row.strip(),
                            },
                        )
            elif self.config.schema == "nusantara_ssp":
                for counter, row in enumerate(f):
                    if row.strip() != "":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": row.strip(),
                            },
                        )