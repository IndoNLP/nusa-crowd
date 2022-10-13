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
import glob

_DATASETNAME = "indo4b"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_CITATION = """\
    @inproceedings{wilie-etal-2020-indonlu,
        title = "{I}ndo{NLU}: Benchmark and Resources for Evaluating {I}ndonesian 
            Natural Language Understanding",
        author = "Wilie, Bryan  and
          Vincentio, Karissa  and
          Winata, Genta Indra  and
          Cahyawijaya, Samuel  and
          Li, Xiaohong  and
          Lim, Zhi Yuan  and
          Soleman, Sidik  and
          Mahendra, Rahmad  and
          Fung, Pascale  and
          Bahar, Syafri  and
          Purwarianti, Ayu",
        booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the 
                Association for Computational Linguistics and the 10th International Joint 
                Conference on Natural Language Processing",
        month = dec,
        year = "2020",
        address = "Suzhou, China",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.aacl-main.85",
        pages = "843--857",
        abstract = "Although Indonesian is known to be the fourth most frequently used language 
            over the internet, the research progress on this language in natural language processing (NLP) 
            is slow-moving due to a lack of available resources. In response, we introduce the first-ever vast 
            resource for training, evaluation, and benchmarking on Indonesian natural language understanding 
            (IndoNLU) tasks. IndoNLU includes twelve tasks, ranging from single sentence classification to 
            pair-sentences sequence labeling with different levels of complexity. The datasets for the tasks 
            lie in different domains and styles to ensure task diversity. We also provide a set of Indonesian 
            pre-trained models (IndoBERT) trained from a large and clean Indonesian dataset (Indo4B) collected 
            from publicly available sources such as social media texts, blogs, news, and websites. 
            We release baseline models for all twelve tasks, as well as the framework for benchmark evaluation, 
            thus enabling everyone to benchmark their system performances.",
    }
"""

_DESCRIPTION = """\
    Indo4B is a large-scale Indonesian self-supervised pre-training corpus
    consists of around 3.6B words, with around 250M sentences. The corpus
    covers both formal and colloquial Indonesian sentences compiled from 
    12 sources, of which two cover Indonesian colloquial language, eight
    cover formal Indonesian language, and the rest have a mixed style of
    both colloquial and formal.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "CC0"

_LANGUAGES_MAP = {
    "ind": "id",
    "jav": "jv",
    "sun": "su",
}

_URLS = {
    "indo4b": "https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_wot_uncased_blanklines.tar.xz",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class Indo4B(datasets.GeneratorBasedBuilder):
    """Indo4B is a large-scale Indonesian self-supervised pre-training corpus
    consists of around 3.6B words, with around 250M sentences."""

    DEFAULT_CONFIG_NAME = "indo4b_source"

    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo4b_source",
            version=_SOURCE_VERSION,
            description="Indo4B source schema",
            schema="source",
            subset_id="indo4b",
        ),
        NusantaraConfig(
            name="indo4b_nusantara_ssp",
            version=_NUSANTARA_VERSION,
            description="Indo4B Nusantara schema",
            schema="nusantara_ssp",
            subset_id="indo4b",
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
        path = dl_manager.download_and_extract(url) + "/processed_uncased_blanklines"

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

        counter = 0
        for txt_path in glob.glob(f'{filepath}/*.txt'):
            with open(txt_path, encoding="utf-8") as f:
                if self.config.schema == "source":
                    for row in f:
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text": row.strip(),
                                },
                            )
                            counter += 1
                elif self.config.schema == "nusantara_ssp":
                    for row in f:
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text": row.strip(),
                                },
                            )
                            counter += 1