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
This corpus is an attempt to recreate the dataset used for training XLM-R. This
corpus comprises of monolingual data for 100+ languages and also includes data
for romanized languages (indicated by *_rom). This was constructed using the
urls and paragraph indices provided by the CC-Net repository by processing
January-December 2018 Commoncrawl snapshots. Each file comprises of documents
separated by double-newlines and paragraphs within the same document separated
by a newline. The data is generated using the open source CC-Net repository. No
claims of intellectual property are made on the work of preparation of the
corpus.

This contains the Indonesian (ind), the Javanese (jav), and the Sundanese (sun) subset.

[nusantara_schema_name] = ssp
"""

from posixpath import split
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "cc100"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "jav", "sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_CITATION = """\
        @inproceedings{conneau-etal-2020-unsupervised,
    title = "Unsupervised Cross-lingual Representation Learning at Scale",
    author = "Conneau, Alexis  and
      Khandelwal, Kartikay  and
      Goyal, Naman  and
      Chaudhary, Vishrav  and
      Wenzek, Guillaume  and
      Guzm{'a}n, Francisco  and
      Grave, Edouard  and
      Ott, Myle  and
      Zettlemoyer, Luke  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.747",
    doi = "10.18653/v1/2020.acl-main.747",
    pages = "8440--8451",
    abstract = "This paper shows that pretraining multilingual language models
    at scale leads to significant performance gains for a wide range of
    cross-lingual transfer tasks. We train a Transformer-based masked language
    model on one hundred languages, using more than two terabytes of filtered
    CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms
    multilingual BERT (mBERT) on a variety of cross-lingual benchmarks,
    including +14.6{%} average accuracy on XNLI, +13{%} average F1 score on
    MLQA, and +2.4{%} F1 score on NER. XLM-R performs particularly well on
    low-resource languages, improving 15.7{%} in XNLI accuracy for Swahili and
    11.4{%} for Urdu over previous XLM models. We also present a detailed
    empirical analysis of the key factors that are required to achieve these
    gains, including the trade-offs between (1) positive transfer and capacity
    dilution and (2) the performance of high and low resource languages at
    scale. Finally, we show, for the first time, the possibility of
    multilingual modeling without sacrificing per-language performance; XLM-R
    is very competitive with strong monolingual models on the GLUE and XNLI
    benchmarks. We will make our code and models publicly available.",
}

@inproceedings{wenzek-etal-2020-ccnet,
    title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
    author = "Wenzek, Guillaume  and
      Lachaux, Marie-Anne  and
      Conneau, Alexis  and
      Chaudhary, Vishrav  and
      Guzm{'a}n, Francisco  and
      Joulin, Armand  and
      Grave, Edouard",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.494",
    pages = "4003--4012",
    abstract = "Pre-training text representations have led to significant
    improvements in many areas of natural language processing. The quality of
    these models benefits greatly from the size of the pretraining corpora as
    long as its quality is preserved. In this paper, we describe an automatic
    pipeline to extract massive high-quality monolingual datasets from Common
    Crawl for a variety of languages. Our pipeline follows the data processing
    introduced in fastText (Mikolov et al., 2017; Grave et al., 2018), that
    deduplicates documents and identifies their language. We augment this
    pipeline with a filtering step to select documents that are close to high
    quality corpora like Wikipedia.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DESCRIPTION = """\
        This corpus is an attempt to recreate the dataset used for training
        XLM-R. This corpus comprises of monolingual data for 100+ languages and
        also includes data for romanized languages (indicated by *_rom). This
        was constructed using the urls and paragraph indices provided by the
        CC-Net repository by processing January-December 2018 Commoncrawl
        snapshots. Each file comprises of documents separated by
        double-newlines and paragraphs within the same document separated by a
        newline. The data is generated using the open source CC-Net repository.
        No claims of intellectual property are made on the work of preparation
        of the corpus.
"""

_HOMEPAGE = "https://data.statmt.org/cc-100/"

_LICENSE = "MIT"

_LANGUAGES_MAP = {
    "ind": "id",
    "jav": "jv",
    "sun": "su",
}

_URLS = {
    "train": "https://data.statmt.org/cc-100/{lang}.txt.xz",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "2018.12.01"

_NUSANTARA_VERSION = "1.0.0"

def nusantara_config_constructor(lang, schema, version):
    """Construct NusantaraConfig with cc100_{lang}_{schema} as the name format."""
    if schema != "source" and schema != "nusantara_ssp":
        raise ValueError(f"Invalid schema: {schema}")

    if lang == "":
        raise ValueError(f"Language is required. Choose one of these languages: {_LANGUAGES}.")
    elif lang in _LANGUAGES:
        return NusantaraConfig(
            name=f"cc100_{lang}_{schema}",
            version=datasets.Version(version),
            description=f"CC100 with {schema} schema for {lang} language",
            schema=schema,
            subset_id="cc100",
        )
    else:
        raise ValueError(f"Invalid language: {lang}. Choose one of these languages: {_LANGUAGES}.")


class CC100(datasets.GeneratorBasedBuilder):
    """Monolingual Datasets from Web Crawl Data."""

    DEFAULT_CONFIG_NAME = "cc100_jav_source"

    BUILDER_CONFIGS = [
        nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES_MAP
    ] + [
        nusantara_config_constructor(lang, "nusantara_ssp", _NUSANTARA_VERSION) for lang in _LANGUAGES_MAP
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

        split_name = self.config.name.split("_")
        if split_name[1] == "source" or split_name[1] == "nusantara":
            lang = _DEFAULT_LANGUAGE
        else:
            lang = split_name[1]
        url = _URLS["train"].format(lang=_LANGUAGES_MAP[lang])
        path = dl_manager.download_and_extract(url)

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