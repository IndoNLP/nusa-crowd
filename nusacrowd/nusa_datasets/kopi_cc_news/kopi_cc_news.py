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
KoPI-CC_NEWS corpus

[nusantara_schema_name] = ssp
"""

import json
from typing import List

import datasets
import zstandard as zstd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "kopi_cc_news"
_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME
_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_URL = "https://commoncrawl.org/"
_CITATION = """\

"""

_DESCRIPTION = """\
    KoPI(Korpus Perayapan Indonesia)-CC_News is Indonesian Only Extract from CC NEWS Common Crawl from 2016-2022(july) ,each snapshots get extracted using warcio,trafilatura and filter using fasttext
"""

_HOMEPAGE = "https://huggingface.co/datasets/munggok/KoPI-CC_News"

_LICENSE = "CC0"

_URLS = "https://huggingface.co/datasets/munggok/KoPI-CC_News/resolve/main/data/cc_news_{year}_id.jsonl.zst"

_YEAR = ["2016", "2017", "2018", "2019", "2020", "2021", "2022"]

_ALL_CONFIG = _YEAR + ["all"]

_SOURCE_VERSION = "2018.12.01"

_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(year, schema, version):
    """Construct NusantaraConfig"""
    if schema != "source" and schema != "nusantara_ssp":
        raise ValueError(f"Invalid schema: {schema}")

    if year == "":
        raise ValueError(f"Snapshot is required. Choose one of these Snapshot: {_ALL_CONFIG}.")
    elif year in _ALL_CONFIG:
        return NusantaraConfig(
            name=f"{_DATASETNAME}_{year}_{schema}",
            version=datasets.Version(version),
            description=f"KoPI-CC_News with {schema} schema for {year}",
            schema=schema,
            subset_id="kopi_cc_news",
        )
    else:
        raise ValueError(f"Invalid language: {year}. Choose one of these snapshots: {_ALL_CONFIG}.")


class KoPICCNEWS(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "2016"

    BUILDER_CONFIGS = [nusantara_config_constructor(sn, "source", _SOURCE_VERSION) for sn in _ALL_CONFIG] + [nusantara_config_constructor(sn, "nusantara_ssp", _NUSANTARA_VERSION) for sn in _ALL_CONFIG]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "meta": datasets.Value("string"),
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
        name = self.config.name.replace("_" + self.config.schema, "")
        name = name.replace(_DATASETNAME + "_", "")
        if name == "all":
            urls = [_URLS.format(year=m) for m in _YEAR]
        else:
            urls = [_URLS.format(year=name)]
        path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": path, "split": "train"},
            ),
        ]

    def _generate_examples(self, filepaths, split):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)
                        if self.config.schema == "nusantara_ssp":
                            yield id_, {"id": str(id_), "text": example["text"]}
                            id_ += 1
                        else:
                            yield id_, {"text": example["text"], "url": example["url"], "timestamp": example["timestamp"], "meta": example["meta"]}
                            id_ += 1
