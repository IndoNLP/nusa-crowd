# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""KoPI-NLLB corpus."""
import json

import datasets
import zstandard as zstd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

logger = datasets.logging.get_logger(__name__)

_CITATION = """

Hefferman et al, Bitext Mining Using Distilled Sentence Representations for Low-Resource Languages. Arxiv https://arxiv.org/abs/2205.12654, 2022.
NLLB Team et al, No Language Left Behind: Scaling Human-Centered Machine Translation, Arxiv https://arxiv.org/abs/2207.04672, 2022.

"""
_DESCRIPTION = """\

KopI(Korpus Perayapan Indonesia)-NLLB, is Indonesian family language(aceh,bali,banjar,indonesia,jawa,minang,sunda) only extracted from NLLB Dataset, allenai/nllb

each language set also filtered using some some deduplicate technique such as exact hash(md5) dedup technique and minhash LSH neardup

"""
_TYPE = ["raw", "dedup", "neardup"]


_CONF_LANG = ["ace_Latn", "ban_Latn", "bjn_Latn", "ind_Latn", "jav_Latn", "min_Latn", "sun_Latn"]

_CONFIGS = []
for j in _CONF_LANG:
    for m in _TYPE:
        _CONFIGS.append(j + "-" + m)

_ALL_CONFIG = ["all-raw", "all-dedup", "all-neardup"] + _CONFIGS

_HOMEPAGE = "https://huggingface.co/datasets/munggok/KoPI-NLLB"

_LICENSE = "ODC_C"

_BASE_URL = "https://huggingface.co/datasets/munggok/KoPI-NLLB/resolve/main/{tipe}/{lang}.json.zst"

_DATASETNAME = "kopi_nllb"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_LANGUAGES = ["ind", "jav", "ace", "ban", "bjn", "min", "sun"]

_NUSANTARA_VERSION = "1.0.0"

_SOURCE_VERSION = "2022.09.13"

_LOCAL = False

_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME

_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_URL = "https://huggingface.co/datasets/allenai/nllb"


def nusantara_config_constructor(lang, schema, version):
    """Construct NusantaraConfig"""
    if schema != "source" and schema != "nusantara_ssp":
        raise ValueError(f"Invalid schema: {schema}")

    if lang == "":
        raise ValueError(f"Snapshot is required. Choose one of these Snapshot: {_ALL_CONFIG}.")
    elif lang in _ALL_CONFIG:
        return NusantaraConfig(
            name=f"{_DATASETNAME}_{lang}_{schema}",
            version=datasets.Version(version),
            description=f"KoPI-NLLB with {schema} schema for {lang}",
            schema=schema,
            subset_id="kopi_nllb",
        )
    else:
        raise ValueError(f"Invalid language: {lang}. Choose one of these snapshots: {_ALL_CONFIG}.")


class KoPINLLBConfig(datasets.BuilderConfig):
    """BuilderConfig for the Clean KoPI corpus."""

    def __init__(self, **kwargs):
        """BuilderConfig for Clean KoPI corpus.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class KoPINLLB(datasets.GeneratorBasedBuilder):
    """KoPI NLLB corpus."""

    BUILDER_CONFIGS = [nusantara_config_constructor(sn, "source", _SOURCE_VERSION) for sn in _ALL_CONFIG] + [nusantara_config_constructor(sn, "nusantara_ssp", _NUSANTARA_VERSION) for sn in _ALL_CONFIG]

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "score": datasets.Value("float32"),
                    "source": datasets.Value("string"),
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

    def _split_generators(self, dl_manager):
        name = self.config.name.replace("_" + self.config.schema, "")
        name = name.replace(_DATASETNAME + "_", "")
        split_name = name.split("-")
        if split_name[0] == "all":
            train = [_BASE_URL.format(tipe=split_name[1], lang=m) for m in _CONF_LANG]
        else:
            train = [_BASE_URL.format(tipe=split_name[1], lang=split_name[0])]
        train_downloaded_files = dl_manager.download(train)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_downloaded_files})]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            logger.info(f"Generating examples from {filepath}")
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)
                        if self.config.schema == "nusantara_ssp":
                            yield id_, {"id": str(id_), "text": example["text"]}
                            id_ += 1
                        else:
                            yield id_, {"text": example["text"], "url": example["url"], "source": example["source"], "score": float(example["score"])}
                            id_ += 1
