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
KoPI-CC corpus

[nusantara_schema_name] = ssp
"""

import gzip
import json
from typing import List

import datasets
import zstandard as zstd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "kopi_cc"
_LANGUAGES  = ["ind"]
_LOCAL = False
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME
_URL = "https://commoncrawl.org/"
_CITATION = """\
       @ARTICLE{2022arXiv220106642A,
       author = {{Abadji}, Julien and {Ortiz Suarez}, Pedro and {Romary}, Laurent and {Sagot}, Benoit},
        title = "{Towards a Cleaner Document-Oriented Multilingual Crawled Corpus}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = 2022,
        month = jan,
          eid = {arXiv:2201.06642},
        pages = {arXiv:2201.06642},
archivePrefix = {arXiv},
       eprint = {2201.06642},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220106642A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@inproceedings{AbadjiOrtizSuarezRomaryetal.2021,
  author    = {Julien Abadji and Pedro Javier Ortiz Su{\'a}rez and Laurent Romary and Benoit Sagot},
  title     = {Ungoliant: An optimized pipeline for the generation of a very large-scale multilingual web corpus},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-9) 2021. Limerick, 12 July 2021 (Online-Event)},
  editor    = {Harald L{\"u}ngen and Marc Kupietz and Piotr Bański and Adrien Barbaresi and Simon Clematide and Ines Pisetta},
  publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-10468},
  url       = {https://nbn-resolving.org/urn:nbn:de:bsz:mh39-104688},
  pages     = {1 -- 9},
  year      = {2021},
  abstract  = {Since the introduction of large language models in Natural Language Processing, large raw corpora have played a crucial role in Computational Linguistics.},
  language  = {en}
}

"""

_DESCRIPTION = """\
    KoPI-CC (Korpus Perayapan Indonesia)-CC is Indonesian Only Extract from Common Crawl snapshots ,each snapshots get extracted using ungoliant and get extra "filtering" using deduplication technique

"""

_HOMEPAGE = "https://huggingface.co/datasets/munggok/KoPI-CC"

_LICENSE = "CC0"

_URLS = {
    "raw": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/raw/id_meta_{index}.jsonl.zst",
    "dedup": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/dedup/oscar-{index:012d}.json.gz",
    "neardup": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/neardup/oscar-neardup-{index:012d}.json.gz",
    "neardup_clean": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/neardup_clean/cleaned_oscar-neardup-{index:012d}.json.gz",
}


_N_SHARDS_PER_SNAPSHOT = {
    "2021_10": {"dedup": 132, "neardup": 120, "neardup_clean": 120},
    "2021_17": {"raw": 31, "dedup": 47, "neardup": 41, "neardup_clean": 41},
    "2021_21": {"raw": 63, "dedup": 37, "neardup": 33, "neardup_clean": 33},
    "2021_25": {"raw": 31, "dedup": 32, "neardup": 28, "neardup_clean": 28},
    "2021_31": {"raw": 35, "dedup": 47, "neardup": 42, "neardup_clean": 42},
    "2021_39": {"raw": 35, "dedup": 44, "neardup": 38, "neardup_clean": 38},
    "2021_43": {"raw": 35, "dedup": 44, "neardup": 39, "neardup_clean": 39},
    "2021_49": {"dedup": 31, "neardup": 28, "neardup_clean": 28},
    "2022_05": {"raw": 40, "dedup": 18, "neardup": 18, "neardup_clean": 35},
    "2022_21": {"raw": 40, "dedup": 42, "neardup": 37, "neardup_clean": 37},
    "2022_27": {"raw": 79, "dedup": 38, "neardup": 33, "neardup_clean": 33},
}

_SNAP_CONFIG = []
for m in list(_N_SHARDS_PER_SNAPSHOT.keys()):
    ka = list(_N_SHARDS_PER_SNAPSHOT[m].keys())
    conf = [m + "-" + a for a in ka]
    _SNAP_CONFIG.extend(conf)
_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_ALL_CONFIG = ["all-raw", "all-dedup", "all-neardup", "all-neardup_clean"] + _SNAP_CONFIG

_SOURCE_VERSION = "2018.12.01"

_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(snapshot, schema, version):
    """Construct NusantaraConfig"""
    if schema != "source" and schema != "nusantara_ssp":
        raise ValueError(f"Invalid schema: {schema}")

    if snapshot == "":
        raise ValueError(f"Snapshot is required. Choose one of these Snapshot: {_ALL_CONFIG}.")
    elif snapshot in _SNAP_CONFIG + _ALL_CONFIG:
        return NusantaraConfig(
            name=f"{_DATASETNAME}_{snapshot}_{schema}",
            version=datasets.Version(version),
            description=f"KoPI-CC with {schema} schema for {snapshot}",
            schema=schema,
            subset_id="kopi_cc",
        )
    else:
        raise ValueError(f"Invalid language: {snapshot}. Choose one of these snapshots: {_ALL_CONFIG}.")


class KoPICC(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "2021_17_dedup"

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
        split_name = name.split("-")
        if split_name[0] == "all":
            urls = []
            keys = list(_N_SHARDS_PER_SNAPSHOT.keys())
            idx = 0
            if split_name[1] == "raw":
                idx = 1
                keys = [ur for ur in list(_N_SHARDS_PER_SNAPSHOT.keys()) if _N_SHARDS_PER_SNAPSHOT[ur].get("raw") is not None]
            for m in keys:
                urls.extend([_URLS[split_name[1]].format(snapshot=m, index=k + idx) for k in range(_N_SHARDS_PER_SNAPSHOT[m].get(split_name[1]))])
        else:
            urls = [_URLS[split_name[1]].format(snapshot=split_name[0], index=k + 1) for k in range(_N_SHARDS_PER_SNAPSHOT[split_name[0]][split_name[1]])]
        path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": path, "split": "train", "type": split_name[1]},
            ),
        ]

    def _generate_examples(self, filepaths, split, type):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            if type == "raw":
                with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            meta = dict()
                            meta["warc_headers"] = example["warc_headers"]
                            meta["warc_headers"]["warc-identified-content-language"] = example["warc_headers"].get("warc-identified-content-language")
                            meta["identification"] = example["metadata"]["identification"]
                            meta["annotations"] = example["metadata"]["annotation"]
                            meta["line_identifications"] = example["metadata"]["sentence_identifications"]
                            if self.config.schema == "nusantara_ssp":
                                yield id_, {"id": str(id_), "text": example["content"]}
                                id_ += 1
                            else:
                                yield id_, {"text": example["content"], "url": example["warc_headers"]["warc-target-uri"], "timestamp": example["warc_headers"]["warc-date"], "meta": json.dumps(meta)}
                                id_ += 1
            else:
                with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            if self.config.schema == "nusantara_ssp":
                                yield id_, {"id": str(id_), "text": example["text"]}
                                id_ += 1
                            else:
                                yield id_, {"text": example["text"], "url": example["url"], "timestamp": example["timestamp"], "meta": example["meta"]}
                                id_ += 1
