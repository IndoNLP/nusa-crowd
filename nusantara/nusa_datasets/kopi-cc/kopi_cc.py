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
import zstandard as zstd
import json
import gzip
import os

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "kopi_cc"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME
_URL = "https://commoncrawl.org/"
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

_HOMEPAGE = ""

_LICENSE = "CC0"

_URLS = {
    "raw":"https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/raw/id_meta_{index}.jsonl.zst",
    "dedup": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/dedup/oscar-{index:012d}.json.gz",
    "neardup": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/neardup/oscar-neardup-{index:012d}.json.gz",
    "neardup_clean": "https://huggingface.co/datasets/munggok/KoPI-CC/resolve/main/{snapshot}/neardup_clean/cleaned_oscar-neardup-{index:012d}.json.gz"
}


_N_SHARDS_PER_SNAPSHOT = {
    "2021_10": {
        "dedup":132,
        "neardup":120,
        "neardup_clean":120
    },
    "2021_17": {
        "raw":31,
        "dedup":47,
        "neardup":41,
        "neardup_clean":41
    },
    "2021_21":{
        "raw":63,
        "dedup":37,
        "neardup":33,
        "neardup_clean":33
    },
    "2021_25":{
        "raw":31,
        "dedup":32,
        "neardup":28,
        "neardup_clean":28
    },
    "2021_31":{
        "raw":35,
        "dedup":47,
        "neardup":42,
        "neardup_clean":42
    },
    "2021_39":{
        "raw":35,
        "dedup":44,
        "neardup":38,
        "neardup_clean":38
    },
    "2021_43":{
        "raw":35,
        "dedup":44,
        "neardup":39,
        "neardup_clean":39
    },
    "2021_49":{
        "dedup":31,
        "neardup":28,
        "neardup_clean":28
    },
    "2022_05":{
        "raw":40,
        "dedup":18,
        "neardup":18,
        "neardup_clean":35
    },
    "2022_21":{
        "raw":40,
        "dedup":42,
        "neardup":37,
        "neardup_clean":37
    },
    "2022_27":{
        "raw":79,
        "dedup":38,
        "neardup":33,
        "neardup_clean":33
    }
}

_SNAP_CONFIG = []
for m in list(_N_SHARDS_PER_SNAPSHOT.keys()):
    ka = list(_N_SHARDS_PER_SNAPSHOT[m].keys())
    conf = [m + "-" + a for a in ka]
    _SNAP_CONFIG.extend(conf)
_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_ALL_CONFIG = ["all-raw","all-dedup","all-neardup","all-neardup_clean"] + _SNAP_CONFIG

_SOURCE_VERSION = "2018.12.01"

_NUSANTARA_VERSION = "1.0.0"

def nusantara_config_constructor(snapshot, schema, version):
    """Construct NusantaraConfig with cc100_{lang}_{schema} as the name format."""
    if schema != "source" and schema != "nusa":
        raise ValueError(f"Invalid schema: {schema}")

    if snapshot == "":
        raise ValueError(f"Snapshot is required. Choose one of these Snapshot: {_ALL_CONFIG}.")
    elif snapshot in _SNAP_CONFIG + _ALL_CONFIG:
        if schema == "nusa":
            return NusantaraConfig(
                name=f"{snapshot}-{schema}",
                version=datasets.Version(version),
                description=f"KoPI-CC with {schema} schema for {snapshot}",
                schema=schema,
                subset_id="KoPI-CC",
            )
        else:
            return NusantaraConfig(
                name=f"{snapshot}",
                version=datasets.Version(version),
                description=f"KoPI-CC with {schema} schema for {snapshot}",
                schema=schema,
                subset_id="KoPI-CC",
            )
    else:
        raise ValueError(f"Invalid language: {snapshot}. Choose one of these snapshots: {_ALL_CONFIG}.")


class KoPICC(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        nusantara_config_constructor(sn, "source", _SOURCE_VERSION) for sn in _ALL_CONFIG
    ] + [
        nusantara_config_constructor(sn, "nusa", _NUSANTARA_VERSION) for sn in _ALL_CONFIG
    ] 

    def _info(self):
        if self.config.schema == "source":
            features=datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "timestamp": datasets.Value("string"),
                        "url": datasets.Value("string"),
                        "meta": datasets.Value("string"),
                    }
                )
        elif self.config.schema == "nusa":
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

        split_name = self.config.name.split("-")
        if split_name[0]== "all":
            urls = []
            keys = list(_N_SHARDS_PER_SNAPSHOT.keys())
            idx = 0
            if split_name[1] == "raw":
                idx = 1
                keys = [ur for ur in list(_N_SHARDS_PER_SNAPSHOT.keys()) if _N_SHARDS_PER_SNAPSHOT[ur].get('raw') is not None]
            for m in keys:
                urls.extend([_URLS[split_name[1]].format(snapshot=m,index=k+idx) for k in range(_N_SHARDS_PER_SNAPSHOT[m].get(split_name[1]))])
        else:
            urls = [_URLS[split_name[1]].format(snapshot=split_name[0],index=k+1) for k in range(_N_SHARDS_PER_SNAPSHOT[split_name[0]][split_name[1]])]
        path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        filename, file_extension = os.path.splitext(filepaths)
        for filepath in filepaths:
            if file_extension == '.zst':
                with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            meta = dict()
                            meta["warc_headers"] = example["warc_headers"]
                            meta["warc_headers"]["warc-identified-content-language"] = example[
                                "warc_headers"
                            ].get("warc-identified-content-language")
                            meta["identification"] = example["metadata"]["identification"]
                            meta["annotations"] = example["metadata"]["annotation"]
                            meta["line_identifications"] = example["metadata"][
                                "sentence_identifications"
                            ]
                            if self.config.schema == "nusa":
                                yield id_, {"id":str(id_),'text':example['content']}
                                id_ += 1
                            else:
                                yield id_, {'text':example['content'],'url':example['warc_headers']['warc-target-uri'],'timestamp':example['warc_headers']['warc-date'],"meta": json.dumps(meta)}
                                id_ += 1
            else:
                with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            if self.config.schema == "nusa":
                                yield id_, {"id":str(id_),'text':example['text']}
                                id_ += 1
                            else:
                                yield id_, {'text':example['text'],'url':example['url'],'timestamp':example['timestamp'],'meta': example['meta']}
                                id_ += 1
        