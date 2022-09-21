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
from typing import Dict, List, Tuple

import datasets
from conllu import TokenList

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_ud_data, load_ud_data_as_nusantara_kb
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article {10.3844/jcssp.2020.1585.1597,
author = {Alfina, Ika and Budi, Indra and Suhartanto, Heru},
title = {Tree Rotations for Dependency Trees: Converting the Head-Directionality of Noun Phrases},
article_type = {journal},
volume = {16},
number = {11},
year = {2020},
month = {Nov},
pages = {1585-1597},
doi = {10.3844/jcssp.2020.1585.1597},
url = {https://thescipub.com/abstract/jcssp.2020.1585.1597},
journal = {Journal of Computer Science},
publisher = {Science Publications}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "ud_id_csui"

_DESCRIPTION = """\
UD Indonesian-CSUI is a conversion from an Indonesian constituency treebank in the Penn Treebank format named Kethu that was also a conversion from a constituency treebank built by Dinakaramani et al. (2015).
This treebank is named after the place where treebanks were built: Faculty of Computer Science (CS), Universitas Indonesia (UI).

About this treebank:
- Genre is news in formal Indonesian (the majority is economic news)
- 1030 sentences (28K words) divided into testing and training dataset of around 10K words and around 18K words respectively.
- Average of 27.4 words per-sentence.
"""

_HOMEPAGE = "https://github.com/UniversalDependencies/UD_Indonesian-CSUI"

_LICENSE = "CC BY-SA 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-train.conllu",
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-test.conllu",
    },
}

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING, Tasks.MACHINE_TRANSLATION, Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class UdIdCsuiDataset(datasets.GeneratorBasedBuilder):
    """Treebank of formal Indonesian news which consists of 1030 sentences (28K words)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    # source: https://universaldependencies.org/u/pos/
    UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_kb",
            version=NUSANTARA_VERSION,
            description=f"{_DATASETNAME} Nusantara KB schema",
            schema="nusantara_kb",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description=f"{_DATASETNAME} Nusantara Text to Text schema",
            schema="nusantara_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_seq_label",
            version=NUSANTARA_VERSION,
            description=f"{_DATASETNAME} Nusantara Seq Label schema",
            schema="nusantara_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_en": datasets.Value("string"),
                    # tokens
                    "id": [datasets.Value("string")],
                    "form": [datasets.Value("string")],
                    "lemma": [datasets.Value("string")],
                    "upos": [datasets.Value("string")],
                    "xpos": [datasets.Value("string")],
                    "feats": [datasets.Value("string")],
                    "head": [datasets.Value("string")],
                    "deprel": [datasets.Value("string")],
                    "deps": [datasets.Value("string")],
                    "misc": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "nusantara_kb":
            features = schemas.kb_features

        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.UPOS_TAGS)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path["test"],
                },
            ),
        ]

    @staticmethod
    def _assert_multispan_range_is_one(token_list: TokenList):
        """
        Asserting that all tokens with multiple span can only have 2 span, and \
        no field other than form has important information
        """
        for token in token_list.filter(id=lambda i: not isinstance(i, int)):
            _id = token["id"]
            assert len(_id) == 3, f"Unexpected length of non-int CONLLU Token's id. Expected 3, found {len(_id)};"
            assert all(isinstance(a, b) for a, b in zip(_id, [int, str, int])), f"Non-int ID should be in format of '\\d+-\\d+'. Found {_id};"
            assert _id[2] - _id[0] == 1, f"Token has more than 2 spans. Found {_id[2] - _id[0] + 1} spans;"
            for key in ["lemma", "upos", "xpos", "feats", "head", "deprel", "deps"]:
                assert token[key] in {"_", None}, f"Field other than 'form' should not contain extra information. Found: '{key}' = '{token[key]}'"

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

        dataset = list(load_ud_data(filepath, filter_kwargs={"id": lambda i: isinstance(i, int)}, assert_fn=self._assert_multispan_range_is_one))

        if self.config.schema == "source":
            pass

        elif self.config.schema == "nusantara_kb":
            dataset = load_ud_data_as_nusantara_kb(filepath, dataset)

        elif self.config.schema == "nusantara_t2t":
            dataset = list(
                map(
                    lambda d: {
                        "id": d["sent_id"],
                        "text_1": d["text"],
                        "text_2": d["text_en"],
                        "text_1_name": "ind",
                        "text_2_name": "eng",
                    },
                    dataset,
                )
            )

        elif self.config.schema == "nusantara_seq_label":
            dataset = list(
                map(
                    lambda d: {
                        "id": d["sent_id"],
                        "tokens": d["form"],
                        "labels": d["upos"],
                    },
                    dataset,
                )
            )

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        for key, example in enumerate(dataset):
            yield key, example


if __name__ == "__main__":
    datasets.load_dataset(__file__)
