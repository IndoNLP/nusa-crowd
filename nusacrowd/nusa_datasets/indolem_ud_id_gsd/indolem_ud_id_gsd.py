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
This template serves as a starting point for contributing a dataset to the Nusantara Dataset repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[nusantara_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_ud_data, load_ud_data_as_nusantara_kb
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{mcdonald-etal-2013-universal,
    title = "{U}niversal {D}ependency Annotation for Multilingual Parsing",
    author = {McDonald, Ryan  and
      Nivre, Joakim  and
      Quirmbach-Brundage, Yvonne  and
      Goldberg, Yoav  and
      Das, Dipanjan  and
      Ganchev, Kuzman  and
      Hall, Keith  and
      Petrov, Slav  and
      Zhang, Hao  and
      T{\"a}ckstr{\"o}m, Oscar  and
      Bedini, Claudia  and
      Bertomeu Castell{\'o}, N{\'u}ria  and
      Lee, Jungmee},
    booktitle = "Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P13-2017",
    pages = "92--97",
}

@article{DBLP:journals/corr/abs-2011-00677,
    author    = {Fajri Koto and
                 Afshin Rahimi and
                 Jey Han Lau and
                 Timothy Baldwin},
    title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
                 Model for Indonesian {NLP}},
    journal   = {CoRR},
    volume    = {abs/2011.00677},
    year      = {2020},
    url       = {https://arxiv.org/abs/2011.00677},
    eprinttype = {arXiv},
    eprint    = {2011.00677},
    timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "indolem_ud_id_gsd"

_DESCRIPTION = """\
The Indonesian-GSD treebank consists of 5598 sentences and 122k words split into train/dev/test of 97k/12k/11k words.
The treebank was originally converted from the content head version of the universal dependency treebank v2.0 (legacy) in 2015.\
In order to comply with the latest Indonesian annotation guidelines, the treebank has undergone a major revision between UD releases v2.8 and v2.9 (2021).
"""

_HOMEPAGE = "https://indolem.github.io/"

_LICENSE = "Creative Commons Attribution 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-train.conllu",
        "validation": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-dev.conllu",
        "test": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-test.conllu",
    },
}

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class IndolemUdIdGsdDataset(datasets.GeneratorBasedBuilder):
    """The Indonesian-GSD treebank, part of Universal-Dependency project. Consists of 5598 sentences and 122k words."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

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
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
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
        urls = _URLS[self.config.subset_id]
        data_dir = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

        try:
            generator_fn = {
                "source": load_ud_data,
                "nusantara_kb": load_ud_data_as_nusantara_kb,
            }[self.config.schema]
        except KeyError:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        for key, example in enumerate(generator_fn(filepath)):
            yield key, example


# if __name__ == "__main__":
#     datasets.load_dataset(__file__)
