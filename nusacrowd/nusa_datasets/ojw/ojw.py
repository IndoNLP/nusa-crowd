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
Dataloader implementation for Old Javanese Wordnet dataset.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils.configs import NusantaraConfig

_CITATION = """\
@inproceedings{moeljadi-aminullah-2020-building,
    title = "Building the Old {J}avanese {W}ordnet",
    author = "Moeljadi, David  and
      Aminullah, Zakariya Pamuji",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.359",
    pages = "2940--2946",
    abstract = "This paper discusses the construction and the ongoing development of the Old Javanese Wordnet.
     The words were extracted from the digitized version of the Old Javanese{--}English Dictionary (Zoetmulder, 1982).
     The wordnet is built using the {`}expansion{'} approach (Vossen, 1998), leveraging on the Princeton Wordnet{'}s
     core synsets and semantic hierarchy, as well as scientific names. The main goal of our project was to produce a
     high quality, human-curated resource. As of December 2019, the Old Javanese Wordnet contains 2,054 concepts or
     synsets and 5,911 senses. It is released under a Creative Commons Attribution 4.0 International License
     (CC BY 4.0). We are still developing it and adding more synsets and senses. We believe that the lexical data
     made available by this wordnet will be useful for a variety of future uses such as the development of Modern
     Javanese Wordnet and many language processing tasks and linguistic research on Javanese.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DATASETNAME = "ojw"

_DESCRIPTION = """\
This dataset contains Old Javanese written language aimed to build a machine readable sources for Old Javanese: providing a wordnet for the language (Moeljadi et. al., 2020).
"""


_HOMEPAGE = "https://github.com/davidmoeljadi/OJW"


_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"


_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/davidmoeljadi/OJW/master/wn-kaw.tab",
}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_LOCAL = False

_LANGUAGES = ["kaw"]


class OJW(datasets.GeneratorBasedBuilder):
    """Old Javanese Wordnet (OJW) is a dataset that contains Old Javanese words and each variants of the words if available.
    The dataset consists of 5038 rows and three columns: synset, tlemma, and tvariants."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="ojw_source",
            version=SOURCE_VERSION,
            description="ojw source schema",
            schema="source",
            subset_id="ojw",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ojw_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"synset": datasets.Value("string"), "tlemma": datasets.Value("string"), "tvariants": datasets.Value("string")})

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
        data_dir = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, sep="\t", names=["synset", "tlemma", "tvariants"])

        if self.config.schema == "source":
            for key, example in df.iterrows():
                yield key, {
                    "synset": example["synset"],
                    "tlemma": example["tlemma"],
                    "tvariants": example["tvariants"],
                }
