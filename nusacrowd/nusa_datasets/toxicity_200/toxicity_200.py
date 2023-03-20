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

import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME

_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME
_LOCAL = False

_CITATION = """\
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
"""

_DATASETNAME = "toxicity_200"

_DESCRIPTION = """\
Toxicity-200 is a wordlist to detect toxicity in 200 languages. It contains files that include frequent words and phrases generally considered toxic because they represent: 1) frequently used profanities; 2) frequently used insults and hate speech terms, or language used to bully, denigrate, or demean; 3) pornographic terms; and 4) terms for body parts associated with sexual activity.
"""

_HOMEPAGE = "https://github.com/facebookresearch/flores/blob/main/toxicity"

_LICENSE = "CC-BY-SA 4.0"
_LANGUAGES = ["ind", "ace", "bjn", "bug", "jav"]
_LANGUAGE_MAP = {"ind": "Indonesia", "ace": "Aceh", "bjn": "Banjar", "bug": "Bugis", "jav": "Java"}
_URLS = {
    "toxicity_200": "https://tinyurl.com/NLLB200TWL",
}
_PASS = "tL4nLLb"
_SUPPORTED_TASKS = []  # [Tasks.SELF_SUPERVISED_PRETRAINING]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(lang, schema, version):
    if lang == "":
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source":
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="toxicity_200_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="toxicity 200 with {schema} schema for {lang} language".format(lang=_LANGUAGE_MAP[lang], schema=schema),
        schema=schema,
        subset_id="toxicity_200",
    )

def extract_toxic_zip(filepath):
    with ZipFile(filepath, "r") as zip:
        zip.extractall(path=filepath[:-4], pwd=_PASS.encode("utf-8"))
    return filepath[:-4]

class Toxicity200(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES]

    DEFAULT_CONFIG_NAME = "toxicity_200_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "toxic_word": [datasets.Value("string")]})
        else:
            raise NotImplementedError()
        
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
        data_dir = Path(dl_manager.download_and_extract(urls)) / "NLLB-200_TWL"

        data_subdir = {
            "ind": os.path.join(data_dir, "ind_Latn_twl.zip"),
            "ace": os.path.join(data_dir, "ace_Latn_twl.zip"),
            "bjn": os.path.join(data_dir, "bjn_Latn_twl.zip"),
            "bug": os.path.join(data_dir, "bug_Latn_twl.zip"),
            "jav": os.path.join(data_dir, "jav_Latn_twl.zip"),
        }

        lang = self.config.name.split("_")[2]
        text_dir = extract_toxic_zip(data_subdir[lang])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {"text_file": os.path.join(text_dir, lang + "_Latn_twl.txt")},
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        text = open(filepath["text_file"], "r").readlines()
        word_list = list(map(str.strip, text))
        print(text[:5])
        if self.config.schema == "source":
            for id, word in enumerate(word_list):
                row = {"id": str(id), "toxic_word": [word]}
                yield id, row
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

if __name__ == "__main__":
    datasets.load_dataset(__file__)
