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

import csv
from fnmatch import translate
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from translate.storage.tmx import tmxfile

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{anastasopoulos-etal-2020-tico,
    title = "{TICO}-19: the Translation Initiative for {CO}vid-19",
    author = {Anastasopoulos, Antonios  and
      Cattelan, Alessandro  and
      Dou, Zi-Yi  and
      Federico, Marcello  and
      Federmann, Christian  and
      Genzel, Dmitriy  and
      Guzm{\'a}n, Franscisco  and
      Hu, Junjie  and
      Hughes, Macduff  and
      Koehn, Philipp  and
      Lazar, Rosie  and
      Lewis, Will  and
      Neubig, Graham  and
      Niu, Mengmeng  and
      {\"O}ktem, Alp  and
      Paquin, Eric  and
      Tang, Grace  and
      Tur, Sylwia},
    booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID}-19 (Part 2) at {EMNLP} 2020",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlpcovid19-2.5",
    doi = "10.18653/v1/2020.nlpcovid19-2.5",
}
"""

# We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LANGUAGES = ["ind", "ara", "spa", "fra", "hin", "por", "rus", "zho", "eng"]
_LOCAL = False
_SUPPORTED_LANG_PAIRS = [
    ("ind", "ara"), ("ind", "spa"), ("ind", "fra"), ("ind", "hin"), ("ind", "por"), ("ind", "rus"), ("ind", "zho"), ("ind", "eng"),
    ("ara", "ind"), ("spa", "ind"), ("fra", "ind"), ("hin", "ind"), ("por", "ind"), ("rus", "ind"), ("zho", "ind"), ("eng", "ind")
]

_LANG_CODE_MAP = {
    "ind": "id",
    "ara": "ar",
    "spa": "es-LA",
    "fra": "fr",
    "hin": "hi",
    "por": "pt-BR",
    "rus": "ru",
    "zho": "zh",
    "eng": "en"
}

_DATASETNAME = "tico_19"

_DESCRIPTION = """\
TICO-19 (Translation Initiative for COVID-19) is sampled from a variety of public sources containing 
COVID-19 related content, representing different domains (e.g., news, wiki articles, and others). TICO-19 
includes 30 documents (3071 sentences, 69.7k words) translated from English into 36 languages: Amharic, 
Arabic (Modern Standard), Bengali, Chinese (Simplified), Dari, Dinka, Farsi, French (European), Hausa, 
Hindi, Indonesian, Kanuri, Khmer (Central), Kinyarwanda, Kurdish Kurmanji, Kurdish Sorani, Lingala, 
Luganda, Malay, Marathi, Myanmar, Nepali, Nigerian Fulfulde, Nuer, Oromo, Pashto, Portuguese (Brazilian), 
Russian, Somali, Spanish (Latin American), Swahili, Congolese Swahili, Tagalog, Tamil, Tigrinya, Urdu, Zulu.
"""

_HOMEPAGE = "https://tico-19.github.io"

_LICENSE = "CC0"

_URLS = {
    "evaluation": "https://tico-19.github.io/data/tico19-testset.zip",
    "all": "https://tico-19.github.io/data/TM/all.{lang_pairs}.tmx.zip"
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(lang_source, lang_target, schema, version):
    """Construct NusantaraConfig with tico_19_{lang_source}_{lang_target}_{schema} as the name format"""
    if schema != "source" and schema != "nusantara_t2t":
        raise ValueError(f"Invalid schema: {schema}")

    if lang_source == "" and lang_target == "":
        return NusantaraConfig(
            name="tico_19_{schema}".format(schema=schema),
            version=datasets.Version(version),
            description="tico_19 {schema} schema for default language pair (eng-ind)".format(schema=schema),
            schema=schema,
            subset_id="tico_19",
        )
    else:
        return NusantaraConfig(
            name="tico_19_{src}_{tgt}_{schema}".format(src=lang_source, tgt=lang_target, schema=schema),
            version=datasets.Version(version),
            description="tico_19 {schema} schema for {src}-{tgt} language pair".format(src=lang_source, tgt=lang_target, schema=schema),
            schema=schema,
            subset_id="tico_19",
        )

class Tico19(datasets.GeneratorBasedBuilder):
    """TICO-19 is MT dataset sampled from a variety of public sources containing COVID-19 related content"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        nusantara_config_constructor(src, tgt, schema, version)
        for src, tgt in [("", "")] + _SUPPORTED_LANG_PAIRS for schema, version in zip(["source", "nusantara_t2t"], [_SOURCE_VERSION, _NUSANTARA_VERSION])
    ]

    DEFAULT_CONFIG_NAME = "tico_19_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sourceLang": datasets.Value("string"),
                    "targetLang": datasets.Value("string"),
                    "sourceString": datasets.Value("string"),
                    "targetString": datasets.Value("string"),
                    "stringID": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "translatorId": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        
        try:
            lang_pairs_config = re.search("tico_19_(.+?)_(source|nusantara_t2t)", self.config.name).group(1)
            lang_src, lang_tgt = lang_pairs_config.split("_")
        except AttributeError:
            lang_src, lang_tgt = "eng", "ind"

        lang_pairs = _LANG_CODE_MAP[lang_src] + "-" + _LANG_CODE_MAP[lang_tgt]

        # dev & test split only applicable to eng-ind language pair
        if lang_pairs in ["en-id", "id-en"]:
            data_dir = dl_manager.download_and_extract(_URLS["evaluation"])
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "tico19-testset", "test", f"test.en-id.tsv"),
                        "lang_source": lang_src,
                        "lang_target": lang_tgt
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "tico19-testset", "dev", f"dev.en-id.tsv"),
                        "lang_source": lang_src,
                        "lang_target": lang_tgt
                    },
                ),
            ]
        else:
            data_dir = dl_manager.download_and_extract(_URLS["all"].format(lang_pairs=lang_pairs))
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, f"all.{lang_pairs}.tmx"),
                        "lang_source": lang_src,
                        "lang_target": lang_tgt
                    },
                )
            ]

    def _generate_examples(self, filepath: Path, lang_source: str, lang_target: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        if self.config.schema == "source":
            # eng-ind language pair dataset provided in .tsv format
            if (lang_source == "eng" and lang_target == "ind") or (lang_source == "ind" and lang_target == "eng"):
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t", quotechar='"')
                    for id_, row in enumerate(reader):
                        if id_ == 0:
                            continue
                        if lang_source == "eng":
                            source_lang = row[0]
                            target_lang = row[1]
                            source_string = row[2]
                            target_string = row[3]
                        else:
                            source_lang = row[1]
                            target_lang = row[0]
                            source_string = row[3]
                            target_string = row[2]
                        yield id_, {
                            "sourceLang": source_lang,
                            "targetLang": target_lang,
                            "sourceString": source_string,
                            "targetString": target_string,
                            "stringID": row[4],
                            "url": row[5],
                            "license": row[6],
                            "translatorId": row[7],
                        }
            
            # all language pairs except eng-ind dataset provided in .tmx format
            else:
                with open(filepath, "rb") as f:
                    tmx_file = tmxfile(f)

                for id_, node in enumerate(tmx_file.unit_iter()):
                    try:
                        url = [text for text in node.xmlelement.itertext('prop')][0]
                    except:
                        url = ""
                    yield id_, {
                        "sourceLang": _LANG_CODE_MAP[lang_source],
                        "targetLang": _LANG_CODE_MAP[lang_target],
                        "sourceString": node.source,
                        "targetString": node.target,
                        "stringID": node.getid(),
                        "url": url,
                        "license": "",
                        "translatorId": "",
                    }

        elif self.config.schema == "nusantara_t2t":
            if (lang_source == "eng" and lang_target == "ind") or (lang_source == "ind" and lang_target == "eng"):
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t", quotechar='"')
                    for id_, row in enumerate(reader):
                        if id_ == 0:
                            continue
                        if lang_source == "eng":
                            source_string = row[2]
                            target_string = row[3]
                        else:
                            source_string = row[3]
                            target_string = row[2]
                        yield id_, {
                            "id": row[4],
                            "text_1": source_string,
                            "text_2": target_string,
                            "text_1_name": lang_source,
                            "text_2_name": lang_target
                        }
            else:
                with open(filepath, "rb") as f:
                    tmx_file = tmxfile(f)
                
                for id_, node in enumerate(tmx_file.unit_iter()):
                    yield id_, {
                        "id": node.getid(),
                        "text_1": node.source,
                        "text_2": node.target,
                        "text_1_name": lang_source,
                        "text_2_name": lang_target
                    }
