import os
from pathlib import Path
from typing import Dict, List, Tuple
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import datasets
import json
import xml.etree.ElementTree as ET

from nusacrowd.utils.configs import NusantaraConfig

_CITATION = """\
@INPROCEEDINGS{8074648,
  author={Suherik, Gilang Julian and Purwarianti, Ayu},
  booktitle={2017 5th International Conference on Information and Communication Technology (ICoIC7)}, 
  title={Experiments on coreference resolution for Indonesian language with lexical and shallow syntactic features}, 
  year={2017},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICoICT.2017.8074648}}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "id_coreference_resolution"

_DESCRIPTION = """\
We built Indonesian coreference resolution that solves not only pronoun referenced to proper noun, but also proper noun to proper noun and pronoun to pronoun.
The differences with the available Indonesian coreference resolution lay on the problem scope and features. 
We conducted experiments using various features (lexical and shallow syntactic features) such as appositive feature, nearest candidate feature, direct sentence feature, previous and next word feature, and a lexical feature of first person. 
We also modified the method to build the training set by selecting the negative examples by cross pairing every single markable that appear between antecedent and anaphor. 
Compared with two available methods to build the training set, we conducted experiments using C45 algorithm. 
Using 200 news sentences, the best experiment achieved 71.6% F-Measure score.
"""

_HOMEPAGE = "https://github.com/tugas-akhir-nlp/indonesian-coreference-resolution-cnn/tree/master/data"

_LICENSE = "Creative Commons Attribution-ShareAlike 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/tugas-akhir-nlp/indonesian-coreference-resolution-cnn/master/data/training/data.xml",
        "test": "https://raw.githubusercontent.com/tugas-akhir-nlp/indonesian-coreference-resolution-cnn/master/data/testing/data.xml"
    }
}

_SUPPORTED_TASKS = [Tasks.COREFERENCE_RESOLUTION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class IDCoreferenceResolution(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_coreference_resolution_source",
            version=SOURCE_VERSION,
            description="ID Coreference Resolution source schema",
            schema="source",
            subset_id="id_coreference_resolution",
        ),
        NusantaraConfig(
            name="id_coreference_resolution_nusantara_kb",
            version=NUSANTARA_VERSION,
            description="ID Coreference Resolution Nusantara schema",
            schema="nusantara_kb",
            subset_id="id_coreference_resolution",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_coreference_resolution_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "phrases": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": [
                                {
                                    "word": datasets.Value("string"),
                                    "ne": datasets.Value("string"),
                                    "label": datasets.Value("string")
                                }
                            ]
                        }
                    ]
                }
            )

        elif self.config.schema == "nusantara_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]

        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
        ]

    def _parse_phrase(self, phrase):
        splitted_text = phrase.text.split(" ")
        splitted_ne = []
        if ("ne" in phrase.attrib):
            splitted_ne = phrase.attrib["ne"].split("|")
        words = []
        for i in range(0, len(splitted_text)):
            word = splitted_text[i].split("\\")
            ne = ""
            label = ""
            if (i < len(splitted_ne)):
                ne = splitted_ne[i]
            if (len(word) > 1):
                label = word[1]
            words.append({
                "word": word[0],
                "ne": ne,
                "label": label
            })
        
        id = ""

        if ("id" in phrase.attrib):
            id = phrase.attrib["id"]

        return {
            "id": id,
            "type": phrase.attrib["type"],
            "text": words
        }


    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        data = ET.parse(filepath).getroot()

        for each_sentence in data:
            sentence = {
                "id": each_sentence.attrib["id"],
                "phrases": [],
            }
            for phrase in each_sentence:
                parsed_phrase = self._parse_phrase(phrase)
                sentence["phrases"].append(parsed_phrase)

            if self.config.schema == "source":
                yield int(each_sentence.attrib["id"]), sentence

            elif self.config.schema == "nusantara_kb":
                ex = {
                    "id": each_sentence.attrib["id"],
                    "passages": [],
                    "entities": [
                        {
                            "id": phrase["id"],
                            "type": phrase["type"],
                            "text": [text["word"] for text in phrase["text"]],
                            "offsets": [[0, len(text["word"])] for text in phrase["text"]],
                            "normalized": [{
                                "db_name": text["ne"],
                                "db_id": ""
                            } for text in phrase["text"]],
                        }
                        for phrase in sentence["phrases"]
                    ],
                    "coreferences": [
                        {
                            "id": each_sentence.attrib["id"],
                            "entity_ids": [phrase["id"] for phrase in sentence["phrases"]]
                        }
                    ],
                    "events": [],
                    "relations": [],
                }
                yield int(each_sentence.attrib["id"]), ex
