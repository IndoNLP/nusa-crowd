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

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{faruqui-kumar-2015-multilingual,
    title = "Multilingual Open Relation Extraction Using Cross-lingual Projection",
    author = "Faruqui, Manaal  and
      Kumar, Shankar",
    booktitle = "Proceedings of the 2015 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = may # "{--}" # jun,
    year = "2015",
    address = "Denver, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N15-1151",
    doi = "10.3115/v1/N15-1151",
    pages = "1351--1356",
}
"""

_DATASETNAME = "multilingual_open_relations"

_DESCRIPTION = """\
Relation extraction is the task of assigning a semantic relationship between a pair of arguments. This dataset provides automatically extracted relations obtained using the algorithm in Faruqui and Kumar (2015).
Faruqui and Kumar (2015) describe a cross-lingual projection algorithm for multilingual RE that translates text from a foreign language to English, performs relation extraction in English and then projects these relations back to the foreign language.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/shankkumar/multilingualopenrelations15"

_LICENSE = "Attribution 3.0 Unported (CC BY 3.0)"

_LANGUAGES = ["ind"]

_URLS = {
    _DATASETNAME: "local_dataset/multilingual_open_relations-auto-extractions-ind", # TODO: update
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """Relation extraction is the task of assigning a semantic relationship between a pair of arguments. This dataset provides automatically extracted relations obtained using the algorithm in Faruqui and Kumar (2015)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="multilingual_open_relations_source",
            version=SOURCE_VERSION,
            description="Multilingual Open Relations source schema",
            schema="source",
            subset_id="multilingual_open_relations",
        ),
        NusantaraConfig(
            name="multilingual_open_relations_nusantara_kb",
            version=NUSANTARA_VERSION,
            description="Multilingual Open Relations Nusantara schema",
            schema="nusantara_kb",
            subset_id="multilingual_open_relations",
        ),
    ]

    DEFAULT_CONFIG_NAME = "multilingual_open_relations_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # TODO: update if necessary
            features = datasets.Features({
                    "index": datasets.Value("string"),
                    "wikipedia_url": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "sentence_en": datasets.Value("string"),
                    "relations": [{
                        "argument_1": datasets.Value("string"),
                        "argument_2": datasets.Value("string"),
                        "relation": datasets.Value("string"),
                        "argument_1_en": datasets.Value("string"),
                        "argument_2_en": datasets.Value("string"),
                        "relation_en": datasets.Value("string"),
                    }]
                })

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
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]

        # data_dir = dl_manager.download_and_extract(urls)  # TODO: update to get from url
        url_path = Path(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": url_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        rows = self._read_from_source_file(filepath)

        if self.config.schema == "source":
            for idx, row in enumerate(rows):
                row["index"] = str(idx)
                yield idx, row

        elif self.config.schema == "nusantara_kb":
            for idx, row in enumerate(rows):
                row = self._to_nusa_kb_scheme(idx, row)
                yield idx, row
                
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

    def _read_from_source_file(self, filepath: Path):

        """
        Original Data format is the following:
        Wikipedia URL ||| Source Language (SL) Sentence ||| Argument 1 in SL ||| Relation in SL ||| Argument 2 in SL ||| English Sentence ||| Argument 1 in English ||| Relation in English ||| Argument 2 in English
        """

        def parse_row(line):
            sections = line.split("|||")
            row = {
                "wikipedia_url": sections[0].strip(),
                "sentence": sections[1].strip(),
                "argument_1": sections[2].strip(),
                "argument_2": sections[4].strip(),
                "relation": sections[3].strip(),
                "sentence_en": sections[5].strip(),
                "argument_1_en": sections[6].strip(),
                "argument_2_en": sections[8].strip(),
                "relation_en": sections[7].strip(),
            }
            return row

        map_url_sentence_to_idx = {}
        data = []
                
        with open(filepath, "r+") as fr:
            for line in fr:
                row = parse_row(line)                
                
                url_sentence = f"{row['wikipedia_url']}_{row['sentence']}"
                if url_sentence not in map_url_sentence_to_idx:
                    map_url_sentence_to_idx[url_sentence] = len(map_url_sentence_to_idx)
                    data.append({
                        "wikipedia_url": row["wikipedia_url"],
                        "sentence": row["sentence"],
                        "sentence_en": row["sentence_en"],
                        "relations": []
                    })
                rel = {
                    "argument_1": row["argument_1"],
                    "argument_2": row["argument_2"],
                    "relation": row["relation"],
                    "argument_1_en": row["argument_1_en"],
                    "argument_2_en": row["argument_2_en"],
                    "relation_en": row["relation_en"],
                }
                data[map_url_sentence_to_idx[url_sentence]]["relations"].append(rel)
        return data
    
    def _to_nusa_kb_scheme(self, idx, row):
        
        rel_id = 0
        ent_id = 0

        relations = []
        entities = []

        def get_entity(ent_id, entity_str):
            i = f"{idx}_EntID_{ent_id}"
            entity = {
                "id": i,
                "type": "",
                "text": [entity_str],
                "offsets": [[0, 0]], # TODO: calculate the offset
                "normalized": [],
            }
            ent_id += 1
            return i, ent_id, entity

        for rel in row["relations"]:             
            id_1, ent_id, ent_1 = get_entity(ent_id, rel["argument_1"])
            id_2, ent_id, ent_2 = get_entity(ent_id, rel["argument_2"])
            entities.append(ent_1)
            entities.append(ent_2)
            relations.append({
                "id": f"{idx}_RelID_{rel_id}",
                "type": rel["relation"],
                "arg1_id": id_1,
                "arg2_id": id_2,
                "normalized": [
                    {
                        "db_name": None,
                        "db_id": None,
                    }
                ]
            })
            rel_id += 1

        nusa_scheme = {
            "id": str(idx),
            "passages": [
                {
                    "id": f"{idx}_PsgID_0", 
                    "type": "text", 
                    "text": [row["sentence"]], 
                    "offsets": [
                        [0, len(row["sentence"])]
                    ]
                }
            ],
            "entities": entities,
            "coreferences": [],
            "events": [],
            "relations": relations,
        }
        return nusa_scheme
