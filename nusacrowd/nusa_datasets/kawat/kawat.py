import os
from pathlib import Path
from typing import Dict, List, Tuple
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import datasets
import json

from nusacrowd.utils.configs import NusantaraConfig

_CITATION = """\
@article{kurniawan2019,
  title={KaWAT: A Word Analogy Task Dataset for Indonesian},
  url={http://arxiv.org/abs/1906.09912},
  journal={arXiv:1906.09912 [cs]},
  author={Kurniawan, Kemal},
  year={2019},
  month={Jun}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "kawat"

_DESCRIPTION = """\
We introduced KaWAT (Kata Word Analogy  Task), a new word analogy task dataset for Indonesian. 
We evaluated on it several existing pretrained Indonesian word embeddings and embeddings trained on Indonesian online news corpus. 
We also tested them on two downstream tasks and found that pretrained word embeddings helped either by reducing the training epochs
or yielding significant performance gains.
"""

_HOMEPAGE = "https://github.com/kata-ai/kawat"

_LICENSE = "Creative Commons Attribution-ShareAlike 4.0"

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/kata-ai/kawat/master/{}/{}",
}

_SUPPORTED_TASKS = [Tasks.WORD_SENSE_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_PATH_FILE = [
    {
        "folder": "semantic",
        "file": [
            "antonyms.txt",
            "country-capitals.txt",
            "country-currencies.txt",
            "gender-specific-words.txt",
            "measure-words.txt",
            "province-capitals.txt"
        ]
    },
    {
        "folder": "syntax",
        "file": [
            "nouns.txt",
            "plurals.txt",
            "reduplications.txt",
            "verbs.txt"
        ]
    }
]

class Kawat(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="kawat_source",
            version=SOURCE_VERSION,
            description="Kawat source schema",
            schema="source",
            subset_id="kawat",
        ),
        NusantaraConfig(
            name="kawat_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="Kawat Nusantara schema",
            schema="nusantara_t2t",
            subset_id="kawat",
        ),
    ]

    DEFAULT_CONFIG_NAME = "kawat_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_2_name": datasets.Value("string"),
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
        datas = []

        num = 0

        for each_path_file in _PATH_FILE:
            for each_file in each_path_file["file"]:
                data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME].format(each_path_file['folder'], each_file))

                parsed_lines = open(data_dir, "r").readlines()

                titles = parsed_lines[0].split("\t")

                num_columns = len(titles)

                titles[num_columns-1] = titles[num_columns-1][:-1]

                for i in range(1, len(parsed_lines)):
                    words = parsed_lines[i].split("\t")

                    words[num_columns-1] = words[num_columns-1][:-1]

                    for j in range(1, num_columns):
                        if words[j] != "-":
                            datas.append({
                                "id": str(num),
                                "text_1": words[0],
                                "text_1_name": titles[0],
                                "text_2": words[j],
                                "text_2_name": titles[j],
                            })
                        num+=1
        
        with open(data_dir, 'w') as f:
            f.write(json.dumps(datas))

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
        data = json.load(open(filepath, "r"))

        if self.config.schema == "source":
            key = 0
            for each_data in data:
                example = {
                    "id": each_data["id"],
                    "text_1": each_data["text_1"],
                    "text_1_name": each_data["text_1_name"],
                    "text_2": each_data["text_2"],
                    "text_2_name": each_data["text_2_name"],
                }
                yield key, example
                key+=1

        elif self.config.schema == "nusantara_t2t":
            key = 0
            for each_data in data:
                example = {
                    "id": each_data["id"],
                    "text_1": each_data["text_1"],
                    "text_1_name": each_data["text_1_name"],
                    "text_2": each_data["text_2"],
                    "text_2_name": each_data["text_2_name"],
                }
                yield key, example
                key+=1
