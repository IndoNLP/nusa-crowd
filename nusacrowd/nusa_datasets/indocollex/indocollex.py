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

from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{wibowo-etal-2021-indocollex,
    title = "{I}ndo{C}ollex: A Testbed for Morphological Transformation of {I}ndonesian Word Colloquialism",
    author = {Wibowo, Haryo Akbarianto  and Nityasya, Made Nindyatama  and Aky{\"u}rek, Afra Feyza  and Fitriany, Suci  and Aji, Alham Fikri  and Prasojo, Radityo Eko  and Wijaya, Derry Tanti},
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.280",
    doi = "10.18653/v1/2021.findings-acl.280",
    pages = "3170--3183",
}"""

_LANGUAGES = ["ind"]
_LOCAL = False

_DATASETNAME = "indocollex"

_DESCRIPTION = """\
IndoCollex: A Testbed for Morphological Transformation of Indonesian Colloquial Words
"""

_HOMEPAGE = "https://github.com/haryoa/indo-collex"

_LICENSE = "CC BY-SA 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://github.com/haryoa/indo-collex/raw/main/data/full.csv",
    },
    f"{_DATASETNAME}_f2i": {
        "train": "https://github.com/haryoa/indo-collex/raw/main/data/formal_to_informal/train.csv",
        "dev": "https://github.com/haryoa/indo-collex/raw/main/data/formal_to_informal/dev.csv",
        "test": "https://github.com/haryoa/indo-collex/raw/main/data/formal_to_informal/test.csv",
    },
    f"{_DATASETNAME}_i2f": {
        "train": "https://github.com/haryoa/indo-collex/raw/main/data/informal_to_formal/train.csv",
        "dev": "https://github.com/haryoa/indo-collex/raw/main/data/informal_to_formal/dev.csv",
        "test": "https://github.com/haryoa/indo-collex/raw/main/data/informal_to_formal/test.csv",
    },
}

_SUPPORTED_TASKS = [Tasks.MORPHOLOGICAL_INFLECTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """IndoCollex: A Testbed for Morphological Transformation of Indonesian Colloquial Words"""

    label_classes = ["acronym", "affixation", "disemvoweling", "rev", "shorten", "sound-alter", "space-dash"]

    BUILDER_CONFIGS = list(
        chain(
            *[
                [
                    NusantaraConfig(
                        name=f"{_DATASETNAME}{suffix}_source",
                        version=datasets.Version(_SOURCE_VERSION),
                        description=f"{_DATASETNAME} source schema",
                        schema="source",
                        subset_id=f"{_DATASETNAME}{suffix}",
                    ),
                    NusantaraConfig(
                        name=f"{_DATASETNAME}{suffix}_nusantara_pairs_multi",
                        version=datasets.Version(_NUSANTARA_VERSION),
                        description=f"{_DATASETNAME} Nusantara schema",
                        schema="nusantara_pairs_multi",
                        subset_id=f"{_DATASETNAME}{suffix}",
                    ),
                ]
                for suffix in ["", "_f2i", "_i2f"]
            ]
        )
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "no": datasets.Value("string"),
                    "transformed": datasets.Value("string"),
                    "original-for": datasets.Value("string"),
                    "transformation": datasets.Value("string"),
                }
            )

        elif self.config.schema == "nusantara_pairs_multi":
            features = schemas.pairs_multi_features(self.label_classes)

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
        data_paths = dl_manager.download(urls)

        ret = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_paths["train"]},
            )
        ]

        if len(data_paths) > 1:
            ret.extend(
                [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"filepath": data_paths["test"]},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"filepath": data_paths["dev"]},
                    ),
                ]
            )

        return ret

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf8") as f:
            dataset = list(map(lambda l: l.rstrip("\r\n").split(","), f))

        _assert = set(map(len, dataset))
        if _assert != {4}:
            raise AssertionError(f"Expecting exactly 4 fields (no, transformed, base, label), but found: {_assert}")

        _assert = dataset[0]
        source_columns = ["no", "transformed", "original-for", "transformation"]
        if _assert != source_columns:
            raise AssertionError(f"The expected header is not found. {_assert}")

        dataset = dataset[1:]

        if self.config.schema == "source":
            for key, ex in enumerate(dataset):
                yield key, dict(zip(source_columns, ex))

        elif self.config.schema == "nusantara_pairs_multi":
            for key, ex in enumerate(dataset):
                yield key, {
                    "id": str(key),
                    "text_1": ex[2],
                    "text_2": ex[1],
                    "label": [ex[3]],
                }
