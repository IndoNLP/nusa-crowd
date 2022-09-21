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

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@INPROCEEDINGS{8355036,
  author={Alfina, Ika and Savitri, Septiviana and Fanany, Mohamad Ivan},
  title={Modified DBpedia entities expansion for tagging automatically NER dataset},
  booktitle={2017 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
  pages={216-221},
  year={2017},
  url={https://ieeexplore.ieee.org/document/8355036},
  doi={10.1109/ICACSIS.2017.8355036}}

@INPROCEEDINGS{7872784,
  author={Alfina, Ika and Manurung, Ruli and Fanany, Mohamad Ivan},
  booktitle={2016 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
  title={DBpedia entities expansion in automatically building dataset for Indonesian NER},
  year={2016},
  pages={335-340},
  doi={10.1109/ICACSIS.2016.7872784}}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "singgalang"

_DESCRIPTION = """\
Rule-based annotation Indonesian NER Dataset of 48,957 sentences or 1,478,286 tokens.
Annotation conforms the Stanford-NER format (https://stanfordnlp.github.io/CoreNLP/ner.html) for 3 NER tags of Person, Organisation, and Place.
This dataset consists of 41,297, 14,770, and 82,179 tokens of entity (respectively) from over 14, 6, and 5 rules.
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/singgalang"

_LICENSE = """\
You can use this dataset for free. You don't need our permission to use it. Please cite our paper if your work uses our data in your publication.
Please note that you are not allowed to create a copy of this dataset and share it publicly in your own repository without our permission.\
"""

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/ir-nlp-csui/singgalang/main/SINGGALANG.tsv",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class SinggalangDataset(datasets.GeneratorBasedBuilder):
    """Rule-based annotation Indonesian NER Dataset of 48,957 sentences with 3 NER tags"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    label_classes = [
        "O",
        "Person",
        "Organisation",
        "Place",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_seq_label",
            version=NUSANTARA_VERSION,
            description=f"{_DATASETNAME} Nusantara schema",
            schema="nusantara_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentence": [datasets.Value("string")],
                    "label": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.label_classes)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        url = _URLS[_DATASETNAME]
        data_path = dl_manager.download(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for key, ex in enumerate(dataset):
                yield key, ex

        elif self.config.schema == "nusantara_seq_label":
            for key, ex in enumerate(dataset):
                yield key, {
                    "id": str(key),
                    "tokens": ex["sentence"],
                    "labels": ex["label"],
                }
