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
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import datasets

from nusacrowd.utils.configs import NusantaraConfig

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{siallagan2022sampiran,
  title={Poetry Generation for Indonesian Pantun: Comparison Between SeqGAN and GPT-2},
  author={Emmanuella Anggi Siallagan and Ika Alfina},
  booktitle={Jurnal Ilmu Komputer dan Informasi (Journal of Computer Science and Information) Vol 1x No x February 2023 (Minor Revision)},
  year={2023},
}
"""

_DATASETNAME = "sampiran"


_DESCRIPTION = """\
Sampiran is a dataset for pantun generation. It consists of 7.8K Indonesian pantun, collected from various sources (online). 
Pantun is a traditional Malay poem consisting of four lines: two lines of deliverance and two lines of message. 
This dataset filtered the gathered Pantun to follow the general rules of Pantun; four lines with ABAB rhyme and eight to twelve syllables per line.
"""

_LANGUAGES = ["ind"]
_LOCAL = False
_HOMEPAGE = "https://github.com/ir-nlp-csui/sampiran"
_LICENSE = "AGPL-3.0"

_URLS = "https://raw.githubusercontent.com/ir-nlp-csui/sampiran/main/sampiran.txt"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class SampiranDataset(datasets.GeneratorBasedBuilder):
    """Sampiran is a dataset for pantun generation. It consists of 7.8K Indonesian pantun,
    collected from various sources (online)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="sampiran_source",
            version=SOURCE_VERSION,
            description="sampiran source schema",
            schema="source",
            subset_id="sampiran",
        ),
        NusantaraConfig(
            name="sampiran_nusantara_ssp",
            version=NUSANTARA_VERSION,
            description="sampiran Nusantara schema",
            schema="nusantara_ssp",
            subset_id="sampiran",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sampiran_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "pantun": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_ssp":
            # e.g. features = schemas.kb_features
            # TODO: Choose your nusantara schema here
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        filepath = Path(dl_manager.download(_URLS))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepath},
            ),
        ]

    def _read_data(self, filepath: Path) -> List[Dict]:
        """Reads the data from the source file and returns a list of dicts."""

    def _generate_examples(self, filepath: Path, split: str = None) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema != "source" and self.config.schema != "nusantara_ssp":
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        # Read the file line by line

        if self.config.name == "sampiran_source":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    ex = {
                        "id": str(id_),
                        "pantun": str(row).rstrip(),
                    }
                    yield id_, ex

        elif self.config.name == "sampiran_nusantara_ssp":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    ex = {"id": str(id_), "text": str(row).rstrip()}
                    yield id_, ex
