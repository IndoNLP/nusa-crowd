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
""" NERGrit Long Dataset"""

from pathlib import Path
from typing import List
import datasets
from nusantara.utils import schemas
from nusantara.utils.common_parser import load_conll_data
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "nergrit"

_DESCRIPTION = """\
This NER dataset is taken from the Grit-ID repository, and the labels are spans in IOB chunking representation.
The dataset consists of three kinds of named entity tags, PERSON (name of person), PLACE (name of location), and
ORGANIZATION (name of organization).
"""

_HOMEPAGE = "https://github.com/grit-id/nergrit-corpus"

_LICENSE = "MIT"

_URL_ROOT = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit"
_URLs = {split: f"{_URL_ROOT}/{split}_preprocess.txt" for split in ["train", "validation", "test"]}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class NergritDataset(datasets.GeneratorBasedBuilder):
    """NERGrit."""

    label_classes = ["I-PERSON", "B-ORGANISATION", "I-ORGANISATION", "B-PLACE", "I-PLACE", "O", "B-PERSON"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="nergrit_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="NERGrit source schema",
            schema="source",
            subset_id="nergrit",
        ),
        NusantaraConfig(
            name="nergrit_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="NERGrit Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="nergrit",
        ),
    ]

    DEFAULT_CONFIG_NAME = "nergrit_source"

    def _info(self):
        features = None
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")],
                                          "ner_tag": [datasets.Value("string")]})
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
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        validation_tsv_path = Path(dl_manager.download_and_extract(_URLs["validation"]))
        test_tsv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "train": train_tsv_path,
            "validation": validation_tsv_path,
            "test": test_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        conll_dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for index, row in enumerate(conll_dataset):
                ex = {"index": str(i), "tokens": row["sentence"], "ner_tag": row["label"]}
                yield index, ex
        elif self.config.schema == "nusantara_seq_label":
            for index, row in enumerate(conll_dataset):
                ex = {"id": str(i), "tokens": row["sentence"], "labels": row["label"]}
                yield index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


# TODO: Remove this before making your PR
if __name__ == "__main__":
    ds = datasets.load_dataset(__file__, name="nergrit_source")
    print(ds)
    for i in range(3):
        print(ds["train"][i])
