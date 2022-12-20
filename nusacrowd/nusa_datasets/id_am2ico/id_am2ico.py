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
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{liu-etal-2021-am2ico,
    title = "{AM}2i{C}o: Evaluating Word Meaning in Context across Low-Resource Languages with Adversarial Examples",
    author = "Liu, Qianchu  and
      Ponti, Edoardo Maria  and
      McCarthy, Diana  and
      Vuli{\'c}, Ivan  and
      Korhonen, Anna",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.571",
    doi = "10.18653/v1/2021.emnlp-main.571",
    pages = "7151--7162",
    abstract = "Capturing word meaning in context and distinguishing between correspondences and variations across languages is key to building successful multilingual and cross-lingual text representation models. However, existing multilingual evaluation datasets that evaluate lexical semantics {``}in-context{''} have various limitations. In particular, 1) their language coverage is restricted to high-resource languages and skewed in favor of only a few language families and areas, 2) a design that makes the task solvable via superficial cues, which results in artificially inflated (and sometimes super-human) performances of pretrained encoders, and 3) no support for cross-lingual evaluation. In order to address these gaps, we present AM2iCo (Adversarial and Multilingual Meaning in Context), a wide-coverage cross-lingual and multilingual evaluation set; it aims to faithfully assess the ability of state-of-the-art (SotA) representation models to understand the identity of word meaning in cross-lingual contexts for 14 language pairs. We conduct a series of experiments in a wide range of setups and demonstrate the challenging nature of AM2iCo. The results reveal that current SotA pretrained encoders substantially lag behind human performance, and the largest gaps are observed for low-resource languages and languages dissimilar to English.",
}
"""

_LANGUAGES = ["ind", "eng"]
_LOCAL = False

_DATASETNAME = "id_am2ico"

_DESCRIPTION = """\
In this work, we present AM2iCo, a wide-coverage and carefully designed cross-lingual and multilingual evaluation set;
it aims to assess the ability of state-of-the-art representation models to reason over cross-lingual 
lexical-level concept alignment in context for 14 language pairs. 

This dataset only contain Indonesian - English language pair.
"""

_HOMEPAGE = "https://github.com/cambridgeltl/AM2iCo"

_LICENSE = "CC-BY 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/cambridgeltl/AM2iCo/master/data/id/train.tsv",
        "dev": "https://raw.githubusercontent.com/cambridgeltl/AM2iCo/master/data/id/dev.tsv",
        "test": "https://raw.githubusercontent.com/cambridgeltl/AM2iCo/master/data/id/test.tsv",
    }
}

_SUPPORTED_TASKS = [Tasks.CONCEPT_ALIGNMENT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """IndoCollex: A Testbed for Morphological Transformation of Indonesian 
    Colloquial Words"""

    label_classes = ["T", "F"]

    BUILDER_CONFIGS = (
        NusantaraConfig(
            name="id_am2ico_source",
            version=_SOURCE_VERSION,
            description="Indonesia am2ico source schema",
            schema="source",
            subset_id="id_am2ico",
        ),
        NusantaraConfig(
            name="id_am2ico_nusantara_pairs",
            version=_NUSANTARA_VERSION,
            description="Indonesia am2ico Nusantara schema",
            schema="nusantara_pairs",
            subset_id="id_am2ico",
        ),
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "no": datasets.Value("string"),
                    "context1": datasets.Value("string"),
                    "context2": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        elif self.config.schema == "nusantara_pairs":
            features = schemas.pairs_features(self.label_classes)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

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

        # Read tsv (separated by tab).
        df = pd.read_csv(filepath, sep="\t")

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "no": str(row.Index),
                    "context1": str(row.context1).rstrip(),
                    "context2": str(row.context2).rstrip(),
                    "label": str(row.label).rstrip(),
                }
                yield row.Index, ex

        elif self.config.schema == "nusantara_pairs":
            for row in df.itertuples():
                ex = {
                    "id": str(row.Index),
                    "text_1": str(row.context1).rstrip(),
                    "text_2": str(row.context2).rstrip(),
                    "label": str(row.label).rstrip(),
                }
                yield row.Index, ex
