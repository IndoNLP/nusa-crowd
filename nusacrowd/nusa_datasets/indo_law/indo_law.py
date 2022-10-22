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
@article{nuranti2022predicting,
  title={Predicting the Category and the Length of Punishment in Indonesian Courts Based on Previous Court Decision Documents},
  author={Nuranti, Eka Qadri and Yulianti, Evi and Husin, Husna Sarirah},
  journal={Computers},
  volume={11},
  number={6},
  pages={88},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "indo_law"

_DESCRIPTION = """\
This study presents predictions of first-level judicial decisions by utilizing a collection of Indonesian court decision documents. 
We propose using multi-level learning, namely, CNN+attention, using decision document sections as features to predict the category and the length of punishment in Indonesian courts. 
Our results demonstrate that the decision document sections that strongly affected the accuracy of the prediction model were prosecution history, facts, legal facts, and legal considerations.
"""

_HOMEPAGE = ""

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://github.com/ir-nlp-csui/indo-law/zipball/master",
}

_SUPPORTED_TASKS = [Tasks.LEGAL_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class IndoLaw(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    _LABELS = ["pidana-khusus", "pidana-umum"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo_law_source",
            version=SOURCE_VERSION,
            description="Indo-Law source schema",
            schema="source",
            subset_id="indo_law",
        ),
        NusantaraConfig(
            name="indo_law_nusantara_text",
            version=NUSANTARA_VERSION,
            description="Indo-Law Nusantara schema",
            schema="nusantara_text",
            subset_id="indo_law",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indo_law_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "klasifikasi": datasets.Value("string"),
                    "sub_klasifikasi": datasets.Value("string"),
                    "paragraphs": datasets.Sequence({
                        "tag": datasets.Value("string"),
                        "value": datasets.Value("string"),
                    }),
                }
            )

        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(self._LABELS)

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

        data_dir = os.path.join(data_dir, "ir-nlp-csui-indo-law-6734033", "dataset")

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
        files = os.listdir(filepath)

        results = []

        for file in files:
            data = self._parse_file(os.path.join(filepath, file))
            results.append(data)

        if self.config.schema == "source":
            key = 0
            for result in results:
                example = {
                    "id": result["id"],
                    "klasifikasi": result["klasifikasi"],
                    "sub_klasifikasi": result["klasifikasi"],
                    "paragraphs": [],
                }
                for tag in result["paragraphs"]:
                    example["paragraphs"].append({
                        "tag": tag,
                        "value": result["paragraphs"][tag]
                    })
                yield key, example
                key+=1

        elif self.config.schema == "nusantara_text":
            key = 0
            for result in results:
                example = {
                    "id": result["id"],
                    "text": json.dumps(result["paragraphs"]),
                    "label": result["klasifikasi"],
                }
                yield key, example
                key+=1

    def _parse_file(self, file_path):
        root = ET.parse(file_path).getroot()

        data = {
            "id": root.attrib["id"],
            "klasifikasi": root.attrib["klasifikasi"],
            "sub_klasifikasi": root.attrib["sub_klasifikasi"],
            "paragraphs": {}
        }

        for child in root:
            data["paragraphs"].update({
                child.tag: child.text
            })

        return data
                

