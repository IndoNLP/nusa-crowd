from pathlib import Path
from typing import List

import datasets
import json

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "parallel_su_id"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@INPROCEEDINGS{7437678,
  author={Suryani, Arie Ardiyanti and Widyantoro, Dwi Hendratmo and Purwarianti, Ayu and Sudaryat, Yayat},
  booktitle={2015 International Conference on Information Technology Systems and Innovation (ICITSI)}, 
  title={Experiment on a phrase-based statistical machine translation using PoS Tag information for Sundanese into Indonesian}, 
  year={2015},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICITSI.2015.7437678}}
"""

_DESCRIPTION = """\
This data contains 3616 lines of Sundanese sentences taken from the online Sundanese language magazine Mangle, West Java Dakwah Council, and Balebat, and translated into Indonesian by several students of the Sundanese language study program UPI Bandung.
"""

_HOMEPAGE = "https://dataverse.telkomuniversity.ac.id/dataset.xhtml?persistentId=doi:10.34820/FK2/HDYWXW"

_LICENSE = "Creative Commons CC0 - No Rights Reserved"

_URLs = {"ind": "https://dataverse.telkomuniversity.ac.id/api/access/datafile/:persistentId?persistentId=doi:10.34820/FK2/HDYWXW/032QZD",
         "sun": "https://dataverse.telkomuniversity.ac.id/api/access/datafile/:persistentId?persistentId=doi:10.34820/FK2/HDYWXW/IVP3G5"}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class ParallelSuId(datasets.GeneratorBasedBuilder):
    """Parallel Su-Id is a machine translation dataset containing Indonesian-Sundanese parallel sentences collected from the online Sundanese language magazine Mangle, West Java Dakwah Council, and Balebat."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="parallel_su_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Parallel Su-Id source schema",
            schema="source",
            subset_id="parallel_su_id",
        ),
        NusantaraConfig(
            name="parallel_su_id_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Parallel Su-Id Nusantara schema",
            schema="nusantara_t2t",
            subset_id="parallel_su_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "parallel_su_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string")})
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
        ind_path = Path(dl_manager.download_and_extract(_URLs["ind"]))
        sun_path = Path(dl_manager.download_and_extract(_URLs["sun"]))
        data_files = {
            "ind": ind_path,
            "sun": sun_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath_dict": data_files},
            )
        ]

    def _generate_examples(self, filepath_dict):
        data = {}
        for lang, path in filepath_dict.items():
            file = open(path, "r")
            data[lang] = []
            for line in file:
                data[lang].append(line)
        if self.config.schema == "source":
            for i in range(len(data[lang])):
                ex = {
                      "id": i, 
                      "text": data['sun'][i].replace("\n",""),
                      "label": data['ind'][i].replace("\n","")
                }
                yield i, ex
        elif self.config.schema == "nusantara_t2t":
            for i in range(len(data[lang])):
                ex = {
                    "id": i,
                    "text_1": data['sun'][i].replace("\n",""),
                    "text_2": data['ind'][i].replace("\n",""),
                    "text_1_name": "sun",
                    "text_2_name": "ind",
                }
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
