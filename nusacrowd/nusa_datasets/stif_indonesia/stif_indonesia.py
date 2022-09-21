from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "stif_indonesia"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{wibowo2020semi,
  title={Semi-supervised low-resource style transfer of indonesian informal to formal language with iterative forward-translation},
  author={Wibowo, Haryo Akbarianto and Prawiro, Tatag Aziz and Ihsan, Muhammad and Aji, Alham Fikri and Prasojo, Radityo Eko and Mahendra, Rahmad and Fitriany, Suci},
  booktitle={2020 International Conference on Asian Language Processing (IALP)},
  pages={310--315},
  year={2020},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
STIF-Indonesia is formal-informal (bahasa baku - bahasa alay/slang) style transfer for Indonesian. Texts were collected from Twitter. Then, native speakers were aksed to transform the text into formal style.
"""

_HOMEPAGE = "https://github.com/haryoa/stif-indonesia"

_LICENSE = "MIT"

_BASEURL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/main/data/labelled/"
_URLs = {
    "dev.for": _BASEURL + "dev.for",
    "dev.inf": _BASEURL + "dev.inf",
    "test.for": _BASEURL + "test.for",
    "test.inf": _BASEURL + "test.inf",
    "train.for": _BASEURL + "train.for",
    "train.inf": _BASEURL + "train.inf",
}

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class STIFIndonesia(datasets.GeneratorBasedBuilder):
    """STIF-Indonesia is formal-informal/colloquial style transfer for Indonesian."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="stif_indonesia_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="STIF Indonesia source schema",
            schema="source",
            subset_id="stif_indonesia",
        ),
        NusantaraConfig(
            name="stif_indonesia_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="STIF Indonesia Nusantara schema",
            schema="nusantara_t2t",
            subset_id="stif_indonesia",
        ),
    ]

    DEFAULT_CONFIG_NAME = "stif_indonesia_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "formal": datasets.Value("string"), "informal": datasets.Value("string")})
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
        data_files = {}
        for key in _URLs:
            data_files[key] = Path(dl_manager.download_and_extract(_URLs[key]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "formal": data_files["test.for"],
                        "informal": data_files["test.inf"],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {
                        "formal": data_files["dev.for"],
                        "informal": data_files["dev.inf"],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "formal": data_files["train.for"],
                        "informal": data_files["train.inf"],
                    }
                },
            ),
        ]

    def _generate_examples(self, filepath: Path):
        data_for = open(filepath["formal"], "r").readlines()
        data_inf = open(filepath["informal"], "r").readlines()

        if self.config.schema == "source":
            for id, (row_for, row_inf) in enumerate(zip(data_for, data_inf)):
                ex = {"id": id, "formal": row_for.strip(), "informal": row_inf.strip()}
                yield id, ex
        elif self.config.schema == "nusantara_t2t":
            for id, (row_for, row_inf) in enumerate(zip(data_for, data_inf)):
                ex = {
                    "id": id,
                    "text_1": row_for.strip(),
                    "text_2": row_inf.strip(),
                    "text_1_name": "formal",
                    "text_2_name": "informal",
                }
                yield id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
