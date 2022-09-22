from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "emotcmt"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{barik-etal-2019-normalization,
    title = "Normalization of {I}ndonesian-{E}nglish Code-Mixed {T}witter Data",
    author = "Barik, Anab Maulana  and
      Mahendra, Rahmad  and
      Adriani, Mirna",
    booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5554",
    doi = "10.18653/v1/D19-5554",
    pages = "417--424"
}

@article{Yulianti2021NormalisationOI,
  title={Normalisation of Indonesian-English Code-Mixed Text and its Effect on Emotion Classification},
  author={Evi Yulianti and Ajmal Kurnia and Mirna Adriani and Yoppy Setyo Duto},
  journal={International Journal of Advanced Computer Science and Applications},
  year={2021}
}
"""

_DESCRIPTION = """\
EmotCMT is an emotion classification Indonesian-English code-mixing dataset created through an Indonesian-English code-mixed Twitter data pipeline consisting of 4 processing steps, i.e., tokenization, language identification, lexical normalization, and translation. The dataset consists of 825 tweets, 22.736 tokens with 11.204 Indonesian tokens and 5.613 English tokens. Each tweet is labelled with an emotion, i.e., cinta (love), takut (fear), sedih (sadness), senang (joy), or marah (anger).
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/emotcmt"

_LICENSE = "MIT"

_URLs = {
    "test": "https://raw.githubusercontent.com/ir-nlp-csui/emotcmt/main/codeswitch_emotion.csv"
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class EmotCMT(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="emotcmt_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="EmotCMT source schema",
            schema="source",
            subset_id="emotcmt",
        ),
        NusantaraConfig(
            name="emotcmt_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="EmotCMT Nusantara schema",
            schema="nusantara_text",
            subset_id="emotcmt",
        ),
    ]

    DEFAULT_CONFIG_NAME = "emotcmt_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"tweet": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["cinta", "takut", "sedih", "senang", "marah"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        test_csv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "test": test_csv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            )
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_csv(filepath).reset_index()
        df.columns = ["id", "label", "sentence"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"tweet": row.sentence, "label": row.label}
                yield row.id, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.sentence,
                    "label": row.label
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
