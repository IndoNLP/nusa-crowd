from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "nusax_senti"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "ace", "ban", "bjn", "bbc", "bug", "jav", "mad", "min", "nij", "sun", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_CITATION = """\
@misc{winata2022nusax,
      title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages}, 
      author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau, Jey Han and Sennrich, Rico and Ruder, Sebastian},
      year={2022},
      eprint={2205.15960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak.

NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.
"""

_HOMEPAGE = "https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/{lang}/train.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/{lang}/valid.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/{lang}/test.csv",
}

def builder_config_constructor(lang, schema, version):
    if schema != "source" and schema != "nusantara_text":
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="nusax_senti_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="nusax_senti with {schema} schema for language {lang}".format(lang=lang, schema=schema),
        schema=schema,
        subset_id="nusax_senti",
    )

LANGUAGES = {
        "ace": "acehnese",
        "ban": "balinese",
        "bjn": "banjarese",
        "bug": "buginese",
        "eng": "english",
        "ind": "indonesian",
        "jav": "javanese",
        "mad": "madurese",
        "min": "minangkabau",
        "nij": "ngaju",
        "sun": "sundanese",
        "bbc": "toba_batak",
    }

class NusaXSenti(datasets.GeneratorBasedBuilder):
    """NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English."""

    BUILDER_CONFIGS = [builder_config_constructor(lang, "source", _SOURCE_VERSION) for lang in LANGUAGES] + [builder_config_constructor(lang, "nusantara_text", _NUSANTARA_VERSION) for lang in LANGUAGES]

    DEFAULT_CONFIG_NAME = "nusax_senti_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["negative", "neutral", "positive"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        lang = self.config.name[12:15]
        train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"].format(lang=LANGUAGES[lang])))
        validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"].format(lang=LANGUAGES[lang])))
        test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"].format(lang=LANGUAGES[lang])))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_csv_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": validation_csv_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_csv_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema != "source" and self.config.schema != "nusantara_text":
            raise ValueError(f"Invalid config: {self.config.name}")

        df = pd.read_csv(filepath).reset_index()
        for row in df.itertuples():
            ex = {
                "id": str(row.id),
                "text": row.text,
                "label": row.label
            }
            yield row.id, ex
