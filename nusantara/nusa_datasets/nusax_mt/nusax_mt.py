from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "nusax_mt"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "ace", "ban", "bjn", "bbc", "bug", "jav", "mad", "min", "nij", "sun", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_CITATION = """\
@misc{winata2022nusax,
      title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
      author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya,
      Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony,
      Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo,
      Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau,
      Jey Han and Sennrich, Rico and Ruder, Sebastian},
      year={2022},
      eprint={2205.15960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak.

NusaX-MT is a parallel corpus for training and benchmarking machine translation models across 10 Indonesian local languages + Indonesian and English. The data is presented in csv format with 12 columns, one column for each language.
"""

_HOMEPAGE = "https://github.com/IndoNLP/nusax/tree/main/datasets/mt"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/{lang}/train.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/{lang}/valid.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/{lang}/test.csv",
}


def nusantara_config_constructor(lang_source, lang_target, schema, version):
    """Construct NusantaraConfig with nusax_mt_{lang_source}_{lang_target}_{schema} as the name format"""
    if schema != "source" and schema != "nusantara_text":
        raise ValueError(f"Invalid schema: {schema}")

    if lang == "":
        return NusantaraConfig(
            name="nusax_mt_{schema}".format(schema=schema),
            version=datasets.Version(version),
            description="nusax_mt with {schema} schema for all 132 language pairs".format(schema=schema),
            schema=schema,
            subset_id="nusax_mt",
        )
    else:
        return NusantaraConfig(
            name="nusax_mt_{lang_source}_{lang_target}_{schema}".format(lang_source=lang_source, lang_target=lang_target, schema=schema),
            version=datasets.Version(version),
            description="nusax_mt with {schema} schema for {lang} language".format(lang=lang, schema=schema),
            schema=schema,
            subset_id="nusax_mt",
        )


LANGUAGES_MAP = {
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


class NusaXMT(datasets.GeneratorBasedBuilder):
    """NusaX-MT is a parallel corpus for training and benchmarking machine translation models across 10 Indonesian local languages + Indonesian and English. The data is presented in csv format with 12 columns, one column for each language."""

    BUILDER_CONFIGS = (
        [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in LANGUAGES_MAP]
        + [nusantara_config_constructor(lang, "nusantara_text", _NUSANTARA_VERSION) for lang in LANGUAGES_MAP]
        + [nusantara_config_constructor("", "source", _SOURCE_VERSION), nusantara_config_constructor("", "nusantara_text", _NUSANTARA_VERSION)]
    )

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
        if self.config.name == "nusax_senti_source" or self.config.name == "nusax_senti_nusantara_text":
            # Load all 12 languages
            train_csv_path = dl_manager.download_and_extract([_URLS["train"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
            validation_csv_path = dl_manager.download_and_extract([_URLS["validation"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
            test_csv_path = dl_manager.download_and_extract([_URLS["test"].format(lang=LANGUAGES_MAP[lang]) for lang in LANGUAGES_MAP])
        else:
            lang = self.config.name[12:15]
            train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"].format(lang=LANGUAGES_MAP[lang])))
            validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"].format(lang=LANGUAGES_MAP[lang])))
            test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"].format(lang=LANGUAGES_MAP[lang])))

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

        if self.config.name == "nusax_senti_source" or self.config.name == "nusax_senti_nusantara_text":
            ldf = []
            for fp in filepath:
                ldf.append(pd.read_csv(fp))
            df = pd.concat(ldf, axis=0, ignore_index=True).reset_index()
            # Have to use index instead of id to avoid duplicated key
            df = df.drop(columns=["id"]).rename(columns={"index": "id"})
        else:
            df = pd.read_csv(filepath).reset_index()

        for row in df.itertuples():
            ex = {"id": str(row.id), "text": row.text, "label": row.label}
            yield row.id, ex
