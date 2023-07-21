from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "nusatranslation_mt"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["btk", "bew", "bug", "jav", "mad", "mak", "min", "mui", "rej", "sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_CITATION = """\
    @unpublished{anonymous2023nusawrites:,
    title={NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
    author={Anonymous},
    journal={OpenReview Preprint},
    year={2023},
    note={anonymous preprint under review}
    }
    """

_DESCRIPTION = """\
    Democratizing access to natural language processing (NLP) technology is crucial, especially for underrepresented and extremely low-resource languages. Previous research has focused on developing labeled and unlabeled corpora for these languages through online scraping and document translation. While these methods have proven effective and cost-efficient, we have identified limitations in the resulting corpora, including a lack of lexical diversity and cultural relevance to local communities. To address this gap, we conduct a case study on Indonesian local languages. We compare the effectiveness of online scraping, human translation, and paragraph writing by native speakers in constructing datasets. Our findings demonstrate that datasets generated through paragraph writing by native speakers exhibit superior quality in terms of lexical diversity and cultural content. In addition, we present the NusaWrites benchmark, encompassing 12 underrepresented and extremely low-resource languages spoken by millions of individuals in Indonesia. Our empirical experiment results using existing multilingual large language models conclude the need to extend these models to more underrepresented languages.
    We introduce a novel high quality human curated corpora, i.e., NusaMenulis, which covers 12 languages spoken in Indonesia. The resource extend the coverage of languages to 5 new languages, i.e., Ambon (abs), Bima (bhp), Makassarese (mak), Palembang / Musi (mui), and Rejang (rej).
    For the rhetoric mode classification task, we cover 5 rhetoric modes, i.e., narrative, persuasive, argumentative, descriptive, and expository.
"""

_HOMEPAGE = "https://github.com/IndoNLP/nusatranslation/tree/main/datasets/mt"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_kalimat-mt-{lang}-train.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_kalimat-mt-{lang}-valid.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_kalimat-mt-{lang}-test.csv",
}


def nusantara_config_constructor(lang, schema, version):
    """Construct NusantaraConfig with nusatranslation_mt_{lang}_{schema} as the name format"""
    if schema != "source" and schema != "nusantara_t2t":
        raise ValueError(f"Invalid schema: {schema}")

    if lang == "":
        return NusantaraConfig(
            name="nusatranslation_mt_{schema}".format(schema=schema),
            version=datasets.Version(version),
            description="nusatranslation_mt with {schema} schema for all 12 languages".format(schema=schema),
            schema=schema,
            subset_id="nusatranslation_mt",
        )
    else:
        return NusantaraConfig(
            name="nusatranslation_mt_{lang}_{schema}".format(lang=lang, schema=schema),
            version=datasets.Version(version),
            description="nusatranslation_mt with {schema} schema for {lang} language".format(lang=lang, schema=schema),
            schema=schema,
            subset_id="nusatranslation_mt",
        )


LANGUAGES_MAP = {
    "abs": "ambon",
    "btk": "batak",
    "bew": "betawi",
    "bhp": "bima",
    "jav": "javanese",
    "mad": "madurese",
    "mak": "makassarese",
    "min": "minangkabau",
    "mui": "musi",
    "rej": "rejang",
    "sun": "sundanese",
}


class NusaTranslationMT(datasets.GeneratorBasedBuilder):
    """NusaTranslation-MT is a parallel corpus for training and benchmarking machine translation models from 11 Indonesian local language to Bahasa Indonesia. The data is presented in csv format with 2 columns, where one column contain sentence in Bahasa and another in the local language."""

    BUILDER_CONFIGS = (
        [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in LANGUAGES_MAP]
        + [nusantara_config_constructor(lang, "nusantara_t2t", _NUSANTARA_VERSION) for lang in LANGUAGES_MAP]
        + [nusantara_config_constructor("", "source", _SOURCE_VERSION), nusantara_config_constructor("", "nusantara_t2t", _NUSANTARA_VERSION)]
    )

    DEFAULT_CONFIG_NAME = "nusatranslation_mt_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source" or self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.name == "nusatranslation_mt_source" or self.config.name == "nusatranslation_mt_nusantara_t2t":
            # Load all 12 languages
            train_csv_path = dl_manager.download_and_extract([_URLS["train"].format(lang=lang) for lang in LANGUAGES_MAP])
            validation_csv_path = dl_manager.download_and_extract([_URLS["validation"].format(lang=lang) for lang in LANGUAGES_MAP])
            test_csv_path = dl_manager.download_and_extract([_URLS["test"].format(lang=lang) for lang in LANGUAGES_MAP])
        else:
            lang = self.config.name.split("_")[2]
            train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"].format(lang=lang)))
            validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"].format(lang=lang)))
            test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"].format(lang=lang)))

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
        if self.config.schema != "source" and self.config.schema != "nusantara_t2t":
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        # print(filepath)
        # df = pd.read_csv(filepath).reset_index()
        if self.config.name == "nusatranslation_mt_source" or self.config.name == "nusatranslation_mt_nusantara_t2t":
            # load all 132 language pairs
            ldf = []
            for fp in filepath:
                ldf.append(pd.read_csv(fp))
            df = pd.concat(ldf, axis=0, ignore_index=True).reset_index()
            # Have to use index instead of id to avoid duplicated key
            df = df.drop(columns=["id"]).rename(columns={"index": "id"})

            id_count = -1
            for lang in LANGUAGES_MAP:
                for _, row in df.iterrows():
                    id_count += 1
                    ex = {
                        "id": str(id_count),
                        "text_1": row["ind_text"],
                        "text_2": row["tgt_text"],
                        "text_1_name": "indonesian",
                        "text_2_name": LANGUAGES_MAP[lang],
                    }
                    yield id_count, ex

        else:
            df = pd.read_csv(filepath).reset_index()
            lang = self.config.name.split("_")[2]

            for index, row in df.iterrows():
                ex = {
                    "id": str(index),
                    "text_1": row["ind_text"],
                    "text_2": row["tgt_text"],
                    "text_1_name": "indonesian",
                    "text_2_name": LANGUAGES_MAP[lang],
                }
                yield str(index), ex
