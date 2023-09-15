from pathlib import Path
from typing import Dict, List, Tuple
import re


import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "nusatranslation_mt"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "btk", "bew", "bug", "jav", "mad", "mak", "min", "mui", "rej", "sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
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
        [
            NusantaraConfig(
                name=f"nusatranslation_mt_ind_{subset}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"nusatranslation_mt ind2{subset} source schema",
                schema="source",
                subset_id=f"nusatranslation_mt",
            )
            for subset in _LANGUAGES[1:]
        ]
        + [
            NusantaraConfig(
                name=f"nusatranslation_mt_ind_{subset}_nusantara_t2t",
                version=datasets.Version(_NUSANTARA_VERSION),
                description=f"nusatranslation_mt ind2{subset} Nusantara schema",
                schema="nusantara_t2t",
                subset_id=f"nusatranslation_mt",
            )
            for subset in _LANGUAGES[1:]
        ]
        + [
            NusantaraConfig(
                name=f"nusatranslation_mt_{subset}_ind_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"nusatranslation_mt {subset}2ind source schema",
                schema="source",
                subset_id=f"nusatranslation_mt",
            )
            for subset in _LANGUAGES[1:]
        ]
        + [
            NusantaraConfig(
                name=f"nusatranslation_mt_{subset}_ind_nusantara_t2t",
                version=datasets.Version(_NUSANTARA_VERSION),
                description=f"nusatranslation_mt {subset}2ind Nusantara schema",
                schema="nusantara_t2t",
                subset_id=f"nusatranslation_mt",
            )
            for subset in _LANGUAGES[1:]
        ]
    )

    DEFAULT_CONFIG_NAME = "nusatranslation_mt_jav_ind_source"

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
        """Returns SplitGenerators."""
        lang = self.config.name.split("_")[2] if self.config.name.split("_")[2] != "ind" else self.config.name.split("_")[3]
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

    def _merge_subsets(self, df, subsets, revert=False):
        if not subsets:
            return None
        # df = None
        # print(dfs)
        # print(subsets)
        orig_columns = df.columns.tolist()
        print(df.columns)

        df.columns = orig_columns[:1] + ["label", "text"] if revert else orig_columns[:1] + ["text", "label"]
        return df

    def get_domain_data(self, dfs):
        domain = self.config.name
        matched_domain = re.findall(r"nusatranslation_mt_.*?_.*?_", domain)

        assert len(matched_domain) == 1
        domain = matched_domain[0][:-1].replace("nusatranslation_mt_", "").split("_")
        src_lang, tgt_lang = domain[0], domain[1]

        subsets = LANGUAGES_MAP.get(src_lang if src_lang != "ind" else tgt_lang, None)
        return src_lang, tgt_lang, self._merge_subsets(dfs, subsets, revert=(src_lang != "ind"))

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema != "source" and self.config.schema != "nusantara_t2t":
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        # print(filepath)
        df = pd.read_csv(filepath)
        # ldf = []
        # for fp in filepath:
        #     ldf.append(pd.read_csv(fp))
        src_lang, tgt_lang, df = self.get_domain_data((df))

        if self.config.schema == "source":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text": row.text,
                    "label": row.label,
                }
                yield idx, ex

        elif self.config.schema == "nusantara_t2t":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text_1": row.text,
                    "text_2": row.label,
                    "text_1_name": src_lang,
                    "text_2_name": tgt_lang,
                }
                yield idx, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
