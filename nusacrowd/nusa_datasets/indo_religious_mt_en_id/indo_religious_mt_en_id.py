from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "indo_religious_mt_en_id"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{guntara-etal-2020-benchmarking,
    title = "Benchmarking Multidomain {E}nglish-{I}ndonesian Machine Translation",
    author = "Guntara, Tri Wahyu  and
      Aji, Alham Fikri  and
      Prasojo, Radityo Eko",
    booktitle = "Proceedings of the 13th Workshop on Building and Using Comparable Corpora",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.bucc-1.6",
    pages = "35--43",
    abstract = "In the context of Machine Translation (MT) from-and-to English, Bahasa Indonesia has been considered a low-resource language, and therefore applying Neural Machine Translation (NMT) which typically requires large training dataset proves to be problematic. In this paper, we show otherwise by collecting large, publicly-available datasets from the Web, which we split into several domains: news, religion, general, and conversation, to train and benchmark some variants of transformer-based NMT models across the domains. We show using BLEU that our models perform well across them , outperform the baseline Statistical Machine Translation (SMT) models, and perform comparably with Google Translate. Our datasets (with the standard split for training, validation, and testing), code, and models are available on https://github.com/gunnxx/indonesian-mt-data.",
    language = "English",
    ISBN = "979-10-95546-42-9",
}
"""

_DESCRIPTION = """\
Indonesian Religious Domain MT En-Id consists of religious manuscripts or articles. These articles are different from news as they are not in a formal, informative style. Instead, they are written to advocate and inspire religious values, often times citing biblical or quranic anecdotes. An interesting property in the religion domain corpus is the localized names, for example, David to Daud, Mary to Maryam, Gabriel to Jibril, and more. In contrast, entity names are usually kept unchanged in other domains. We also find quite a handful of Indonesian translations of JW300 are missing the end sentence dot (.), even though the end sentence dot is present in their English counterpart. Some inconsistencies in the transliteration are also found, for example praying is sometimes written as \"salat\" or \"shalat\", or repentance as \"tobat\" or \"taubat\".
"""

_HOMEPAGE = "https://github.com/gunnxx/indonesian-mt-data/tree/master/religious"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "test.en": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/test.en",
    "test.id": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/test.id",
    "valid.en": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/valid.en",
    "valid.id": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/valid.id",
    "train.en.0": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/train.en.0",
    "train.en.1": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/train.en.1",
    "train.id.0": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/train.id.0",
    "train.id.1": "https://raw.githubusercontent.com/gunnxx/indonesian-mt-data/master/religious/train.id.1",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndoReligiousMTEnId(datasets.GeneratorBasedBuilder):
    """Indonesian Religious Domain MT En-Id is a machine translation dataset containing English-Indonesian parallel sentences collected from the religious manuscripts."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo_religious_mt_en_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Bible En-Id source schema",
            schema="source",
            subset_id="indo_religious_mt_en_id",
        ),
        NusantaraConfig(
            name="indo_religious_mt_en_id_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Bible En-Id Nusantara schema",
            schema="nusantara_t2t",
            subset_id="indo_religious_mt_en_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indo_religious_mt_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                }
            )
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
        data_files = {
            "test.en": Path(dl_manager.download_and_extract(_URLs["test.en"])),
            "test.id": Path(dl_manager.download_and_extract(_URLs["test.id"])),
            "valid.en": Path(dl_manager.download_and_extract(_URLs["valid.en"])),
            "valid.id": Path(dl_manager.download_and_extract(_URLs["valid.id"])),
            "train.en.0": Path(dl_manager.download_and_extract(_URLs["train.en.0"])),
            "train.en.1": Path(dl_manager.download_and_extract(_URLs["train.en.1"])),
            "train.id.0": Path(dl_manager.download_and_extract(_URLs["train.id.0"])),
            "train.id.1": Path(dl_manager.download_and_extract(_URLs["train.id.1"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "en": [data_files["test.en"]],
                        "id": [data_files["test.id"]],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {
                        "en": [data_files["valid.en"]],
                        "id": [data_files["valid.id"]],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "en": [data_files["train.en.0"], data_files["train.en.1"]],
                        "id": [data_files["train.id.0"], data_files["train.id.1"]],
                    }
                },
            ),
        ]

    def _generate_examples(self, filepath: dict):

        data_en = None
        for file in filepath["en"]:
            if data_en is None:
                data_en = open(file, "r").readlines()
            else:
                data_en += open(file, "r").readlines()

        data_id = None
        for file in filepath["id"]:
            if data_id is None:
                data_id = open(file, "r").readlines()
            else:
                data_id += open(file, "r").readlines()

        if self.config.schema == "source":
            for id, (row_en, row_id) in enumerate(zip(data_en, data_id)):
                ex = {
                    "text_1": row_en,
                    "text_2": row_id,
                }
                yield id, ex
        elif self.config.schema == "nusantara_t2t":
            for id, (row_en, row_id) in enumerate(zip(data_en, data_id)):
                ex = {
                    "id": id,
                    "text_1": row_en,
                    "text_2": row_id,
                    "text_1_name": "eng",
                    "text_2_name": "ind",
                }
                yield id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
