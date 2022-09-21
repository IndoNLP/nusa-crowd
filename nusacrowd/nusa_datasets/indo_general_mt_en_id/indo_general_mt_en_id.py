from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

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
    language = "English",
    ISBN = "979-10-95546-42-9",
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indo_general_mt_en_id"
_DESCRIPTION = """\
"In the context of Machine Translation (MT) from-and-to English, Bahasa Indonesia has been considered a low-resource language,
and therefore applying Neural Machine Translation (NMT) which typically requires large training dataset proves to be problematic.
In this paper, we show otherwise by collecting large, publicly-available datasets from the Web, which we split into several domains: news, religion, general, and
conversation,to train and benchmark some variants of transformer-based NMT models across the domains.
We show using BLEU that our models perform well across them , outperform the baseline Statistical Machine Translation (SMT) models,
and perform comparably with Google Translate. Our datasets (with the standard split for training, validation, and testing), code, and models are available on https://github.com/gunnxx/indonesian-mt-data."
"""

_HOMEPAGE = "https://github.com/gunnxx/indonesian-mt-data"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLS = {
    _DATASETNAME: "https://github.com/gunnxx/indonesian-mt-data/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
# Dataset does not have versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndoGeneralMTEnId(datasets.GeneratorBasedBuilder):
    """Indonesian General Domain MT En-Id is a machine translation dataset containing English-Indonesian parallel sentences collected from the general manuscripts."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo_general_mt_en_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Indonesian General Domain MT En-Id source schema",
            schema="source",
            subset_id="indo_general_mt_en_id",
        ),
        NusantaraConfig(
            name="indo_general_mt_en_id_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Indonesian General Domain MT Nusantara schema",
            schema="nusantara_t2t",
            subset_id="indo_general_mt_en_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indo_general_mt_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": datasets.Value("string"),
                    "tgt": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls)) / "indonesian-mt-data-master" / "general"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "en": [data_dir / "train.en.0", data_dir / "train.en.1", data_dir / "train.en.2", data_dir / "train.en.3"],
                        "id": [data_dir / "train.id.0", data_dir / "train.id.1", data_dir / "train.id.2", data_dir / "train.id.3"],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "en": [data_dir / "test.en"],
                        "id": [data_dir / "test.id"],
                    }
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {
                        "en": [data_dir / "valid.en"],
                        "id": [data_dir / "valid.id"],
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

        data_en = list(map(str.strip, data_en))
        data_id = list(map(str.strip, data_id))

        if self.config.schema == "source":
            for id, (src, tgt) in enumerate(zip(data_en, data_id)):
                row = {
                    "id": str(id),
                    "src": src,
                    "tgt": tgt,
                }
                yield id, row
        elif self.config.schema == "nusantara_t2t":
            for id, (src, tgt) in enumerate(zip(data_en, data_id)):
                row = {
                    "id": str(id),
                    "text_1": src,
                    "text_2": tgt,
                    "text_1_name": "eng",
                    "text_2_name": "ind",
                }
                yield id, row
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
