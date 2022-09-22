from pathlib import Path
from typing import List

import datasets
import json

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "news_en_id"
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
    language = "English",
    ISBN = "979-10-95546-42-9",
}
"""

_DESCRIPTION = """\
News En-Id is a machine translation dataset containing Indonesian-English parallel sentences collected from the news. The news dataset is collected from multiple sources: Pan Asia Networking Localization (PANL), Bilingual BBC news articles, Berita Jakarta, and GlobalVoices. We split the dataset and use 75% as the training set, 10% as the validation set, and 15% as the test set. Each of the datasets is evaluated in both directions, i.e., English to Indonesian (En → Id) and Indonesian to English (Id → En) translations.
"""

_HOMEPAGE = "https://github.com/gunnxx/indonesian-mt-data"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {"indonlg": "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip"}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class NewsEnId(datasets.GeneratorBasedBuilder):
    """Bible Su-Id is a machine translation dataset containing Indonesian-Sundanese parallel sentences collected from the bible.."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="news_en_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="News En-Id source schema",
            schema="source",
            subset_id="news_en_id",
        ),
        NusantaraConfig(
            name="news_en_id_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="News En-Id Nusantara schema",
            schema="nusantara_t2t",
            subset_id="news_en_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "news_en_id_source"

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
        base_path = Path(dl_manager.download_and_extract(_URLs["indonlg"])) / "IndoNLG_downstream_tasks" / "MT_IMD_NEWS"
        data_files = {
            "train": base_path / "train_preprocess.json",
            "validation": base_path / "valid_preprocess.json",
            "test": base_path / "test_preprocess.json",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        data = json.load(open(filepath, "r"))
        if self.config.schema == "source":
            for row in data:
                ex = {"id": row["id"], "text": row["text"], "label": row["label"]}
                yield row["id"], ex
        elif self.config.schema == "nusantara_t2t":
            for row in data:
                ex = {
                    "id": row["id"],
                    "text_1": row["text"],
                    "text_2": row["label"],
                    "text_1_name": "eng",
                    "text_2_name": "ind",
                }
                yield row["id"], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
