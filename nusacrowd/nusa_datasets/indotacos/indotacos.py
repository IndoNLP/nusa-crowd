from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import pandas as pd

_CITATION = """\
    author = "Christian Wibisono"
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indotacos"

_DESCRIPTION = """\
Predicting the outcome or the probability of winning a legal case has always been highly attractive in legal sciences and practice.
Hardly any dataset has been developed to analyze and accelerate the research of court verdict analysis.
Find out what factor affects the outcome of tax court verdict using Natural Language Processing.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/christianwbsn/indonesia-tax-court-verdict"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
        # "indotacos": "christianwbsn/indotacos"
        "indotacos": "https://huggingface.co/datasets/christianwbsn/indotacos/resolve/main/indonesia_tax_court_verdict.csv"
        # ?datasetVersionNumber=1/archive.zip"
    }
}

_SUPPORTED_TASKS = [Tasks.TAX_COURT_VERDICT]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IndoTacos(datasets.GeneratorBasedBuilder):
    """IndoNLI, a human-elicited NLI dataset for Indonesian containing ~18k sentence pairs annotated by crowd workers."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indotacos_source",
            version=SOURCE_VERSION,
            description="indotacos source schema",
            schema="source",
            subset_id="indotacos",
        ),
        NusantaraConfig(
            name="indotacos_nusantara_text",
            version=NUSANTARA_VERSION,
            description="IndoTacos Nusantara schema",
            schema="nusantara_text",
            subset_id="indotacos",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indotacos_source"
    labels = ["mengabulkan sebagian", "mengabulkan seluruhnya", "menolak", "lain-lain", "menambah pajak", "mengabulkan", "membetulkan"]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "nomor_putusan": datasets.Value("string"),
                    "tahun_pajak": datasets.Value("int32"),
                    "jenis_pajak": datasets.Value("string"),
                    "tahun_putusan": datasets.Value("int32"),
                    "pokok_sengketa": datasets.Value("string"),
                    "pokok_sengketa": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.pairs_features(self.labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        url = _URLS["indotacos"]
        # path = dl_manager.download_and_extract(url) + "/indonesia_tax_court_verdict.csv"

        # data_files = {"train": "indonesia_tax_court_verdict.csv"}

        path = dl_manager.download(url)["indotacos"]
        data_files = {"train": path}
        # path = data

        print(path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                },
            )
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_csv(filepath)
        # print(df)
        if self.config.schema == "source":
            for row in df.itertuples():
                # print(row)
                ex = {
                    "text": row.text,
                    "nomor_putusan": row.nomor_putusan,
                    "tahun_pajak": row.tahun_pajak,
                    "jenis_pajak": row.jenis_pajak,
                    "tahun_putusan": row.tahun_putusan,
                    "pokok_sengketa": row.pokok_sengketa,
                    "pokok_sengketa": row.pokok_sengketa,
                }
                yield row.Index, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "text": row.text,
                    "nomor_putusan": row.nomor_putusan,
                    "tahun_pajak": row.tahun_pajak,
                    "jenis_pajak": row.jenis_pajak,
                    "tahun_putusan": row.tahun_putusan,
                    "pokok_sengketa": row.pokok_sengketa,
                    "pokok_sengketa": row.pokok_sengketa,
                }
                yield row.Index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
