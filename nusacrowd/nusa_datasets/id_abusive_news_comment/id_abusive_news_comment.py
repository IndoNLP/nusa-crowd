from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "id_abusive_news_comment"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 langauge code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@INPROCEEDINGS{9034620,  author={Kiasati Desrul, Dhamir Raniah and Romadhony, Ade},  booktitle={2019 International Seminar on Research of Information Technology and Intelligent Systems (ISRITI)},   title={Abusive Language Detection on Indonesian Online News Comments},   year={2019},  volume={},  number={},  pages={320-325},  doi={10.1109/ISRITI48646.2019.9034620}}
"""

_DESCRIPTION = """\
Abusive language is an expression used by a person with insulting delivery of any person's aspect.
In the modern era, the use of harsh words is often found on the internet, one of them is in the comment section of online news articles which contains harassment, insult, or a curse.
An abusive language detection system is important to prevent the negative effect of such comments.
This dataset contains 3184 samples of Indonesian online news comments with 3 labels.
"""

_HOMEPAGE = "https://github.com/dhamirdesrul/Indonesian-Online-News-Comments"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://github.com/dhamirdesrul/Indonesian-Online-News-Comments/raw/master/Dataset/Abusive%20Language%20Detection%20on%20Indonesian%20Online%20News%20Comments%20Dataset%20.xlsx",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdAbusiveNewsComment(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_abusive_news_comment_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Abusive Online News Comment source schema",
            schema="source",
            subset_id="id_abusive_news_comment",
        ),
        NusantaraConfig(
            name="id_abusive_news_comment_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Abusive Online News Comment Nusantara schema",
            schema="nusantara_text",
            subset_id="id_abusive_news_comment",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_abusive_news_comment"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(['1', '2', '3'])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download(_URLs["train"]))
        data_files = {
            "train": train_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_excel(filepath).reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": str(row.index), "text": row.Kalimat, "label": str(row.label)}
                yield row.index, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {"id": str(row.index), "text": row.Kalimat, "label": str(row.label)}
                yield row.index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
