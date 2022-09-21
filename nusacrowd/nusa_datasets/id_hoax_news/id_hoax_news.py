from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "id_hoax_news"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 langauge code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@INPROCEEDINGS{8265649,  author={Pratiwi, Inggrid Yanuar Risca and Asmara, Rosa Andrie and Rahutomo, Faisal},  booktitle={2017 11th International Conference on Information & Communication Technology and System (ICTS)},   title={Study of hoax news detection using naïve bayes classifier in Indonesian language},   year={2017},  volume={},  number={},  pages={73-78},  doi={10.1109/ICTS.2017.8265649}}
"""

_DESCRIPTION = """\
This research proposes to build an automatic hoax news detection and collects 250 pages of hoax and valid news articles in Indonesian language.
Each data sample is annotated by three reviewers and the final taggings are obtained by voting of those three reviewers.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/p3hfgr5j3m/1"

_LICENSE = "Creative Commons Attribution 4.0 International"

_URLs = {
    "train": "https://data.mendeley.com/public-files/datasets/p3hfgr5j3m/files/38bfcff2-8a32-4920-9c26-4f63b5b2dad8/file_downloaded",
}

_SUPPORTED_TASKS = [Tasks.HOAX_NEWS_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdHoaxNews(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_hoax_news_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Hoax News source schema",
            schema="source",
            subset_id="id_hoax_news",
        ),
        NusantaraConfig(
            name="id_hoax_news_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Hoax News Nusantara schema",
            schema="nusantara_text",
            subset_id="id_hoax_news",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_hoax_news_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "news": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["Valid", "Hoax"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        data_files = {
            "train": train_tsv_path / "250 news with valid hoax label.csv",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        news_file = open(filepath, 'r', encoding='ISO-8859-1')
        lines = news_file.readlines()
        news = []
        labels = []

        curr_news = ''
        for l in lines[1:]:
            l = l.replace('\n', '')
            if ';Valid' in l:
                curr_news += l.replace(';Valid', '')
                news.append(curr_news)
                labels.append('Valid')
                curr_news = ''
            elif ';Hoax' in l:
                curr_news += l.replace(';Hoax', '')
                news.append(curr_news)
                labels.append('Hoax')
                curr_news = ''
            else:
                curr_news += l + ' '

        if self.config.schema == "source":
            for i in range(len(news)):
                ex = {"index": str(i), "news": news[i], "label": labels[i]}
                yield i, ex
        elif self.config.schema == "nusantara_text":
            for i in range(len(news)):
                ex = {"id": str(i), "text": news[i], "label": labels[i]}
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
