from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{IBROHIM2018222,
title = {A Dataset and Preliminaries Study for Abusive Language Detection in Indonesian Social Media},
journal = {Procedia Computer Science},
volume = {135},
pages = {222-229},
year = {2018},
note = {The 3rd International Conference on Computer Science and Computational Intelligence (ICCSCI 2018) : Empowering Smart Technology in Digital Era for a Better Life},
issn = {1877-0509},
doi = {https://doi.org/10.1016/j.procs.2018.08.169},
url = {https://www.sciencedirect.com/science/article/pii/S1877050918314583},
author = {Muhammad Okky Ibrohim and Indra Budi},
keywords = {abusive language, twitter, machine learning},
abstract = {Abusive language is an expression (both oral or text) that contains abusive/dirty words or phrases both in the context of jokes, a vulgar sex conservation or to cursing someone. Nowadays many people on the internet (netizens) write and post an abusive language in the social media such as Facebook, Line, Twitter, etc. Detecting an abusive language in social media is a difficult problem to resolve because this problem can not be resolved just use word matching. This paper discusses a preliminaries study for abusive language detection in Indonesian social media and the challenge in developing a system for Indonesian abusive language detection, especially in social media. We also built reported an experiment for abusive language detection on Indonesian tweet using machine learning approach with a simple word n-gram and char n-gram features. We use Naive Bayes, Support Vector Machine, and Random Forest Decision Tree classifier to identify the tweet whether the tweet is a not abusive language, abusive but not offensive, or offensive language. The experiment results show that the Naive Bayes classifier with the combination of word unigram + bigrams features gives the best result i.e. 70.06% of F1 - Score. However, if we classifying the tweet into two labels only (not abusive language and abusive language), all classifier that we used gives a higher result (more than 83% of F1 - Score for every classifier). The dataset in this experiment is available for other researchers that interest to improved this study.}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "id_abusive"

_DESCRIPTION = """\
The ID_ABUSIVE dataset is collection of 2,016 informal abusive tweets in Indonesian language,
designed for sentiment analysis NLP task. This dataset is crawled from Twitter, and then filtered
and labelled manually by 20 volunteer annotators. The dataset labelled into three labels namely
not abusive language, abusive but not offensive, and offensive language.
"""

_HOMEPAGE = "https://www.sciencedirect.com/science/article/pii/S1877050918314583"
_LICENSE = "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International"
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/okkyibrohim/id-abusive-language-detection/master/re_dataset_three_labels.csv",
}
_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdAbusive(datasets.GeneratorBasedBuilder):
    """The ID_Abusive dataset is collection of Indonesian informal abusive tweets, annotated into three labels namely
    not abusive language, abusive but not offensive, and offensive language."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    LABEL_STRING_MAP = {
        1: "not_abusive",
        2: "abusive",
        3: "abusive_and_offensive",
    }

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_abusive_source",
            version=SOURCE_VERSION,
            description="ID Abusive source schema",
            schema="source",
            subset_id="id_abusive",
        ),
        NusantaraConfig(
            name="id_abusive_nusantara_text",
            version=NUSANTARA_VERSION,
            description="ID Abusive Nusantara schema",
            schema="nusantara_text",
            subset_id="id_abusive",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_abusive_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"tweet": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["not_abusive", "abusive", "abusive_and_offensive"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # Dataset does not have predetermined split, putting all as TRAIN
        urls = _URLS[_DATASETNAME]
        base_dir = Path(dl_manager.download_and_extract(urls))
        data_files = {"train": base_dir}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset does not have id, using row index as id
        df = pd.read_csv(filepath).reset_index()
        df.columns = ["id", "label", "tweet"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "tweet": row.tweet,
                    "label": self.LABEL_STRING_MAP[row.label],
                }
                yield row.id, ex

        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.tweet,
                    "label": self.LABEL_STRING_MAP[row.label],
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
