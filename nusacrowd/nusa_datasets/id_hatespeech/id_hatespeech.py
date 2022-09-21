from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{inproceedings,
author = {Alfina, Ika and Mulia, Rio and Fanany, Mohamad Ivan and Ekanata, Yudo},
year = {2017},
month = {10},
pages = {},
title = {Hate Speech Detection in the Indonesian Language: A Dataset and Preliminary Study},
doi = {10.1109/ICACSIS.2017.8355039}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "id_hatespeech"

_DESCRIPTION = """\
The ID Hatespeech dataset is collection of 713 tweets related to a political event, the Jakarta Governor Election 2017
designed for hate speech detection NLP task. This dataset is crawled from Twitter, and then filtered
and annotated manually. The dataset labelled into two; HS if the tweet contains hate speech and Non_HS if otherwise
"""

_HOMEPAGE = "https://www.researchgate.net/publication/320131169_Hate_Speech_Detection_in_the_Indonesian_Language_A_Dataset_and_Preliminary_Study"
_LICENSE = "Unknown"
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/ialfina/id-hatespeech-detection/master/IDHSD_RIO_unbalanced_713_2017.txt",
}
_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdHatespeech(datasets.GeneratorBasedBuilder):
    """The ID Hatespeech dataset is collection of tweets related to a political event, the Jakarta Governor Election 2017
    designed for hate speech detection NLP task."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_hatespeech_source",
            version=SOURCE_VERSION,
            description="ID Hatespeech source schema",
            schema="source",
            subset_id="id_hatespeech",
        ),
        NusantaraConfig(
            name="id_hatespeech_nusantara_text",
            version=NUSANTARA_VERSION,
            description="ID Hatespeech Nusantara schema",
            schema="nusantara_text",
            subset_id="id_hatespeech",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_hatespeech_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"tweet": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["Non_HS", "HS"])
   

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
        df = pd.read_csv(filepath, sep="\t", encoding="ISO-8859-1").reset_index()
        df.columns = ["id", "label", "tweet"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "tweet": row.tweet,
                    "label": row.label,
                }
                yield row.id, ex

        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.tweet,
                    "label": row.label,
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
