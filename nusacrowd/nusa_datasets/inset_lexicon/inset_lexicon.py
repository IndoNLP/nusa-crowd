from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{inproceedings,
author = {Koto, Fajri and Rahmaningtyas, Gemala},
year = {2017},
month = {12},
pages = {},
title = {InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs},
doi = {10.1109/IALP.2017.8300625}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "inset_lexicon"

_DESCRIPTION = """\
InSet, an Indonesian sentiment lexicon built to identify written opinion and categorize it into positive or negative opinion,
which could be utilized to analyze public sentiment towards particular topic, event, or product. Composed using collection
of words from Indonesian tweet, InSet was constructed by manually weighting each words and enhanced by adding stemming and synonym set
"""

_HOMEPAGE = "https://www.researchgate.net/publication/321757985_InSet_Lexicon_Evaluation_of_a_Word_List_for_Indonesian_Sentiment_Analysis_in_Microblogs"
_LICENSE = "Unknown"
_URLS = {_DATASETNAME: "https://github.com/fajri91/InSet/archive/refs/heads/master.zip"}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class InsetLexicon(datasets.GeneratorBasedBuilder):
    """InSet, an Indonesian sentiment lexicon built to identify written opinion and categorize it into positive or negative opinion"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="inset_lexicon_source",
            version=SOURCE_VERSION,
            description="Inset Lexicon source schema",
            schema="source",
            subset_id="inset_lexicon",
        ),
        NusantaraConfig(
            name="inset_lexicon_nusantara_text",
            version=NUSANTARA_VERSION,
            description="Inset Lexicon Nusantara schema",
            schema="nusantara_text",
            subset_id="inset_lexicon",
        ),
    ]

    DEFAULT_CONFIG_NAME = "inset_lexicon_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"word": datasets.Value("string"), "weight": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            labels = list(range(-5, 6, 1))
            labels = [str(label) for label in labels]
            features = schemas.text_features(labels)

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
        base_dir = Path(dl_manager.download_and_extract(urls)) / "InSet-master"
        positive_df = pd.read_csv(base_dir / "positive.tsv", sep="\t")
        negative_df = pd.read_csv(base_dir / "negative.tsv", sep="\t")
        merged_df = pd.concat([positive_df, negative_df]).reset_index(drop=True)
        merged_data_dir = base_dir / "dataset.tsv"
        merged_df.to_csv(merged_data_dir, sep="\t")

        data_files = {"train": merged_data_dir}

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
        df = pd.read_csv(filepath, sep="\t", encoding="ISO-8859-1")
        df.columns = ["id", "word", "weight"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "word": row.word,
                    "weight": str(int(row.weight)),
                }
                yield row.id, ex

        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.word,
                    "label": str(int(row.weight)),
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
