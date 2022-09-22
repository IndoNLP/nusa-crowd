from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@misc{
   research, 
   title={Jakartaresearch/google-play-review · datasets at hugging face}, 
   url={https://huggingface.co/datasets/jakartaresearch/google-play-review},
   author={Research, Jakarta AI}
} 
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "id_google_play_review"
_DESCRIPTION = """\
Indonesian Google Play Review, dataset scrapped from e-commerce app on Google Play for sentiment analysis.
Total number of data: 10041 (train: 7028, validation: 3012). Provided by Jakarta AI Research.
"""

_HOMEPAGE = "https://github.com/jakartaresearch/hf-datasets/tree/main/google-play-review/google-play-review"
_LICENSE = "CC-BY 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/google-play-review/google-play-review/train.csv",
        "valid": "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/google-play-review/google-play-review/validation.csv",
    }
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IDGooglePlayReview(datasets.GeneratorBasedBuilder):
    """
    Indonesian Google Play Review is a dataset containing reviews from Google Play Indonesia, used for sentiment
    analysis.
    The language content is mainly Indonesian, however beware of context-switching (some sentences are partly or
    entirely in English).
    The available labels:
        label: ['pos', 'neg'] for source and nusantara_text scheme
        stars: [1, 2, 3, 4, 5] for source
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_google_play_review_source",
            version=SOURCE_VERSION,
            description="id_google_play_review source schema",
            schema="source",
            subset_id="id_google_play_review",
        ),
        NusantaraConfig(
            name="id_google_play_review_posneg_source",
            version=SOURCE_VERSION,
            description="id_google_play_review source schema",
            schema="source",
            subset_id="id_google_play_review_posneg",
        ),
        NusantaraConfig(
            name="id_google_play_review_nusantara_text",
            version=NUSANTARA_VERSION,
            description="id_google_play_review Nusantara schema, 1-5 stars rating only (for pos/neg labels, please use the subset_id \"id_google_play_review_posneg\")",
            schema="nusantara_text",
            subset_id="id_google_play_review",
        ),
        NusantaraConfig(
            name="id_google_play_review_posneg_nusantara_text",
            version=NUSANTARA_VERSION,
            description="id_google_play_review Nusantara schema, pos/neg label only",
            schema="nusantara_text",
            subset_id="id_google_play_review_posneg",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_google_play_review_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset
        # as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.Value("string"),
                "stars": datasets.Value("int8")
            })
        elif self.config.schema == "nusantara_text":
            if self.config.subset_id == "id_google_play_review_posneg":
                features = schemas.text_features(["pos", "neg"])
            elif self.config.subset_id == "id_google_play_review":
                features = schemas.text_features(["1", "2", "3", "4", "5"])
            else:
                raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        train_data_path = Path(dl_manager.download(urls["train"]))
        valid_data_path = Path(dl_manager.download(urls["valid"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_data_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": valid_data_path, "split": "valid"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, sep=",").reset_index()
        for row in df.itertuples(index=False):
            if self.config.schema == "source":
                example = {"text": row.text, "label": row.label, "stars": row.stars}
                yield row.index, example
            elif self.config.schema == "nusantara_text":
                if self.config.subset_id == "id_google_play_review_posneg":
                    example = {"id": row.index, "text": row.text, "label": row.label}
                elif self.config.subset_id == "id_google_play_review":
                    example = {"id": row.index, "text": row.text, "label": str(row.stars)}
                else:
                    raise ValueError(f"Invalid config: {self.config.name}")
                yield row.index, example
