from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.nusa_datasets.karonese_sentiment.utils.karonese_sentiment_utils import map_label
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks
from nusantara.utils import schemas

_CITATION = """\
@article{karo2022sentiment,
  title={Sentiment Analysis in Karonese Tweet using Machine Learning},
  author={Karo, Ichwanul Muslim Karo and Fudzee, Mohd Farhan Md and Kasim, Shahreen and Ramli, Azizul Azhar},
  journal={Indonesian Journal of Electrical Engineering and Informatics (IJEEI)},
  volume={10},
  number={1},
  pages={219--231},
  year={2022}
}
"""

_DATASETNAME = "karonese_sentiment"

_DESCRIPTION = """\
Karonese sentiment was crawled from Twitter between 1 January 2021 and 31 October 2021.he first crawling process used several keywords related to the Karonese, such as "deleng sinabung, Sinabung mountain", "mejuah-juah, greeting welcome", "Gundaling", and so on. However, due to the insufficient number of tweets obtained using such keywords, a second crawling process was done based on several hashtags, such as #kalakkaro, # #antonyginting, and #lyodra.
"""

_HOMEPAGE = "http://section.iaesonline.com/index.php/IJEEI/article/view/3565"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://github.com/imkarokaro123/karonese/raw/main/karonese%20dataset.xlsx",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class KaroneseSentimentDataset(datasets.GeneratorBasedBuilder):
    """Customer Review (Natasha Skincare) is a customers emotion dataset, with amounted to 19,253 samples with the division for each class is 804 joy, 43 surprise, 154 anger, 61 fear, 287 sad, 167 disgust, and 17736 no-emotions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="karonese_sentiment_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Karonese Sentiment source schema",
            schema="source",
            subset_id="karonese_sentiment",
        ),
        NusantaraConfig(
            name="karonese_sentiment_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Karonese Sentiment Nusantara schema",
            schema="nusantara_text",
            subset_id="karonese_sentiment",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sentiment_nathasa_review_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                    "No": datasets.Value("string"),
                    "Tweets": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "Sumber": datasets.Value("string"),
               }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["negative", "neutral", "positive"])

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
        data_dir = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_excel(filepath).drop("No", axis=1)
        df.columns = ["text", "label", "source"]

        if self.config.schema == "source":
            for idx, row in df.iterrows():
                example = {
                    "no": str(idx),
                    "tweet": row.text,
                    "label": row.label,
                    "source": row.source,
                }
                yield idx, example
        elif self.config.schema == "nusantara_text":
            for idx, row in df.iterrows():
                example = {
                    "id": str(idx),
                    "text": row.text,
                    "label": map_label(row.label),
                }
                yield idx, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
