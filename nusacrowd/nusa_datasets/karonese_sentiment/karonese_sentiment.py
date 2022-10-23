from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

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

_LANGUAGES = ["btx"]
_LOCAL = False

_DATASETNAME = "karonese_sentiment"

_DESCRIPTION = """\
Karonese sentiment was crawled from Twitter between 1 January 2021 and 31 October 2021.
The first crawling process used several keywords related to the Karonese, such as
"deleng sinabung, Sinabung mountain", "mejuah-juah, greeting welcome", "Gundaling",
and so on. However, due to the insufficient number of tweets obtained using such
keywords, a second crawling process was done based on several hashtags, such as
#kalakkaro, # #antonyginting, and #lyodra.
"""

_HOMEPAGE = "http://section.iaesonline.com/index.php/IJEEI/article/view/3565"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/aliakbars/karonese/main/karonese_sentiment.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class KaroneseSentimentDataset(datasets.GeneratorBasedBuilder):
    """Karonese sentiment was crawled from Twitter between 1 January 2021 and 31 October 2021.
    The dataset consists of 397 negative, 342 neutral, and 260 positive tweets.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="karonese_sentiment_source",
            version=SOURCE_VERSION,
            description="Karonese Sentiment source schema",
            schema="source",
            subset_id="karonese_sentiment",
        ),
        NusantaraConfig(
            name="karonese_sentiment_nusantara_text",
            version=NUSANTARA_VERSION,
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
                    "no": datasets.Value("string"),
                    "tweet": datasets.Value("string"),
                    "label": datasets.Value("string"),
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
        df = pd.read_csv(filepath).drop("no", axis=1)
        df.columns = ["text", "label"]

        if self.config.schema == "source":
            for idx, row in df.iterrows():
                example = {
                    "no": str(idx+1),
                    "tweet": row.text,
                    "label": row.label,
                }
                yield idx, example
        elif self.config.schema == "nusantara_text":
            for idx, row in df.iterrows():
                example = {
                    "id": str(idx+1),
                    "text": row.text,
                    "label": row.label,
                }
                yield idx, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
