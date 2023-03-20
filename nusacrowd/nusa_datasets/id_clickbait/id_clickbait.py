import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{WILLIAM2020106231,
title = "CLICK-ID: A novel dataset for Indonesian clickbait headlines",
journal = "Data in Brief",
volume = "32",
pages = "106231",
year = "2020",
issn = "2352-3409",
doi = "https://doi.org/10.1016/j.dib.2020.106231",
url = "http://www.sciencedirect.com/science/article/pii/S2352340920311252",
author = "Andika William and Yunita Sari",
keywords = "Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification",
abstract = "News analysis is a popular task in Natural Language Processing (NLP). In particular, the problem of clickbait in news analysis has gained attention in recent years [1, 2]. However, the majority of the tasks has been focused on English news, in which there is already a rich representative resource. For other languages, such as Indonesian, there is still a lack of resource for clickbait tasks. Therefore, we introduce the CLICK-ID dataset of Indonesian news headlines extracted from 12 Indonesian online news publishers. It is comprised of 15,000 annotated headlines with clickbait and non-clickbait labels. Using the CLICK-ID dataset, we then developed an Indonesian clickbait classification model achieving favourable performance. We believe that this corpus will be useful for replicable experiments in clickbait detection or other experiments in NLP areas."
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "id_clickbait"
_DESCRIPTION = """\
The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news
publishers; detikNews, Fimela, Kapanlagi, Kompas, Liputan6, Okezone, Posmetro-Medan, Republika, Sindonews, Tempo,
Tribunnews, and Wowkeren. This dataset is comprised of mainly two parts; (i) 46,119 raw article data, and (ii)
15,000 clickbait annotated sample headlines. Annotation was conducted with 3 annotator examining each headline.
Judgment were based only on the headline. The majority then is considered as the ground truth. In the annotated
sample, our annotation shows 6,290 clickbait and 8,710 non-clickbait.
"""
_HOMEPAGE = "https://www.sciencedirect.com/science/article/pii/S2352340920311252#!"
_LICENSE = "Creative Commons Attribution 4.0 International"
_URLS = {
    _DATASETNAME: "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/k42j7x2kpn-1.zip",
}
_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdClickbait(datasets.GeneratorBasedBuilder):
    """The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news, annotated with a label whether each is a clickbait"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_clickbait_source",
            version=SOURCE_VERSION,
            description="CLICK-ID source schema",
            schema="source",
            subset_id="id_clickbait",
        ),
        NusantaraConfig(
            name="id_clickbait_nusantara_text",
            version=NUSANTARA_VERSION,
            description="CLICK-ID Nusantara schema",
            schema="nusantara_text",
            subset_id="id_clickbait",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_clickbait_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"title": datasets.Value("string"), "label": datasets.Value("string"), "label_score": datasets.Value("int8")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["non-clickbait", "clickbait"])

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
        base_dir = Path(dl_manager.download_and_extract(urls)) / "annotated" / "combined" / "json"
        data_files = {"train": base_dir / "main.json"}

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
        # Dataset does not have row id, using python enumeration.
        data = json.load(open(filepath, "r"))

        if self.config.schema == "source":
            for row_index, row in enumerate(data):
                ex = {
                    "title": row["title"],
                    "label": row["label"],
                    "label_score": row["label_score"],
                }
                yield row_index, ex

        elif self.config.schema == "nusantara_text":
            for row_index, row in enumerate(data):
                ex = {
                    "id": str(row_index),
                    "text": row["title"],
                    "label": row["label"],
                }
                yield row_index, ex
