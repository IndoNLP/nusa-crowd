# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{putri2021abusive,
  title={Abusive language and hate speech detection for Javanese and Sundanese languages in tweets: Dataset and preliminary study},
  author={Putri, Shofianina Dwi Ananda and Ibrohim, Muhammad Okky and Budi, Indra},
  booktitle={2021 11th International Workshop on Computer Science and Engineering, WCSE 2021},
  pages={461--465},
  year={2021},
  organization={International Workshop on Computer Science and Engineering (WCSE)},
  abstract={Indonesia’s demography as an archipelago with lots of tribes and local languages added variances in their communication style. Every region in Indonesia has its own distinct culture, accents, and languages. The demographical condition can influence the characteristic of the language used in social media, such as Twitter. It can be found that Indonesian uses their own local language for communicating and expressing their mind in tweets. Nowadays, research about identifying hate speech and abusive language has become an attractive and developing topic. Moreover, the research related to Indonesian local languages still rarely encountered. This paper analyzes the use of machine learning approaches such as Naïve Bayes (NB), Support Vector Machine (SVM), and Random Forest Decision Tree (RFDT) in detecting hate speech and abusive language in Sundanese and Javanese as Indonesian local languages. The classifiers were used with the several term weightings features, such as word n-grams and char n-grams. The experiments are evaluated using the F-measure. It achieves over 60 % for both local languages.}
}
"""
_DATASETNAME = "local_id_abusive"

_DESCRIPTION = """\
This dataset is for abusive and hate speech detection, using Twitter text containing Javanese and Sundanese words.

(from the publication source)
The Indonesian local language dataset collection was conducted using Twitter search API to collect the tweets and then
implemented using Tweepy Library. The tweets were collected using queries from the list of abusive words in Indonesian
tweets. The abusive words were translated into local Indonesian languages, which are Javanese and Sundanese. The
translated words are then used as queries to collect tweets containing Indonesian and local languages. The translation
process involved native speakers for each local language. The crawling process has collected a total of more than 5000
tweets. Then, the crawled data were filtered to get tweets that contain local’s vocabulary and/or sentences in Javanese
and Sundanese. Next, after the filtering process, the data will be labeled whether the tweets are labeled as hate speech
and abusive language or not.
"""

_HOMEPAGE = "https://github.com/Shofianina/local-indonesian-abusive-hate-speech-dataset"

_LICENSE = "Unknown"

_LANGUAGES = ["jav", "sun"]
_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "jav": "https://raw.githubusercontent.com/Shofianina/local-indonesian-abusive-hate-speech-dataset/main/Javanese.csv",
        "sun": "https://raw.githubusercontent.com/Shofianina/local-indonesian-abusive-hate-speech-dataset/main/Sundanese.csv",
    }
}

_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class LocalIDAbusive(datasets.GeneratorBasedBuilder):
    """Local ID Abusive is a dataset for abusive and hate speech detection, using Twitter text containing Javanese and
    Sundanese words."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="local_id_abusive_jav_source",
            version=SOURCE_VERSION,
            description="local_id_abusive source schema Javanese",
            schema="source",
            subset_id="local_id_abusive_jav",
        ),
        NusantaraConfig(
            name="local_id_abusive_sun_source",
            version=SOURCE_VERSION,
            description="local_id_abusive source schema Sundanese",
            schema="source",
            subset_id="local_id_abusive_sun",
        ),
        NusantaraConfig(
            name="local_id_abusive_jav_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="local_id_abusive Nusantara schema Javanese",
            schema="nusantara_text_multi",
            subset_id="local_id_abusive_jav",
        ),
        NusantaraConfig(
            name="local_id_abusive_sun_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="local_id_abusive Nusantara schema Sundanese",
            schema="nusantara_text_multi",
            subset_id="local_id_abusive_sun",
        ),
    ]

    DEFAULT_CONFIG_NAME = "local_id_abusive_jav_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "isi_tweet": datasets.Value("string"),
                    "uk": datasets.Value("bool"),
                    "hs": datasets.Value("bool"),
                }
            )
        elif self.config.schema == "nusantara_text_multi":
            features = schemas.text_multi_features([0, 1])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = self.config.name.split("_")[3]
        urls = _URLS[_DATASETNAME][lang]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath, sep=",", encoding="ISO-8859-1").reset_index()
        for i, row in enumerate(df.itertuples()):
            if self.config.schema == "source":
                example = {"isi_tweet": row.isi_tweet, "uk": row.uk, "hs": row.hs}
                yield i, example
            elif self.config.schema == "nusantara_text_multi":
                example = {
                    "id": str(i),
                    "text": row.isi_tweet,
                    "labels": [row.uk, row.hs],
                }
                yield i, example
