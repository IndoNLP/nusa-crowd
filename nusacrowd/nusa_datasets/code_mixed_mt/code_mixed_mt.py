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

"""
Code-mixed machine translation data between Indonesian language and Javanese
language using Lexicon based approach

Nowadays mixing one language with another language either in spoken or written
communication has become a common practice for bilingual speakers in daily
conversation as well as in social media. Lexicon based approach is one of the
approaches in extracting the sentiment analysis. This study is aimed to compare
two lexicon models which are SentiNetWord and VADER in extracting the polarity
of the code-mixed sentences in Indonesian language and Javanese language. 3,963
tweets were gathered from two accounts that provide code-mixed tweets.
Pre-processing such as removing duplicates, translating to English, filter
special characters, transform lower case and filter stop words were conducted
on the tweets. Positive and negative word score from lexicon model was then
calculated using simple mathematic formula in order to classify the polarity.
By comparing with the manual labelling, the result showed that SentiNetWord
perform better than VADER in negative sentiments. However, both of the lexicon
model did not perform well in neutral and positive sentiments. On overall
performance, VADER showed better performance than SentiNetWord. This study
showed that the reason for the misclassified was that most of Indonesian
language and Javanese language consist of words that were considered as
positive in both Lexicon model.

[nusantara_schema_name] = t2t
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{Tho_2021,
  doi = {10.1088/1742-6596/1869/1/012084},
  url = {https://doi.org/10.1088/1742-6596/1869/1/012084},
  year = 2021,
  month = {apr},
  publisher = {{IOP} Publishing},
  volume = {1869},
  number = {1},
  pages = {012084},
  author = {C Tho and Y Heryadi and L Lukas and A Wibowo},
  title = {Code-mixed sentiment analysis of Indonesian language and Javanese language using Lexicon based approach},
  journal = {Journal of Physics: Conference Series},
  abstract = {Nowadays mixing one language with another language either in
  spoken or written communication has become a common practice for bilingual
  speakers in daily conversation as well as in social media. Lexicon based
  approach is one of the approaches in extracting the sentiment analysis. This
  study is aimed to compare two lexicon models which are SentiNetWord and VADER
  in extracting the polarity of the code-mixed sentences in Indonesian language
  and Javanese language. 3,963 tweets were gathered from two accounts that
  provide code-mixed tweets. Pre-processing such as removing duplicates,
  translating to English, filter special characters, transform lower case and
  filter stop words were conducted on the tweets. Positive and negative word
  score from lexicon model was then calculated using simple mathematic formula
  in order to classify the polarity. By comparing with the manual labelling,
  the result showed that SentiNetWord perform better than VADER in negative
  sentiments. However, both of the lexicon model did not perform well in
  neutral and positive sentiments. On overall performance, VADER showed better
  performance than SentiNetWord. This study showed that the reason for the
  misclassified was that most of Indonesian language and Javanese language
  consist of words that were considered as positive in both Lexicon model.}
}
"""

_DATASETNAME = "code_mixed_mt"

_DESCRIPTION = """\
Machine Translation data between Javanese and Indonesian.
"""

_HOMEPAGE = "https://iopscience.iop.org/article/10.1088/1742-6596/1869/1/012084"

_LICENSE = "cc_by_3.0"

_URLS = {
    _DATASETNAME: "https://docs.google.com/spreadsheets/d/1mq2VyPEDfXl7K6p5TbRPsaefYwkuy7RQ/export?format=csv&gid=356398080",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class CodeMixedMt(datasets.GeneratorBasedBuilder):
    """Code-mixed Machine Translation data Javanese to Indonesian"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="code_mixed_mt_jav_ind_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="code_mixed_mt from Javanese to Indonesian",
            schema="source",
            subset_id="code_mixed_mt",
        ),
        NusantaraConfig(
            name="code_mixed_mt_jav_ind_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="code_mixed_mt from Javanese to Indonesian",
            schema="nusantara_t2t",
            subset_id="code_mixed_mt",
        )
    ]

    DEFAULT_CONFIG_NAME = "code_mixed_mt_jav_ind_senti"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"source": datasets.Value("string"), "target": datasets.Value("string")})
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION,)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        url = _URLS[_DATASETNAME]
        path = dl_manager.download_and_extract(url)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": path, "split": "train"}),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath, skiprows=1, names=["text_jav", "label", "text_ind"]).drop(columns=["label"])

        if self.config.schema == "source":
            i = 0
            for row in df.itertuples():
                ex = {"source": row.text_jav, "target": row.text_ind}
                yield i, ex
                i += 1

        elif self.config.schema == "nusantara_t2t":
            i = 0
            for row in df.itertuples():
                ex = {"id": str(i), "text_1": row.text_jav, "text_2": row.text_ind, "text_1_name": "jav", "text_2_name": "ind"}
                yield i, ex
                i += 1


if __name__ == "__main__":
    datasets.load_dataset(__file__)
