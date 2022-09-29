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
Code-mixed sentiment analysis of Indonesian language and Javanese language
using Lexicon based approach

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

[nusantara_schema_name] = text
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

_DATASETNAME = "code_mixed_senti"

_DESCRIPTION = """\
Sentiment analysis for Javanese and Indonesian.
"""

_HOMEPAGE = "https://iopscience.iop.org/article/10.1088/1742-6596/1869/1/012084"

_LICENSE = "cc_by_3.0"

_URLS = {
    _DATASETNAME: "https://docs.google.com/spreadsheets/d/1mq2VyPEDfXl7K6p5TbRPsaefYwkuy7RQ/export?format=csv&gid=356398080",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

LANGUAGES_MAP = {
    "ind": "indonesian",
    "jav": "javanese",
}

LANGUAGES_COLUMNS = {
    "ind": ("text_ind", "text_jav"),
    "jav": ("text_jav", "text_ind"),
}


def nusantara_config_constructor(lang, schema, version):
    """Construct NusantaraConfig with code_mixed_senti_{lang}_{schema} as the name format"""
    if schema != "source" and schema != "nusantara_text":
        raise ValueError(f"Invalid schema: {schema}")
    if lang not in LANGUAGES_MAP:
        raise ValueError(f"Invalid language: {lang}")

    return NusantaraConfig(
        name="code_mixed_senti_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="code_mixed_senti with {schema} schema for {lang} language".format(lang=lang, schema=schema),
        schema=schema,
        subset_id="code_mixed_senti",
    )


class CodeMixedSenti(datasets.GeneratorBasedBuilder):
    """Code-mixed sentiment analysis for Indonesian and Javanese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in LANGUAGES_MAP] + [nusantara_config_constructor(lang, "nusantara_text", _NUSANTARA_VERSION) for lang in LANGUAGES_MAP]

    DEFAULT_CONFIG_NAME = "code_mixed_senti_jav_senti"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"text": datasets.Value("string"), "label": datasets.Value("int32")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["-1", "0", "1"])

        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION,)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        url = _URLS[_DATASETNAME]
        path = dl_manager.download_and_extract(url)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": path, "split": "train"}),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        prefix_length = len(_DATASETNAME)
        start = prefix_length + 1
        end = prefix_length + 1 + 3
        language = self.config.name[start:end]
        keep_column, drop_column = LANGUAGES_COLUMNS[language]
        df = pd.read_csv(filepath, skiprows=1, names=["text_jav", "label", "text_ind"]).drop(columns=[drop_column]).rename(columns={keep_column: "text"})

        if self.config.schema == "source":
            i = 0
            for row in df.itertuples():
                ex = {"text": row.text, "label": row.label}
                yield i, ex
                i += 1

        elif self.config.schema == "nusantara_text":
            i = 0
            for row in df.itertuples():
                ex = {"id": str(i), "text": row.text, "label": str(row.label)}
                yield i, ex
                i += 1


if __name__ == "__main__":
    datasets.load_dataset(__file__)
