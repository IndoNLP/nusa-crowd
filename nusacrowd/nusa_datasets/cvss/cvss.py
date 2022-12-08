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
This template serves as a starting point for contributing a dataset to the Nusantara Dataset repo.
When modifying it for your dataset, look for TODO items that offer specific instructions.
Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html
To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.
TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.
[nusantara_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{jia2022cvss,
    title={{CVSS} Corpus and Massively Multilingual Speech-to-Speech Translation},
    author={Jia, Ye and Tadmor Ramanovich, Michelle and Wang, Quan and Zen, Heiga},
    booktitle={Proceedings of Language Resources and Evaluation Conference (LREC)},
    pages={6691--6703},
    year={2022}
}
"""
_DATASETNAME = "cvss"

_DESCRIPTION = """\
CVSS is a massively multilingual-to-English speech-to-speech translation corpus,
covering sentence-level parallel speech-to-speech translation pairs from 21
languages into English.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/cvss"
_LOCAL = False
_LANGUAGES = ["ind", "eng"]
_LANG_CODE = {"ind": "id", "eng": "en"}
_LICENSE = "CC-BY 4.0"

_URLS = {_DATASETNAME: "https://storage.googleapis.com/cvss"}
_SUPPORTED_TASKS = [
    Tasks.SPEECH_TO_SPEECH_TRANSLATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class CVSS(datasets.GeneratorBasedBuilder):
    """
    CVSS is a dataset on speech-to-speech translation. The data available are Indonesian audio files
    and their English transcriptions. There are two versions of the datasets, both derived from CoVoST 2,
    with each version providing unique values:
    CVSS-C: All the translation speeches are in a single canonical speaker's voice. Despite being synthetic, these
    speeches are of very high naturalness and cleanness, as well as having a consistent speaking style. These properties
    ease the modeling of the target speech and enable models to produce high quality translation speech suitable for
    user-facing applications.
    CVSS-T: The translation speeches are in voices transferred from the corresponding source speeches. Each translation
    pair has similar voices on the two sides despite being in different languages, making this dataset suitable for building models that preserve speakers' voices when translating speech into different languages.
    Together with the source speeches originated from Common Voice, they make two multilingual speech-to-speech
    translation datasets each with about 1,900 hours of speech.
    In addition to translation speech, CVSS also provides normalized translation text matching the pronunciation in the translation speech (e.g. on numbers, currencies, acronyms, etc.), which can be used for both model training as well as standardizing evaluation.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    cv_URL_TEMPLATE = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{lang}.tar.gz"

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="cvss_c_source",
            version=SOURCE_VERSION,
            description="CVSS source schema, all translation speeches are in a single canonical speaker's voice",
            schema="source",
            subset_id="cvss_c",
        ),
        NusantaraConfig(
            name="cvss_t_source",
            version=SOURCE_VERSION,
            description="CVSS source schema, translation speeches are in voices transferred " "from the corresponding source speeches",
            schema="source",
            subset_id="cvss_t",
        ),
        NusantaraConfig(
            name="cvss_c_nusantara_s2s",
            version=NUSANTARA_VERSION,
            description="CVSS Nusantara schema, all translation speeches are in a single canonical speaker's voice.",
            schema="nusantara_s2s",
            subset_id="cvss_c",
        ),
        NusantaraConfig(
            name="cvss_t_nusantara_s2s",
            version=NUSANTARA_VERSION,
            description="CVSS Nusantara schema, translation speeches are in voices transferred " "from the corresponding source speeches",
            schema="nusantara_s2s",
            subset_id="cvss_t",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cvss_c_source"

    def _get_download_urls(self, name="cvss_c", languages="id", version="1.0"):
        return f"{_URLS[_DATASETNAME]}/{name}_v{version}/{name}_{languages}_en_v{version}.tar.gz"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=24_000),
                    "text": datasets.Value("string"),
                    "original_file": datasets.Value("string"),
                    "original_audio": datasets.Audio(sampling_rate=48_000),
                    "original_text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_s2s":
            features = schemas.speech2speech_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = self._get_download_urls(self.config.name[:6])
        data_dir = dl_manager.download_and_extract(urls)

        cv_url = self.cv_URL_TEMPLATE.format(lang=_LANG_CODE[_LANGUAGES[0]])
        cv_dir = dl_manager.download_and_extract(cv_url)
        cv_dir = cv_dir + "/" + "/".join(["cv-corpus-6.1-2020-12-11", _LANG_CODE[_LANGUAGES[0]]])
        cv_tsv_dir = os.path.join(cv_dir, "validated.tsv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "cvss_path": data_dir,
                    "cv_path": cv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "cvss_path": data_dir,
                    "cv_path": cv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "cvss_path": data_dir,
                    "cv_path": cv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, cvss_path: Path, cv_path: Path, cv_tsv_path: Path, split: str) -> Tuple[int, Dict]:
        # open cv tsv
        cvss_tsv = self._load_df_from_tsv(os.path.join(cvss_path, f"{split}.tsv"))
        cv_tsv = self._load_df_from_tsv(cv_tsv_path)
        cvss_tsv.columns = ["path", "translation"]

        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=cvss_tsv[["path", "translation"]],
            how="inner",
            on="path",
        )
        for id, row in df.iterrows():
            translated_audio_path = os.path.join(cvss_path, split, f"{row['path']}.wav")
            translated_text = row["translation"]
            original_audio_path = os.path.join(cv_path, "clips", row["path"])
            original_text = row["sentence"]
            if self.config.schema == "source":
                yield id, {
                    "id": id,
                    "audio": translated_audio_path,
                    "file": translated_audio_path,
                    "text": translated_text,
                    "original_audio": original_audio_path,
                    "original_file": original_audio_path,
                    "original_text": original_text
                }
            elif self.config.schema == "nusantara_s2s":
                yield id, {
                    "id": id,
                    "path_1": original_audio_path,
                    "audio_1": original_audio_path,
                    "text_1": original_text,
                    "metadata_1": {
                        "name": "original_" + row["client_id"],
                        "speaker_age": None,
                        "speaker_gender": None,
                    },
                    "path_2": translated_audio_path,
                    "audio_2": translated_audio_path,
                    "text_2": translated_text,
                    "metadata_2": {
                        "name": 'translation',
                        "speaker_age": None,
                        "speaker_gender": None,
                    },
                }

    @staticmethod
    def _load_df_from_tsv(path):
        return pd.read_csv(
            path,
            sep="\t",
            header=0,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )