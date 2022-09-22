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

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_LANGUAGES = ["ind", "eng"]
_CITATION = """\

@article{wang2020covost,
  title={Covost 2 and massively multilingual speech-to-text translation},
  author={Wang, Changhan and Wu, Anne and Pino, Juan},
  journal={arXiv preprint arXiv:2007.10310},
  year={2020}
}

@inproceedings{wang21s_interspeech,
  author={Wang, Changhan and Wu, Anne and Pino, Juan},
  title={{CoVoST 2 and Massively Multilingual Speech Translation}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2247--2251},
  url={https://www.isca-speech.org/archive/interspeech_2021/wang21s_interspeech}
  doi={10.21437/Interspeech.2021-2027}
}
"""

_DATASETNAME = "covost2"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_DESCRIPTION = """\
CoVoST2 is a large-scale multilingual speech translation corpus covering translations from 21 languages to English
and from English into 15 languages. The dataset is created using Mozilla's open-source Common Voice database of
crowdsourced voice recordings. There are 2,900 hours of speech represented in the corpus.
"""

_HOMEPAGE = "https://huggingface.co/datasets/covost2"

_LOCAL = False
_LICENSE = "CC BY-NC 4.0"

COMMONVOICE_URL_TEMPLATE = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{lang}.tar.gz"
LANG_CODE = {"eng": "en", "ind": "id"}
LANG_COMBINATION_CODE = [("ind", "eng"), ("eng", "ind")]
_URLS = {_DATASETNAME: {"ind": COMMONVOICE_URL_TEMPLATE.format(lang=LANG_CODE["ind"]), "eng": COMMONVOICE_URL_TEMPLATE.format(lang=LANG_CODE["eng"])}}

_SUPPORTED_TASKS = [Tasks.SPEECH_TO_TEXT_TRANSLATION, Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(src_lang, tgt_lang, schema, version):
    if src_lang == "" or tgt_lang == "":
        raise ValueError(f"Invalid src_lang {src_lang} or tgt_lang {tgt_lang}")

    if schema not in ["source", "nusantara_sptext", "nusantara_t2t"]:
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="covost2_{src}_{tgt}_{schema}".format(src=src_lang, tgt=tgt_lang, schema=schema),
        version=datasets.Version(version),
        description="covost2 source schema for {schema} from {src} to {tgt}".format(schema=schema, src=src_lang, tgt=tgt_lang),
        schema=schema,
        subset_id="co_vo_st2_{src}_{tgt}".format(src=src_lang, tgt=tgt_lang),
    )


class Covost2(datasets.GeneratorBasedBuilder):
    """CoVoST2 dataset is a dataset mainly for speech to text translation task. The data was taken from Mozilla Common
    Voices dataset. In the implementation of the source schema, the audio and transcriptions of the source language,
    as well as the translated transcriptions are provided. In the implementation of the nusantara schema, only the audio of the source language and transcriptions of the
    target language are provided. The source and target languages available are eng->ind and ind -> eng respectively.
    In addition to the speech to text translation, this dataset (text only) can be used as a machine translation for
    eng->ind and ind->eng.

    Side note: the amount of data takes about 40GB for the English source data and 1GB for the Indonesian source data.
    """

    COVOST_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/covost/covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = (
        [nusantara_config_constructor(src, tgt, "source", _SOURCE_VERSION) for (src, tgt) in LANG_COMBINATION_CODE]
        + [nusantara_config_constructor(src, tgt, "nusantara_sptext", _NUSANTARA_VERSION) for (src, tgt) in LANG_COMBINATION_CODE]
        + [nusantara_config_constructor(src, tgt, "nusantara_t2t", _NUSANTARA_VERSION) for (src, tgt) in LANG_COMBINATION_CODE]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_eng_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {"client_id": datasets.Value("string"), "file": datasets.Value("string"), "audio": datasets.Audio(sampling_rate=16_000), "sentence": datasets.Value("string"), "translation": datasets.Value("string"), "id": datasets.Value("string")}
            )
        elif self.config.schema == "nusantara_sptext":
            features = schemas.speech_text_features
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentences")] if (self.config.schema == "nusantara_sptext" or self.config.schema == "source") else None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        name_split = self.config.name.split("_")
        src_lang, tgt_lang = name_split[1], name_split[2]

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls[src_lang])

        src_lang = LANG_CODE[src_lang]
        tgt_lang = LANG_CODE[tgt_lang]

        data_dir = data_dir + "/" + "/".join(["cv-corpus-6.1-2020-12-11", src_lang])

        covost_tsv_path = self.COVOST_URL_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang)
        extracted_dir = dl_manager.download_and_extract(covost_tsv_path)

        covost_tsv_filename = "covost_v2.{src_lang}_{tgt_lang}.tsv"
        covost_tsv_dir = os.path.join(extracted_dir, covost_tsv_filename.format(src_lang=src_lang, tgt_lang=tgt_lang))
        cv_tsv_dir = os.path.join(data_dir, "validated.tsv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "covost_tsv_path": covost_tsv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir,
                    "covost_tsv_path": covost_tsv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir,
                    "covost_tsv_path": covost_tsv_dir,
                    "cv_tsv_path": cv_tsv_dir,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, covost_tsv_path: Path, cv_tsv_path: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        name_split = self.config.name.split("_")
        src_lang, tgt_lang = name_split[1], name_split[2]

        covost_tsv = self._load_df_from_tsv(covost_tsv_path)
        cv_tsv = self._load_df_from_tsv(cv_tsv_path)

        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == "train") | (df["split"] == "train_covost")]
        else:
            df = df[df["split"] == split]

        for id, row in df.iterrows():
            if self.config.schema == "source":
                yield id, {
                    "id": row["path"].replace(".mp3", ""),
                    "client_id": row["client_id"],
                    "sentence": row["sentence"],
                    "translation": row["translation"],
                    "file": os.path.join(filepath, "clips", row["path"]),
                    "audio": os.path.join(filepath, "clips", row["path"]),
                }
            elif self.config.schema == "nusantara_sptext":
                yield id, {
                    "id": row["path"].replace(".mp3", ""),
                    "speaker_id": row["client_id"],
                    "text": row["translation"],
                    "path": os.path.join(filepath, "clips", row["path"]),
                    "audio": os.path.join(filepath, "clips", row["path"]),
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": None,
                    },
                }
            elif self.config.schema == "nusantara_t2t":
                yield id, {"id": row["path"].replace(".mp3", ""), "text_1": row["sentence"], "text_2": row["translation"], "text_1_name": src_lang, "text_2_name": tgt_lang}
            else:
                raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

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
