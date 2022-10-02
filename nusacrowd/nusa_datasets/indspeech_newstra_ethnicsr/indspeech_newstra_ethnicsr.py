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

import os
from itertools import chain
from pathlib import Path
from random import sample
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{sakti-cocosda-2013,
    title = "Towards Language Preservation: Design and Collection of Graphemically Balanced and Parallel Speech Corpora of {I}ndonesian Ethnic Languages",
    author = "Sakti, Sakriani and Nakamura, Satoshi",
    booktitle = "Proc. Oriental COCOSDA",
    year = "2013",
    address = "Gurgaon, India"
}

@inproceedings{sakti-sltu-2014,
  title = "Recent progress in developing grapheme-based speech recognition for {I}ndonesian ethnic languages: {J}avanese, {S}undanese, {B}alinese and {B}ataks",
  author = "Sakti, Sakriani and Nakamura, Satoshi",
  booktitle = "Proc. 4th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2014)",
  year = "2014",
  pages = "46--52",
  address = "St. Petersburg, Russia"
}

@inproceedings{novitasari-sltu-2020,
  title = "Cross-Lingual Machine Speech Chain for {J}avanese, {S}undanese, {B}alinese, and {B}ataks Speech Recognition and Synthesis",
  author = "Novitasari, Sashi and Tjandra, Andros and Sakti, Sakriani and Nakamura, Satoshi",
  booktitle = "Proc. Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)",
  year = "2020",
  pages = "131--138",
  address = "Marseille, France"
}
"""

_DATASETNAME = "indspeech_newstra_ethnicsr"
_DESCRIPTION = """\
INDspeech_NEWSTRA_EthnicSR is a collection of graphemically balanced and parallel speech corpora of four major Indonesian ethnic languages: Javanese, Sundanese, Balinese, and Bataks. It was developed in 2013 by the Nara Institute of Science and Technology (NAIST, Japan) [Sakti et al., 2013]. The data has been used to develop Indonesian ethnic speech recognition in supervised learning [Sakti et al., 2014] and semi-supervised learning [Novitasari et al., 2020] based on Machine Speech Chain framework [Tjandra et al., 2020].
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_newstra_ethnicsr"
_LANGUAGES = ["sun", "jav", "btk", "ban"]
_LOCAL = False
_LICENSE = "CC-BY-NC-SA 4.0"

_lst_TYPE = ["traEth", "traInd"]
_lst_LANG = {"Bli": "BALI", "Btk": "BATAK", "Jaw": "JAWA", "Snd": "SUNDA"}
_lst_STD_LANG = {"ban": "Bli", "btk": "Btk", "jav": "Jaw", "sun": "Snd"}
_lst_HEAD_1_TRAIN = "https://raw.githubusercontent.com/s-sakti/data_indsp_newstra_ethnicsr/main/lst/dataset1_train_news_"
_lst_HEAD_1_TEST = ["https://raw.githubusercontent.com/s-sakti/data_indsp_newstra_ethnicsr/main/lst/dataset1_test_" + ltype + "_" for ltype in _lst_TYPE]
_lst_HEAD_2 = "https://raw.githubusercontent.com/s-sakti/data_indsp_newstra_ethnicsr/main/lst/dataset2_"
_sp_TEMPLATE = "https://raw.githubusercontent.com/s-sakti/data_indsp_newstra_ethnicsr/main/speech/16kHz/"  # {lang}/Ind0{index}_{gender}_{lang_code}.zip"
_txt_TEMPLATE = "https://github.com/s-sakti/data_indsp_newstra_ethnicsr/raw/main/text/utts_transcript/"  # {lang}/Ind0{index}_{gender}_{lang_code}.zip"

_URLS = {
    "dataset1_train": {llang.lower(): [_lst_HEAD_1_TRAIN + llang + ".lst"] for llang in _lst_LANG},
    "dataset1_test": {llang.lower(): [head1test + llang + ".lst" for head1test in _lst_HEAD_1_TEST] for llang in _lst_LANG},
    "dataset2_train": {llang.lower(): [_lst_HEAD_2 + "train_news_" + llang + ".lst"] for llang in _lst_LANG},
    "dataset2_test": {llang.lower(): [_lst_HEAD_2 + "test_news_" + llang + ".lst"] for llang in _lst_LANG},
    "speech": {llang.lower(): [_sp_TEMPLATE + _lst_LANG[llang] + "/Ind" + str(idx).zfill(3) + "_" + ("M" if idx % 2 == 0 else "F") + "_" + llang + ".zip" for idx in range(1, 11)] for llang in _lst_LANG},
    "transcript": {llang.lower(): [_txt_TEMPLATE + _lst_LANG[llang] + "/Ind" + str(idx).zfill(3) + "_" + ("M" if idx % 2 == 0 else "F") + "_" + llang + ".zip" for idx in range(1, 11)] for llang in _lst_LANG},
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(lang, schema, version, overlap):
    if lang == "":
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source" and schema != "nusantara_sptext":
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="indspeech_newstra_ethnicsr_{overlap}_{lang}_{schema}".format(lang=lang, schema=schema, overlap=overlap),
        version=datasets.Version(version),
        description="indspeech_newstra_ethnicsr {schema} schema for {lang} language with {overlap}ping dataset".format(lang=_lst_LANG[_lst_STD_LANG[lang]], schema=schema, overlap=overlap),
        schema=schema,
        subset_id="indspeech_newstra_ethnicsr_{overlap}".format(overlap=overlap),
    )

class INDspeechNEWSTRAEthnicSR(datasets.GeneratorBasedBuilder):
    """
    The dataset contains 2 sub-datasets
    Dataset 1 has 2250/1000 train/test samples per language
    Dataset 2 has another 1600/50 train/test per language
    The 'overlap' keyword in the dataset-name combines both sub-datasets, while 'nooverlap' will only use dataset 1
    """
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [nusantara_config_constructor(lang, "source", _SOURCE_VERSION, overlap) for lang in _lst_STD_LANG for overlap in ["overlap","nooverlap"]] +\
                      [nusantara_config_constructor(lang, "nusantara_sptext", _NUSANTARA_VERSION, overlap) for lang in _lst_STD_LANG for overlap in ["overlap","nooverlap"]]

    DEFAULT_CONFIG_NAME = "indspeech_newstra_ethnicsr_jav_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = _lst_STD_LANG[self.config.name.split("_")[4]].lower()
        ds1_train_urls = _URLS["dataset1_train"][lang]
        ds1_test_urls = _URLS["dataset1_test"][lang]
        ds2_train_urls = _URLS["dataset2_train"][lang]
        ds2_test_urls = _URLS["dataset2_test"][lang]
        sp_urls = _URLS["speech"][lang]
        txt_urls = _URLS["transcript"][lang]

        ds1_train_dir = [Path(dl_manager.download_and_extract(ds1_train_url)) for ds1_train_url in ds1_train_urls]
        ds1_test_dir = [Path(dl_manager.download_and_extract(ds1_test_url)) for ds1_test_url in ds1_test_urls]
        ds2_train_dir = [Path(dl_manager.download_and_extract(ds2_train_url)) for ds2_train_url in ds2_train_urls]
        ds2_test_dir = [Path(dl_manager.download_and_extract(ds2_test_url)) for ds2_test_url in ds2_test_urls]
        sp_dir = {str(Path(sp_url).name)[:-4]: os.path.join(Path(dl_manager.download_and_extract(sp_url)), str(Path(sp_url).name))[:-4] for sp_url in sp_urls}
        txt_dir = {str(Path(txt_url).name)[:-4]: os.path.join(Path(dl_manager.download_and_extract(txt_url)), str(Path(txt_url).name))[:-4] for txt_url in txt_urls}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "dataset1": ds1_train_dir,
                        "dataset2": ds2_train_dir,
                        "speech": sp_dir,
                        "transcript": txt_dir,
                    },
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "dataset1": ds1_test_dir,
                        "dataset2": ds2_test_dir,
                        "speech": sp_dir,
                        "transcript": txt_dir,
                    },
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        sample_list=[]
        if self.config.name.split("_")[3] == "nooverlap":
            sample_list = [open(samples).read().splitlines() for samples in filepath["dataset1"]]  # +filepath["dataset2"]]
            sample_list = list(chain(*sample_list))
        elif self.config.name.split("_")[3] == "overlap":
            sample_list = [open(samples).read().splitlines() for samples in filepath["dataset1"]+filepath["dataset2"]]
            sample_list = list(chain(*sample_list))

        for id, row in enumerate(sample_list):
            if self.config.schema == "source":
                ex = {
                    "id": str(id),
                    "speaker_id": str(Path(row).parent).split("/")[1],
                    "path": os.path.join(filepath["speech"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".wav"),
                    "audio": os.path.join(filepath["speech"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".wav"),
                    "text": open(os.path.join(filepath["transcript"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".txt"), "r").read().splitlines()[0],
                    "gender": str(Path(row).parent).split("/")[1].split("_")[1],
                }
                yield id, ex

            elif self.config.schema == "nusantara_sptext":
                ex = {
                    "id": str(id),
                    "speaker_id": str(Path(row).parent).split("/")[1],
                    "path": os.path.join(filepath["speech"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".wav"),
                    "audio": os.path.join(filepath["speech"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".wav"),
                    "text": open(os.path.join(filepath["transcript"][str(Path(row).parent).split("/")[1]], str(Path(row).name) + ".txt"), "r").read().splitlines()[0],
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": str(Path(row).parent).split("/")[1].split("_")[1],
                    },
                }
                yield id, ex
