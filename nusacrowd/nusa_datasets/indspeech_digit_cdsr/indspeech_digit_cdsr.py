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
from cgitb import text
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{sakti-icslp-2004,
    title = "Indonesian Speech Recognition for Hearing and Speaking Impaired People",
    author = "Sakti, Sakriani and Hutagaol, Paulus and Arman, Arry Akhmad and Nakamura, Satoshi",
    booktitle = "Proc. International Conference on Spoken Language Processing (INTERSPEECH - ICSLP)",
    year = "2004",
    pages = "1037--1040"
    address = "Jeju Island, Korea"
}
"""
_DATASETNAME = "indspeech_digit_cdsr"
_LANGUAGES = ["ind"]
_DESCRIPTION = """\
INDspeech_DIGIT_CDSR is the first Indonesian speech dataset for connected digit speech recognition (CDSR). The data was developed by TELKOMRisTI (R&D Division, PT Telekomunikasi Indonesia) in collaboration with Advanced Telecommunication Research Institute International (ATR) Japan and Bandung Institute of Technology (ITB) under the Asia-Pacific Telecommunity (APT) project in 2004 [Sakti et al., 2004]. Although it was originally developed for a telecommunication system for hearing and speaking impaired people, it can be used for other applications, i.e., automatic call centers that recognize telephone numbers.
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_digit_cdsr"
_LOCAL = False
_LICENSE = "CC-BY-NC-SA-4.0"

_TMP_URL = {
    "lst": "https://raw.githubusercontent.com/s-sakti/data_indsp_digit_cdsr/main/lst/",
    "text": "https://github.com/s-sakti/data_indsp_digit_cdsr/raw/main/text/",
    "speech": "https://github.com/s-sakti/data_indsp_digit_cdsr/raw/main/speech/",
}

_URLS = {
    "lst": {
        "train_spk": _TMP_URL["lst"] + "train_spk.lst",
        "train_fname": _TMP_URL["lst"] + "train_fname.lst",
        "test_spk": [_TMP_URL["lst"] + "test" + str(i) + "_spk.lst" for i in range(1, 5)],
        "test_fname": [_TMP_URL["lst"] + "test" + str(i) + "_fname.lst" for i in range(1, 5)],
    },
    "train": {"speech": _TMP_URL["speech"] + "train/", "text": _TMP_URL["text"] + "train/"},
    "test": {"speech": _TMP_URL["speech"] + "test", "text": _TMP_URL["text"] + "test"},
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class INDspeechDIGITCDSR(datasets.GeneratorBasedBuilder):
    """Indonesian speech dataset for connected digit speech recognition"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indspeech_digit_cdsr_source",
            version=SOURCE_VERSION,
            description="indspeech_digit_cdsr source schema",
            schema="source",
            subset_id="indspeech_digit_cdsr",
        ),
        NusantaraConfig(
            name="indspeech_digit_cdsr_nusantara_sptext",
            version=NUSANTARA_VERSION,
            description="indspeech_digit_cdsr Nusantara schema",
            schema="nusantara_sptext",
            subset_id="indspeech_digit_cdsr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indspeech_digit_cdsr_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
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
        """Returns SplitGenerators."""
        lst_train_spk = Path(dl_manager.download_and_extract(_URLS["lst"]["train_spk"]))
        lst_train_fname = Path(dl_manager.download_and_extract(_URLS["lst"]["train_fname"]))
        lst_test_spk = [Path(dl_manager.download_and_extract(url)) for url in _URLS["lst"]["test_spk"]]
        lst_test_fname = [Path(dl_manager.download_and_extract(url)) for url in _URLS["lst"]["test_fname"]]

        fnames = {"test": []}
        speech = {"test": {}}
        text = {"test": {}}

        with open(lst_train_spk, "r") as f:
            speakers = [spk.replace("\n", "") for spk in f.readlines()]
            tmp_speech = [Path(dl_manager.download_and_extract(_URLS["train"]["speech"] + spk + ".zip")) for spk in speakers]
            tmp_text = [Path(dl_manager.download_and_extract(_URLS["train"]["text"] + spk + ".zip")) for spk in speakers]
            speech["train"] = {speech[:-4]: os.path.join(spk, speech) for spk in tmp_speech for speech in os.listdir(spk)}
            text["train"] = {text[:-4]: os.path.join(spk, text) for spk in tmp_text for text in os.listdir(spk)}
        f.close()

        with open(lst_train_fname, "r") as f:
            fnames["train"] = [fname.replace("\n", "") for fname in f.readlines()]
        f.close()

        for i in range(1, 5):
            with open(lst_test_fname[i - 1], "r") as f:
                fnames["test"].append([spk.replace("\n", "") for spk in f.readlines()])
            f.close()

            with open(lst_test_spk[i - 1], "r") as f:
                speakers = [spk.replace("\n", "") for spk in f.readlines()]
                tmp_speech = [Path(dl_manager.download_and_extract(_URLS["test"]["speech"] + str(i) + "/" + spk + ".zip")) for spk in speakers]
                tmp_text = [Path(dl_manager.download_and_extract(_URLS["test"]["text"] + str(i) + "/" + spk + ".zip")) for spk in speakers]
                tmp_dict_speech = {speech[:-4]: os.path.join(spk, speech) for spk in tmp_speech for speech in os.listdir(spk)}
                tmp_dict_text = {text[:-4]: os.path.join(spk, text) for spk in tmp_text for text in os.listdir(spk)}
            f.close()

            for k, v in tmp_dict_speech.items():
                if k in speech["test"]:
                    continue
                else:
                    speech["test"][k] = v

            for k, v in tmp_dict_text.items():
                if k in text["test"]:
                    continue
                else:
                    text["test"][k] = v

        fnames["test"] = list(chain(*fnames["test"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "fnames": fnames["train"],
                        "speech": speech["train"],
                        "text": text["train"],
                    },
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "fnames": fnames["test"],
                        "speech": speech["test"],
                        "text": text["test"],
                    },
                    "split": "test",
                },
            ),
        ]

    @staticmethod
    def text_process(utterance_path):
        with open(utterance_path, "r") as f:
            w = [r.replace("\n", "") for r in f.readlines()]
        f.close()
        return " ".join(w[1:-1])

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        had_used = []
        for key, example in enumerate(filepath["fnames"]):
            if example not in had_used:
                had_used.append(example)
                spk_id, _ = example.split("_")
                if self.config.schema == "source":
                    yield key, {
                        "id": example,
                        "speaker_id": spk_id,
                        "gender": spk_id[0],
                        "path": filepath["speech"][example],
                        "audio": filepath["speech"][example],
                        "text": self.text_process(filepath["text"][example]),
                    }

                elif self.config.schema == "nusantara_sptext":
                    yield key, {
                        "id": example,
                        "speaker_id": spk_id,
                        "text": self.text_process(filepath["text"][example]),
                        "path": filepath["speech"][example],
                        "audio": filepath["speech"][example],
                        "metadata": {
                            "speaker_age": None,
                            "speaker_gender": spk_id[0],
                        },
                    }
            else:
                continue
