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

_DATASETNAME = "indspeech_teldialog_svcsr"

_DESCRIPTION = """\
This is the first Indonesian speech dataset for small vocabulary continuous speech recognition (SVCSR).
The data was developed by TELKOMRisTI (R&D Division, PT Telekomunikasi Indonesia) in collaboration with Advanced
Telecommunication Research Institute International (ATR) Japan and Bandung Institute of Technology (ITB) under the
Asia-Pacific Telecommunity (APT) project in 2004 [Sakti et al., 2004]. Although it was originally developed for
a telecommunication system for hearing and speaking impaired people, it can be used for other applications,
i.e., automatic call centers. Furthermore, as all speakers utter the same sentences,
it can also be used for voice conversion tasks.

The text is based on a word vocabulary which is derived from some necessary dialog calls,
such as dialog calls with the 119 emergency department, 108 telephone information department,
and ticket reservation department. In total, it consists of 20,000 utterances (about 18 hours of speech) from the
70-word dialog vocabulary of 100 sentences (including single word sentences) each uttered by 200 speakers
(100 Females, 100 Males). The age is limited to middle age (20-40 years), but they present a wide range of spoken
dialects from different ethnic groups. The recording is conducted in parallel for both clean and telephone speech,
but we open only the clean speech due to quality issues on telephone speech.
Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16000 Hz.
These utterances are equally split into training and test sets with 100 speakers (50 Females, 50 Males) in each set.
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_teldialog_svcsr/"

_LICENSE = "CC-BY-NC-SA-4.0"

_LANGUAGES = ["ind"]
_LOCAL = False

URL_TEMPLATE = "https://raw.githubusercontent.com/s-sakti/data_indsp_teldialog_svcsr/main/"
_URLS = {
    _DATASETNAME: {"lst": URL_TEMPLATE + "lst/", "speech": URL_TEMPLATE + "speech/", "text": URL_TEMPLATE + "text/"},
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class INDspeechTELDIALOGSVCSR(datasets.GeneratorBasedBuilder):
    """
    This is an Indonesian speech dataset on small vocabulary continuous speech recognition (SVCSR) from necessary
    dialog calls. The dataset loader is designed for speech recognition task.
    There are 20000 utterances (train: 10000, test:10000) uttered by 200 speakers (50 male 50 female each in train and
    test).
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indspeech_teldialog_svcsr_source",
            version=SOURCE_VERSION,
            description="indspeech_teldialog_svcsr source schema",
            schema="source",
            subset_id="indspeech_teldialog_svcsr",
        ),
        NusantaraConfig(
            name="indspeech_teldialog_svcsr_nusantara_sptext",
            version=NUSANTARA_VERSION,
            description="indspeech_teldialog_svcsr Nusantara schema",
            schema="nusantara_sptext",
            subset_id="indspeech_teldialog_svcsr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indspeech_teldialog_svcsr_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "speaker_id": datasets.Value("string"),
                    "gender_id": datasets.Value("string"),
                    "utterance_id": datasets.Value("string"),
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
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentences")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = {
            "spk_data": {"train": dl_manager.download_and_extract(os.path.join(urls["lst"], "train_spk.lst")), "test": dl_manager.download_and_extract(os.path.join(urls["lst"], "test_spk.lst"))},
            "wav_data": {"train": dl_manager.download_and_extract(os.path.join(urls["lst"], "train_wav.lst")), "test": dl_manager.download_and_extract(os.path.join(urls["lst"], "test_wav.lst"))},
            "txt_data": dl_manager.download_and_extract(os.path.join(urls["text"], "text.zip")),
        }
        speakers = {}
        with open(data_dir["spk_data"]["train"], "r") as f:
            speakers["train"] = [sp.replace("\n", "") for sp in f.readlines()]
        f.close()
        with open(data_dir["spk_data"]["test"], "r") as f:
            speakers["test"] = [sp.replace("\n", "") for sp in f.readlines()]
        f.close()
        data_dir["speech_path"] = {
            "train": {sp: dl_manager.download_and_extract(os.path.join(urls["speech"], "train", sp + ".zip")) for sp in speakers["train"]},
            "test": {sp: dl_manager.download_and_extract(os.path.join(urls["speech"], "test", sp + ".zip")) for sp in speakers["test"]},
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["wav_data"]["train"],
                    "audio_path": data_dir["speech_path"]["train"],
                    "text_path": data_dir["txt_data"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["wav_data"]["test"],
                    "audio_path": data_dir["speech_path"]["test"],
                    "text_path": data_dir["txt_data"],
                    "split": "test",
                },
            ),
        ]

    @staticmethod
    def text_process(utterance_txt_dir):
        with open(utterance_txt_dir + ".ANS", "r") as f:
            lines = [x.replace("\n", "") for x in f.readlines()]
        f.close()
        return " ".join(lines)

    def _generate_examples(self, filepath: Path, audio_path, text_path: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath, "r") as f:
            filelist = [x.replace("\n", "") for x in f.readlines()]
        f.close()
        for fn in filelist:
            speaker_id = fn[:3]
            gender_id = fn[:1]
            utterance_id = fn[4:8]
            _id = fn.replace(".wav", "")
            text = self.text_process(os.path.join(text_path, utterance_id))
            if self.config.schema == "source":
                yield _id, {
                    "speaker_id": speaker_id,
                    "gender_id": gender_id,
                    "utterance_id": utterance_id,
                    "audio": os.path.join(audio_path[speaker_id], fn),
                    "text": text,
                }

            elif self.config.schema == "nusantara_sptext":
                yield _id, {
                    "id": _id,
                    "speaker_id": speaker_id,
                    "text": text,
                    "path": os.path.join(audio_path[speaker_id], fn),
                    "audio": os.path.join(audio_path[speaker_id], fn),
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": gender_id,
                    },
                }
