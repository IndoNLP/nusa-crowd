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
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_CITATION = """\
@inproceedings{sakti-tcast-2008,
    title = "Development of {I}ndonesian Large Vocabulary Continuous Speech Recognition System within {A-STAR} Project",
    author = "Sakti, Sakriani and Kelana, Eka and Riza, Hammam and Sakai, Shinsuke and Markov, Konstantin and Nakamura, Satoshi",
    booktitle = "Proc. IJCNLP Workshop on Technologies and Corpora for Asia-Pacific Speech Translation (TCAST)",
    year = "2008",
    pages = "19--24"
    address = "Hyderabad, India"
}

@inproceedings{sakti-icslp-2004,
    title = "Indonesian Speech Recognition for Hearing and Speaking Impaired People",
    author = "Sakti, Sakriani and Hutagaol, Paulus and Arman, Arry Akhmad and Nakamura, Satoshi",
    booktitle = "Proc. International Conference on Spoken Language Processing (INTERSPEECH - ICSLP)",
    year = "2004",
    pages = "1037--1040"
    address = "Jeju Island, Korea"
}

@article{sakti-s2st-csl-2013,
    title = "{A-STAR}: Toward Translating Asian Spoken Languages",
    author = "Sakti, Sakriani and Paul, Michael and Finch, Andrew and Sakai, Shinsuke and Thang, Tat Vu, and Kimura, Noriyuki and Hori, Chiori and Sumita, Eiichiro and Nakamura, Satoshi and Park, Jun and Wutiwiwatchai, Chai and Xu, Bo and Riza, Hammam and Arora, Karunesh and Luong, Chi Mai and Li, Haizhou",
    journal = "Special issue on Speech-to-Speech Translation, Computer Speech and Language Journal",
    volume = "27",
    number ="2",
    pages = "509--527",
    year = "2013",
    publisher = "Elsevier"
}
"""

_DATASETNAME = "indspeech_news_lvcsr"
_LANGUAGES = ["ind"] 
_LOCAL = False

_DESCRIPTION = """\
This is the first Indonesian speech dataset for large vocabulary continuous speech recognition (LVCSR) with more than 40 hours of speech and 400 speakers [Sakti et al., 2008]. R&D Division of PT Telekomunikasi Indonesia (TELKOMRisTI) developed the data in 2005-2006, in collaboration with Advanced Telecommunication Research Institute International (ATR) Japan, as the continuation of the Asia-Pacific Telecommunity (APT) project [Sakti et al., 2004]. It has also been successfully used for developing Indonesian LVCSR in the Asian speech translation advanced research (A-STAR) project [Sakti et al., 2013].
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_news_lvcsr"

_LICENSE = "CC BY-NC-SA 4.0"

URL_TEMPLATE = {
    "lst": "https://raw.githubusercontent.com/s-sakti/data_indsp_news_lvcsr/main/lst/",  # transcript.lst
    "speech": "https://github.com/s-sakti/data_indsp_news_lvcsr/raw/main/speech/",  # Ind3/Ind304.zip~Ind400.zip
    "text": "https://github.com/s-sakti/data_indsp_news_lvcsr/raw/main/text/",  # all_transcript.zip
}

_URLS = {
    "lst_spk_Ind": [URL_TEMPLATE["lst"] + "spk_Ind" + str(n) + ".lst" for n in range(0, 4)],
    "lst_spk_all": URL_TEMPLATE["lst"] + "spk_all.lst",
    "lst_spk_test": URL_TEMPLATE["lst"] + "spk_test.lst",
    "lst_spk_train": URL_TEMPLATE["lst"] + "spk_train.lst",
    "lst_transcript": URL_TEMPLATE["lst"] + "transcript.lst",
    "speech_Ind": [URL_TEMPLATE["speech"] + "Ind" + str(n) + "/Ind" + str(p).zfill(3) + ".zip" for n in range(0, 4) for p in range(n * 100 + 1, n * 100 + 101)],
    "transcript_all": URL_TEMPLATE["text"] + "all_transcript.zip",
    "transcript_spk": URL_TEMPLATE["text"] + "spk_transcript.zip",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndSpeechNewsLVCSR(datasets.GeneratorBasedBuilder):
    """Indonesian automatic speech recognition with several local accents reading short news sentences"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indspeech_news_lvcsr_source",
            version=SOURCE_VERSION,
            description="indspeech_news_lvcsr source schema",
            schema="source",
            subset_id="indspeech_news_lvcsr",
        ),
        NusantaraConfig(
            name="indspeech_news_lvcsr_nusantara_sptext",
            version=NUSANTARA_VERSION,
            description="indspeech_news_lvcsr Nusantara schema",
            schema="nusantara_sptext",
            subset_id="indspeech_news_lvcsr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indspeech_news_lvcsr_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "spk_id": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "accent": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "txt_id": datasets.Value("string"),
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
        lst_spk_train = _URLS["lst_spk_train"]
        lst_spk_test = _URLS["lst_spk_test"]
        transcript = _URLS["lst_transcript"]
        audio_urls = _URLS["speech_Ind"]

        lst_spk_train_dir = Path(dl_manager.download_and_extract(lst_spk_train))
        lst_spk_test_dir = Path(dl_manager.download_and_extract(lst_spk_test))
        transcript_dir = Path(dl_manager.download_and_extract(transcript))
        audio_files_dir = [Path(dl_manager.download_and_extract(aud_url)) / aud_url.split("/")[-1][:-4] for aud_url in audio_urls]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {"lst_spk": lst_spk_train_dir, "transcript": transcript_dir, "aud_files": audio_files_dir},
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {"lst_spk": lst_spk_test_dir, "transcript": transcript_dir, "aud_files": audio_files_dir},
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        speaker_ids = open(filepath["lst_spk"], "r").readlines()
        speaker_ids = [id.replace("\n", "") for id in speaker_ids]
        speech_folders = [aud_folder for aud_folder in filepath["aud_files"] if aud_folder.name in speaker_ids]
        speech_files = list(chain(*[list(map((str(speech_folder) + "/").__add__, os.listdir(speech_folder))) for speech_folder in speech_folders]))

        print(speech_files[0])

        transcript = open(filepath["transcript"], "r").readlines()
        transcript = [sentence.replace("\n", "") for sentence in transcript]

        for key, aud_file in enumerate(speech_files):
            aud_id = str(Path(aud_file).name).split("\\")[-1][:-4]
            aud_info = aud_id.split("_")

            if self.config.schema == "source":
                row = {
                    "spk_id": aud_info[0],
                    "gender": aud_info[1],
                    "accent": aud_info[2],
                    "type": aud_info[3],
                    "txt_id": aud_info[5],
                    "audio": aud_file,
                    "text": transcript[int(aud_info[5])],
                }
                yield key, row

            elif self.config.schema == "nusantara_sptext":
                row = {
                    "id": aud_id,
                    "path": aud_file,
                    "audio": aud_file,
                    "text": transcript[int(aud_info[5])],
                    "speaker_id": aud_info[0],
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": aud_info[1],
                    },
                }
                yield key, row
            else:
                raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
