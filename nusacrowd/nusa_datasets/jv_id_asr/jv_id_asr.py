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
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{kjartansson-etal-sltu2018,
    title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese,  Sinhala, Nepali, and Bangladeshi Bengali}},
    author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
    booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
    year  = {2018},
    address = {Gurugram, India},
    month = aug,
    pages = {52--55},
    URL   = {http://dx.doi.org/10.21437/SLTU.2018-11},
  }
"""

_DATASETNAME = "jv_id_asr"

_DESCRIPTION = """\
This data set contains transcribed audio data for Javanese. The data set consists of wave files, and a TSV file.
The file utt_spk_text.tsv contains a FileID, UserID and the transcription of audio in the file.
The data set has been manually quality checked, but there might still be errors.
This dataset was collected by Google in collaboration with Reykjavik University and Universitas Gadjah Mada in Indonesia.
"""

_HOMEPAGE = "http://openslr.org/35/"
_LANGUAGES = ["jav"]
_LOCAL = False

_LICENSE = "Attribution-ShareAlike 4.0 International"

_URLS = {
    _DATASETNAME: "https://www.openslr.org/resources/35/asr_javanese_{}.zip",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class JvIdASR(datasets.GeneratorBasedBuilder):
    """Javanese ASR training data set containing ~185K utterances."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="jv_id_asr_source",
            version=SOURCE_VERSION,
            description="jv_id_asr source schema",
            schema="source",
            subset_id="jv_id_asr",
        ),
        NusantaraConfig(
            name="jv_id_asr_nusantara_sptext",
            version=NUSANTARA_VERSION,
            description="jv_id_asr Nusantara schema",
            schema="nusantara_sptext",
            subset_id="jv_id_asr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "jv_id_asr_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        base_path = {}
        for id in range(10):
            base_path[id] = dl_manager.download_and_extract(urls.format(str(id)))
        for id in ["a", "b", "c", "d", "e", "f"]:
            base_path[id] = dl_manager.download_and_extract(urls.format(str(id)))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": base_path},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        for key, fp in filepath.items():
            tsv_file = os.path.join(fp, "asr_javanese", "utt_spk_text.tsv")
            with open(tsv_file, "r") as f:
                tsv_file = csv.reader(f, delimiter="\t")
                for line in tsv_file:
                    audio_id, sp_id, text = line[0], line[1], line[2]
                    wav_path = os.path.join(fp, "asr_javanese", "data", "{}".format(audio_id[:2]), "{}.flac".format(audio_id))

                    if os.path.exists(wav_path):
                        if self.config.schema == "source":
                            ex = {
                                "id": audio_id,
                                "speaker_id": sp_id,
                                "path": wav_path,
                                "audio": wav_path,
                                "text": text,
                            }
                            yield audio_id, ex
                        elif self.config.schema == "nusantara_sptext":
                            ex = {
                                "id": audio_id,
                                "speaker_id": sp_id,
                                "path": wav_path,
                                "audio": wav_path,
                                "text": text,
                                "metadata": {
                                    "speaker_age": None,
                                    "speaker_gender": None,
                                },
                            }
                            yield audio_id, ex
            f.close()
