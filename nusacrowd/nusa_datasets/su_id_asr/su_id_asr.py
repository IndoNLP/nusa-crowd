import csv
import os
from typing import Dict, List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "su_id_asr"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["sun"]
_LOCAL = False
_CITATION = """\
@inproceedings{sodimana18_sltu,
  author={Keshan Sodimana and Pasindu {De Silva} and Supheakmungkol Sarin and Oddur Kjartansson and Martin Jansche and Knot Pipatsrisawat and Linne Ha},
  title={{A Step-by-Step Process for Building TTS Voices Using Open Source Data and Frameworks for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese}},
  year=2018,
  booktitle={Proc. 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2018)},
  pages={66--70},
  doi={10.21437/SLTU.2018-14}
}
"""

_DESCRIPTION = """\
Sundanese ASR training data set containing ~220K utterances.
This dataset was collected by Google in Indonesia.


"""

_HOMEPAGE = "https://indonlp.github.io/nusa-catalogue/card.html?su_id_asr"

_LICENSE = "Attribution-ShareAlike 4.0 International."

_URLs = {
    "su_id_asr": "https://www.openslr.org/resources/36/asr_sundanese_{}.zip",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class SuIdASR(datasets.GeneratorBasedBuilder):
    """su_id contains ~220K utterances for Sundanese ASR training data."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="su_id_asr_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="SU_ID_ASR source schema",
            schema="source",
            subset_id="su_id_asr",
        ),
        NusantaraConfig(
            name="su_id_asr_nusantara_sptext",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="SU_ID_ASR Nusantara schema",
            schema="nusantara_sptext",
            subset_id="su_id_asr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "su_id_asr_source"

    def _info(self):
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
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        base_path = {}
        for id in range(10):
            base_path[id] = dl_manager.download_and_extract(_URLs["su_id_asr"].format(str(id)))
        for id in ["a", "b", "c", "d", "e", "f"]:
            base_path[id] = dl_manager.download_and_extract(_URLs["su_id_asr"].format(str(id)))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": base_path},
            ),
        ]

    def _generate_examples(self, filepath: Dict):

        if self.config.schema == "source" or self.config.schema == "nusantara_sptext":

            for key, each_filepath in filepath.items():

                tsv_file = os.path.join(each_filepath, "asr_sundanese", "utt_spk_text.tsv")

                with open(tsv_file, "r") as file:
                    tsv_file = csv.reader(file, delimiter="\t")

                    for line in tsv_file:
                        audio_id, speaker_id, transcription_text = line[0], line[1], line[2]

                        wav_path = os.path.join(each_filepath, "asr_sundanese", "data", "{}".format(audio_id[:2]), "{}.flac".format(audio_id))

                        if os.path.exists(wav_path):
                            if self.config.schema == "source":
                                ex = {
                                    "id": audio_id,
                                    "speaker_id": speaker_id,
                                    "path": wav_path,
                                    "audio": wav_path,
                                    "text": transcription_text,
                                }
                                yield audio_id, ex
                            elif self.config.schema == "nusantara_sptext":
                                ex = {
                                    "id": audio_id,
                                    "speaker_id": speaker_id,
                                    "path": wav_path,
                                    "audio": wav_path,
                                    "text": transcription_text,
                                    "metadata": {
                                        "speaker_age": None,
                                        "speaker_gender": None,
                                    },
                                }
                                yield audio_id, ex

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
