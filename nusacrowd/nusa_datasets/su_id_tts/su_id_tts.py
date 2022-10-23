import csv
import os
from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "su_id_tts"
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
This data set contains high-quality transcribed audio data for Sundanese. The data set consists of wave files, and a TSV file. The file line_index.tsv contains a filename and the transcription of audio in the file. Each filename is prepended with a speaker identification number.
The data set has been manually quality checked, but there might still be errors.
This dataset was collected by Google in collaboration with Universitas Pendidikan Indonesia.
"""

_HOMEPAGE = "http://openslr.org/44/"

_LICENSE = "CC BY-SA 4.0"

_URLs = {
    _DATASETNAME: {
        "female": "https://www.openslr.org/resources/44/su_id_female.zip",
        "male": "https://www.openslr.org/resources/44/su_id_male.zip",
    }
}

_SUPPORTED_TASKS = [Tasks.TEXT_TO_SPEECH]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class SuIdTTS(datasets.GeneratorBasedBuilder):
    """su_id_tts contains high-quality Multi-speaker TTS data for Sundanese (SU-ID)."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="su_id_tts_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="SU_ID_TTS source schema",
            schema="source",
            subset_id="su_id_tts",
        ),
        NusantaraConfig(
            name="su_id_tts_nusantara_sptext",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="SU_ID_TTS Nusantara schema",
            schema="nusantara_sptext",
            subset_id="su_id_tts",
        ),
    ]

    DEFAULT_CONFIG_NAME = "su_id_tts_source"

    def _info(self):
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
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        male_path = Path(dl_manager.download_and_extract(_URLs[_DATASETNAME]["male"]))
        female_path = Path(dl_manager.download_and_extract(_URLs[_DATASETNAME]["female"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "male_filepath": male_path,
                    "female_filepath": female_path,
                },
            ),
        ]

    def _generate_examples(self, male_filepath: Path, female_filepath: Path):

        if self.config.schema == "source" or self.config.schema == "nusantara_sptext":
            tsv_m = os.path.join(male_filepath, "su_id_male", "line_index.tsv")
            tsv_f = os.path.join(female_filepath, "su_id_female", "line_index.tsv")

            with open(tsv_m, "r") as file:
                tsv_m_data = csv.reader(file, delimiter="\t")
                for line in tsv_m_data:
                    spk_trans_info = line[0].split("_")
                    if self.config.schema == "source":
                        ex = {
                            "id": line[0],
                            "speaker_id": spk_trans_info[0] + "_" + spk_trans_info[1],
                            "path": os.path.join(male_filepath, "su_id_male", "wavs", "{}.wav".format(line[0])),
                            "audio": os.path.join(male_filepath, "su_id_male", "wavs", "{}.wav".format(line[0])),
                            "text": line[2],
                            "gender": spk_trans_info[0][2],
                        }
                        yield line[0], ex

                    elif self.config.schema == "nusantara_sptext":
                        ex = {
                            "id": line[0],
                            "speaker_id": spk_trans_info[0] + "_" + spk_trans_info[1],
                            "path": os.path.join(male_filepath, "su_id_male", "wavs", "{}.wav".format(line[0])),
                            "audio": os.path.join(male_filepath, "su_id_male", "wavs", "{}.wav".format(line[0])),
                            "text": line[2],
                            "metadata": {
                                "speaker_age": None,
                                "speaker_gender": spk_trans_info[0][2],
                            },
                        }
                        yield line[0], ex

            with open(tsv_f, "r") as file:
                tsv_f_data = csv.reader(file, delimiter="\t")
                for line in tsv_f_data:
                    spk_trans_info = line[0].split("_")
                    if self.config.schema == "source":
                        ex = {
                            "id": line[0],
                            "speaker_id": spk_trans_info[0] + "_" + spk_trans_info[1],
                            "path": os.path.join(female_filepath, "su_id_female", "wavs", "{}.wav".format(line[0])),
                            "audio": os.path.join(female_filepath, "su_id_female", "wavs", "{}.wav".format(line[0])),
                            "text": line[2],
                            "gender": spk_trans_info[0][2],
                        }
                        yield line[0], ex

                    elif self.config.schema == "nusantara_sptext":
                        ex = {
                            "id": line[0],
                            "speaker_id": spk_trans_info[0] + "_" + spk_trans_info[1],
                            "path": os.path.join(female_filepath, "su_id_female", "wavs", "{}.wav".format(line[0])),
                            "audio": os.path.join(female_filepath, "su_id_female", "wavs", "{}.wav".format(line[0])),
                            "text": line[2],
                            "metadata": {
                                "speaker_age": None,
                                "speaker_gender": spk_trans_info[0][2],
                            },
                        }
                        yield line[0], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
