from pathlib import Path
from typing import List

import datasets
import json
import os

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "indo_lvcsr"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{lestari2006indolvcsr,
  title={A large vocabulary continuous speech recognition system for Indonesian language},
  author={Lestari, Dessi Puji and Iwano, Koji and Furui, Sadaoki},
  booktitle={15th Indonesian Scientific Conference in Japan Proceedings},
  pages={17--22},
  year={2006}
}
"""

_DESCRIPTION = """\
IndoLVCSR is collected to build a pioneering Indonesian Large Vocabulary Continuous Speech Recognition (LVCSR) System. In order to build an LVCSR system, high accurate acoustic models and large-scale language models are essential. Since Indonesian speech corpus was not available yet, we tried to collect speech data from 20 Indonesian native speakers (11 males and 9 females) to construct a speech corpus for training the acoustic model based on Hidden Markov Models (HMMs). A text corpus which was collected by ILPS, Informatics Institute, University of Amsterdam, was used to build a 40K-vocabulary dictionary and a n-gram language model.
"""

_HOMEPAGE = "http://research.nii.ac.jp/src/en/TITML-IDN.html"

_LICENSE = "For research purposes only"

_URLs = {"indo-lvcsr": "https://huggingface.co/datasets/holylovenia/IndoLVCSR/resolve/main/IndoLVCSR.zip"}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndoLVCSR(datasets.GeneratorBasedBuilder):
    """IndoLVCSR is a speech recognition dataset containing Indonesian speech collected with transcriptions from newpaper and magazine articles."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indo_lvcsr_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="IndoLVCSR source schema",
            schema="source",
            subset_id="indo_lvcsr",
        ),
        NusantaraConfig(
            name="indo_lvcsr_nusantara_asr",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="IndoLVCSR Nusantara schema",
            schema="nusantara_asr",
            subset_id="indo_lvcsr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indo_lvcsr_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_asr":
            features = schemas.asr_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[datasets.AutomaticSpeechRecognition(audio_file_path_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        base_path = dl_manager.download_and_extract(_URLs["indo-lvcsr"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": base_path},
            ),
        ]

    def _generate_examples(self, filepath: Path, n_speakers=20):

        if self.config.schema == "source" or self.config.schema == "nusantara_asr":

            for speaker_id in range(1, n_speakers + 1):
                speaker_id = str(speaker_id).zfill(2)
                dir_path = os.path.join(filepath, speaker_id)
                transcription_path = os.path.join(dir_path, "script~")

                with open(transcription_path, "r+") as f:
                    for line in f:
                        audio_id = line[2:8]
                        text = line[9:].strip()
                        wav_path = os.path.join(dir_path, "{}.wav".format(audio_id))

                        if os.path.exists(wav_path):
                            if self.config.schema == "source":
                                ex = {
                                    "id": audio_id,
                                    "speaker": speaker_id,
                                    "path": wav_path,
                                    "audio": wav_path,
                                    "text": text,
                                }
                                yield audio_id, ex
                            elif self.config.schema == "nusantara_asr":
                                ex = {
                                    "id": audio_id,
                                    "speaker": speaker_id,
                                    "path": wav_path,
                                    "audio": wav_path,
                                    "text": text,
                                    "metadata": {
                                        "speaker_age": None,
                                        "speaker_gender": None,
                                    }
                                }
                                yield audio_id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
