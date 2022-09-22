from pathlib import Path
from typing import List

import datasets
import json
import os

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "titml_idn"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{lestari2006titmlidn,
  title={A large vocabulary continuous speech recognition system for Indonesian language},
  author={Lestari, Dessi Puji and Iwano, Koji and Furui, Sadaoki},
  booktitle={15th Indonesian Scientific Conference in Japan Proceedings},
  pages={17--22},
  year={2006}
}
"""

_DESCRIPTION = """\
TITML-IDN (Tokyo Institute of Technology Multilingual - Indonesian) is collected to build a pioneering Indonesian Large Vocabulary Continuous Speech Recognition (LVCSR) System. In order to build an LVCSR system, high accurate acoustic models and large-scale language models are essential. Since Indonesian speech corpus was not available yet, we tried to collect speech data from 20 Indonesian native speakers (11 males and 9 females) to construct a speech corpus for training the acoustic model based on Hidden Markov Models (HMMs). A text corpus which was collected by ILPS, Informatics Institute, University of Amsterdam, was used to build a 40K-vocabulary dictionary and a n-gram language model.
"""

_HOMEPAGE = "http://research.nii.ac.jp/src/en/TITML-IDN.html"

_LICENSE = "For research purposes only. If you use this corpus, you have to cite (Lestari et al, 2006)."

_URLs = {"titml-idn": "https://huggingface.co/datasets/holylovenia/TITML-IDN/resolve/main/IndoLVCSR.zip"}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class TitmlIdn(datasets.GeneratorBasedBuilder):
    """TITML-IDN is a speech recognition dataset containing Indonesian speech collected with transcriptions from newpaper and magazine articles."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="titml_idn_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="TITML-IDN source schema",
            schema="source",
            subset_id="titml_idn",
        ),
        NusantaraConfig(
            name="titml_idn_nusantara_sptext",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="TITML-IDN Nusantara schema",
            schema="nusantara_sptext",
            subset_id="titml_idn",
        ),
    ]

    DEFAULT_CONFIG_NAME = "titml_idn_source"

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
        base_path = dl_manager.download_and_extract(_URLs["titml-idn"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": base_path},
            ),
        ]

    def _generate_examples(self, filepath: Path, n_speakers=20):

        if self.config.schema == "source" or self.config.schema == "nusantara_sptext":

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
                                    "speaker_id": speaker_id,
                                    "path": wav_path,
                                    "audio": wav_path,
                                    "text": text,
                                }
                                yield audio_id, ex
                            elif self.config.schema == "nusantara_sptext":
                                ex = {
                                    "id": audio_id,
                                    "speaker_id": speaker_id,
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
