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
@inproceedings{sakti-tts-cocosda-2008,
    title = "Development of HMM-based Indonesian Speech Synthesis",
    author = "Sakti, Sakriani and Maia, Ranniery and Sakai, Shinsuke and Nakamura, Satoshi",
    booktitle = "Proc. Oriental COCOSDA",
    year = "2008",
    pages = "215--220"
    address = "Kyoto, Japan"
}

@inproceedings{sakti-tts-malindo-2010,
    title = "Quality and Intelligibility Assessment of Indonesian HMM-Based Speech Synthesis System",
    author = "Sakti, Sakriani and Sakai, Shinsuke and Isotani, Ryosuke and Kawai, Hisashi and Nakamura, Satoshi",
    booktitle = "Proc. MALINDO",
    year = "2010",
    pages = "51--57"
    address = "Jakarta, Indonesia"
}

@article{sakti-s2st-csl-2013,
    title = "{A-STAR}: Toward Tranlating Asian Spoken Languages",
    author = "Sakti, Sakriani and Paul, Michael and Finch, Andrew and Sakai, Shinsuke and Thang, Tat Vu, and Kimura, Noriyuki 
    and Hori, Chiori and Sumita, Eiichiro and Nakamura, Satoshi and Park, Jun and Wutiwiwatchai, Chai and Xu, Bo and Riza, Hammam 
    and Arora, Karunesh and Luong, Chi Mai and Li, Haizhou",
    journal = "Special issue on Speech-to-Speech Translation, Computer Speech and Language Journal",
    volume = "27",
    number ="2",
    pages = "509--527",
    year = "2013",
    publisher = "Elsevier"
}
"""

_DATASETNAME = "INDspeech_NEWS_TTS"
_LANGUAGES = ["ind"]

_DESCRIPTION = """\
INDspeech_NEWS_TTS is a speech dataset for developing an Indonesian text-to-speech synthesis system. The data was developed by Advanced Telecommunication Research Institute International (ATR) Japan under the the Asian speech translation advanced research (A-STAR) project [Sakti et al., 2013].
"""
_HOMEPAGE = "https://github.com/s-sakti/data_indsp_news_tts"

_LICENSE = "CC-BY-NC-SA 4.0"

_TRAIN_TASKS = {"120": "Orig_trainset_120min.lst", "60": "Orig_trainset_60min.lst", "30": "Orig_trainset_30min.lst", "12": "Orig_trainset_12min.lst", "ZR": "ZRChallenge_trainset.lst"}

_TEST_TASKS = {
    "MOS": "Orig_testset_MOS.lst",
    # "SUS": "Orig_testset_SUS.lst",
    "ZR": "ZRChallenge_testset.lst",
}

_URLS = {"lst_": "https://github.com/s-sakti/data_indsp_news_tts/raw/main/lst/", "speech_": "https://github.com/s-sakti/data_indsp_news_tts/raw/main/speech/", "text_": "https://github.com/s-sakti/data_indsp_news_tts/raw/main/text/orig_transcript"}

_SUPPORTED_TASKS = [Tasks.TEXT_TO_SPEECH]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"
_LOCAL = False


def nusantara_config_constructor(schema, version, train_task, test_task):

    if schema != "source" and schema != "nusantara_sptext":
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="indspeech_news_tts_{tr_task}_{ts_task}_{schema}".format(schema=schema, tr_task=train_task, ts_task=test_task),
        version=datasets.Version(version),
        description="indspeech_news_tts {schema} schema for {tr_task} train and {ts_task} test task".format(schema=schema, tr_task=train_task, ts_task=test_task),
        schema=schema,
        subset_id="indspeech_news_tts_{tr_task}_{ts_task}".format(tr_task=train_task, ts_task=test_task),
    )


class INDspeechNEWSTTS(datasets.GeneratorBasedBuilder):
    """
    Tasks:
    Original = Train [120, 60, 30, 12], Test [MOS, SUS]
    ZR = Train, Test
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = (
        [nusantara_config_constructor("source", _SOURCE_VERSION, train, test) for train in ["12", "30", "60", "120"] for test in ["MOS"]]
        + [nusantara_config_constructor("nusantara_sptext", _NUSANTARA_VERSION, train, test) for train in ["12", "30", "60", "120"] for test in ["MOS"]]
        + [nusantara_config_constructor("source", _SOURCE_VERSION, "ZR", "ZR")]
        + [nusantara_config_constructor("nusantara_sptext", _NUSANTARA_VERSION, "ZR", "ZR")]
    )

    DEFAULT_CONFIG_NAME = "indspeech_news_tts_120_MOS_source"

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
        """Returns SplitGenerators."""
        tr_task = self.config.name.split("_")[3]  # [12,30,60,120,ZR]
        ts_task = self.config.name.split("_")[4]  # [MOS, ZR]

        lst_train_dir = Path(dl_manager.download_and_extract(_URLS["lst_"] + _TRAIN_TASKS[tr_task]))
        lst_test_dir = Path(dl_manager.download_and_extract(_URLS["lst_"] + _TEST_TASKS[ts_task]))
        txt_dir = Path(dl_manager.download_and_extract(_URLS["text_"]))
        speech_dir = {"SPK00_" + str(spk).zfill(2) + "00": Path(dl_manager.download_and_extract(_URLS["speech_"] + "SPK00_" + str(spk).zfill(2) + "00.zip") + "/SPK00_" + str(spk).zfill(2) + "00") for spk in range(0, 21)}
        # print(os.listdir(speech_dir["SPK00_1500"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": {"samples": lst_train_dir, "text": txt_dir, "speech": speech_dir},
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {"samples": lst_test_dir, "text": txt_dir, "speech": speech_dir},
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        samples = open(filepath["samples"], "r").read().splitlines()

        transcripts = {}
        with open(filepath["text"]) as file:
            for line in file:
                key, text = line.replace("\n", "").split("\t")
                transcripts[key] = text

        for key, id in enumerate(samples):
            spk_id, gender, speech_id = id.split("_")
            spk_group = speech_id[:2]

            if self.config.schema == "source":
                example = {
                    "id": id,
                    "speaker_id": spk_id,
                    "path": os.path.join(filepath["speech"]["SPK00_" + spk_group + "00"], id + ".wav"),
                    "audio": os.path.join(filepath["speech"]["SPK00_" + spk_group + "00"], id + ".wav"),
                    "text": transcripts[id],
                    "gender": gender,
                }
                yield key, example
            elif self.config.schema == "nusantara_sptext":
                example = {
                    "id": str(id),
                    "speaker_id": spk_id,
                    "path": os.path.join(filepath["speech"]["SPK00_" + spk_group + "00"], id + ".wav"),
                    "audio": os.path.join(filepath["speech"]["SPK00_" + spk_group + "00"], id + ".wav"),
                    "text": transcripts[id],
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": gender,
                    },
                }
                yield key, example
