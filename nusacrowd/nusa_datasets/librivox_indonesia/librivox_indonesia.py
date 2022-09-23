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
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

import pandas as pd

_CITATION = """\
@misc{
   research, 
   title={indonesian-nlp/librivox-indonesia · datasets at hugging face}, 
   url={https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia},
   author={Indonesian-nlp}
} 
"""

_DATASETNAME = "librivox_indonesia"
_DESCRIPTION = """\
The LibriVox Indonesia dataset consists of MP3 audio and a corresponding text file we generated from the public domain audiobooks LibriVox. 
We collected only languages in Indonesia for this dataset. 
The original LibriVox audiobooks or sound files' duration varies from a few minutes to a few hours. 
Each audio file in the speech dataset now lasts from a few seconds to a maximum of 20 seconds.
We converted the audiobooks to speech datasets using the forced alignment software we developed. 
It supports multilingual, including low-resource languages, such as Acehnese, Balinese, or Minangkabau. 
We can also use it for other languages without additional work to train the model.
The dataset currently consists of 8 hours in 7 languages from Indonesia. 
We will add more languages or audio files as we collect them. 
"""

_HOMEPAGE = "https://huggingface.co/indonesian-nlp/librivox-indonesia"

_LICENSE = "CC0"

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia/resolve/main/data",
}
_LANGUAGES = {"ind", "sun", "jav", "min", "bug", "ban", "ace"}
_LANG_CODE = {
    "ind": ["ind", "indonesian"],
    "sun": ["sun", "sundanese"],
    "jav": ["jav", "javanese"],
    "min": ["min", "minangkabau"],
    "bug": ["bug", "bugisnese"],
    "ban": ["bal", "balinese"],
    "ace": ["ace", "acehnese"]
}
_LOCAL = False
_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class LibrivoxIndonesia(datasets.GeneratorBasedBuilder):
    """
    Librivox-indonesia is a speech-to-text dataset in 7 languages available in Indonesia.
    The default dataloader contains all languages, while the other available dataloaders contain a designated language.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="librivox_indonesia_source",
            version=_SOURCE_VERSION,
            description="Librivox-Indonesia source schema for all languages",
            schema="source",
            subset_id="librivox_indonesia",
        )] + [
        NusantaraConfig(
            name="librivox_indonesia_{lang}_source".format(lang=lang),
            version=_SOURCE_VERSION,
            description="Librivox-Indonesia source schema for {lang} languages".format(lang=_LANG_CODE[lang][1]),
            schema="source",
            subset_id="librivox_indonesia_{lang}".format(lang=lang),
        ) for lang in _LANGUAGES] + [
        NusantaraConfig(
            name="librivox_indonesia_nusantara_sptext",
            version=_NUSANTARA_VERSION,
            description="Librivox-Indonesia Nusantara schema for all languages",
            schema="nusantara_sptext",
            subset_id="librivox_indonesia",
        )] + [
        NusantaraConfig(
            name="librivox_indonesia_{lang}_nusantara_sptext".format(lang=lang),
            version=_NUSANTARA_VERSION,
            description="Librivox-Indonesia Nusantara schema for {lang} languages".format(lang=_LANG_CODE[lang][1]),
            schema="nusantara_sptext",
            subset_id="librivox_indonesia_{lang}".format(lang=lang),
        )for lang in _LANGUAGES]

    DEFAULT_CONFIG_NAME = "librivox_indonesia_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "reader": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=44100)
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

        audio_path = {}
        local_extracted_archive = {}
        metadata_path = {}
        splits = ["train", "test"]
        for split in splits:
            audio_path[split] = dl_manager.download(os.path.join(urls, "audio_{split}.tgz".format(split=split)))
            local_extracted_archive[split] = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
            metadata_path[split] = dl_manager.download_and_extract(
                os.path.join(urls, "metadata_{split}.csv.gz".format(split=split))
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive["train"],
                    "audio_path": dl_manager.iter_archive(audio_path["train"]),
                    "metadata_path": metadata_path["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive["test"],
                    "audio_path": dl_manager.iter_archive(audio_path["test"]),
                    "metadata_path": metadata_path["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, local_extracted_archive: Path, audio_path, metadata_path: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(
            metadata_path,
            encoding="utf-8"
        )
        lang = self.config.subset_id.split("_")[-1]
        if lang != "indonesia":
            lang = _LANG_CODE[lang][0]
        path_to_audio = "librivox-indonesia"
        metadata = {}
        for id, row in df.iterrows():
            if lang == row["language"] or lang == "indonesia":
                path = os.path.join(path_to_audio, row["path"])
                metadata[path] = row
                metadata[path]["id"] = id

        for path, f in audio_path:
            if path in metadata:
                row = metadata[path]
                path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                if self.config.schema == "source":
                    yield row["id"], {
                        "path": path,
                        "language": row["language"],
                        "reader": row["reader"],
                        "sentence": row["sentence"],
                        "audio": path,
                    }
                elif self.config.schema == "nusantara_sptext":
                    yield row["id"], {
                        "id": row["id"],
                        "speaker_id": row["reader"],
                        "path": path,
                        "audio": path,
                        "text": row["sentence"],
                        "metadata": {
                            "speaker_age": None,
                            "speaker_gender": None,
                        }
                    }
