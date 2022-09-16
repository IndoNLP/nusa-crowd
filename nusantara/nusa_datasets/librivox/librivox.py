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

"""
This template serves as a starting point for contributing a dataset to the Nusantara Dataset repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

sptext = (kb, pairs, qa, text, t2t, entailment)
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

import pandas as pd
_CITATION = """\
@misc{
   research, 
   title={indonesian-nlp/librivox-indonesia · datasets at hugging face}, 
   url={https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia},
   author={Indonesian-nlp}
} 
"""

_DATASETNAME = "librivox"
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

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
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


class Librivox(datasets.GeneratorBasedBuilder):
    """
    Librivox-indonesia is a speech-to-text dataset in 7 languages available in Indonesia.
    The default dataloader contains all languages, while the other available dataloaders contain a designated language.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="librivox_source",
            version=_SOURCE_VERSION,
            description="librivox source schema for all languages",
            schema="source",
            subset_id="librivox",
        )] + [
        NusantaraConfig(
            name="librivox_{lang}_source".format(lang=lang),
            version=_SOURCE_VERSION,
            description="librivox source schema for {lang} languages".format(lang=_LANG_CODE[lang][1]),
            schema="source",
            subset_id="librivox_{lang}".format(lang=lang),
        ) for lang in _LANGUAGES] + [
        NusantaraConfig(
            name="librivox_nusantara_sptext",
            version=_NUSANTARA_VERSION,
            description="librivox Nusantara schema for all languages",
            schema="nusantara_sptext",
            subset_id="librivox",
        )] + [
        NusantaraConfig(
            name="librivox_{lang}_nusantara_sptext".format(lang=lang),
            version=_NUSANTARA_VERSION,
            description="librivox Nusantara schema for {lang} languages".format(lang=_LANG_CODE[lang][1]),
            schema="nusantara_sptext",
            subset_id="librivox_{lang}".format(lang=lang),
        )for lang in _LANGUAGES]

    DEFAULT_CONFIG_NAME = "librivox_source"

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
        # For example nusantara_kb, nusantara_t2t
        elif self.config.schema == "nusantara_sptext":
            features = schemas.speech_text_features          # TODO: Choose your nusantara schema here

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
        metadata_path = {}
        splits = ["train", "test"]
        for split in splits:
            audio_path[split] = dl_manager.download_and_extract(os.path.join(urls, "audio_{split}.tgz".format(split=split)))
            metadata_path[split] = dl_manager.download_and_extract(
                os.path.join(urls, "metadata_{split}.csv.gz".format(split=split))
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "audio_path": audio_path["train"],
                    "metadata_path": metadata_path["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_path": audio_path["test"],
                    "metadata_path": metadata_path["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, audio_path: Path, metadata_path: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(
            metadata_path,
            encoding="utf-8"
        )
        lang = self.config.subset_id.split("_")[-1]
        if lang != "librivox":
            lang = _LANG_CODE[lang][0]
        path_to_audio = "librivox-indonesia"
        for id, row in df.iterrows():
            if lang == row["language"] or lang == "librivox":
                if self.config.schema == "source":
                    yield id, {
                        "path": row["path"],
                        "language": row["language"],
                        "reader": row["reader"],
                        "sentence": row["sentence"],
                        "audio": os.path.join(audio_path, path_to_audio, row["path"])
                    }
                elif self.config.schema == "nusantara_sptext":
                    yield id, {
                        "id": id,
                        "speaker_id": row["reader"],
                        "path": row["path"],
                        "audio": os.path.join(audio_path, path_to_audio, row["path"]),
                        "text": row["sentence"],
                        "metadata": {
                            "speaker_age": None,
                            "speaker_gender": None,
                        }
                    }
