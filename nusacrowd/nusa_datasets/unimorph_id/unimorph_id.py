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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{pimentel-ryskina-etal-2021-sigmorphon,
    title = "SIGMORPHON 2021 Shared Task on Morphological Reinflection: Generalization Across Languages",
    author = "Pimentel, Tiago  and
      Ryskina, Maria  and
      Mielke, Sabrina J.  and
      Wu, Shijie  and
      Chodroff, Eleanor  and
      Leonard, Brian  and
      Nicolai, Garrett  and
      Ghanggo Ate, Yustinus  and
      Khalifa, Salam  and
      Habash, Nizar  and
      El-Khaissi, Charbel  and
      Goldman, Omer  and
      Gasser, Michael  and
      Lane, William  and
      Coler, Matt  and
      Oncevay, Arturo  and
      Montoya Samame, Jaime Rafael  and
      Silva Villegas, Gema Celeste  and
      Ek, Adam  and
      Bernardy, Jean-Philippe  and
      Shcherbakov, Andrey  and
      Bayyr-ool, Aziyana  and
      Sheifer, Karina  and
      Ganieva, Sofya  and
      Plugaryov, Matvey  and
      Klyachko, Elena  and
      Salehi, Ali  and
      Krizhanovsky, Andrew  and
      Krizhanovsky, Natalia  and
      Vania, Clara  and
      Ivanova, Sardana  and
      Salchak, Aelita  and
      Straughn, Christopher  and
      Liu, Zoey  and
      Washington, Jonathan North  and
      Ataman, Duygu  and
      Kiera{\'s}, Witold  and
      Woli{\'n}ski, Marcin  and
      Suhardijanto, Totok  and
      Stoehr, Niklas  and
      Nuriah, Zahroh  and
      Ratan, Shyam  and
      Tyers, Francis M.  and
      Ponti, Edoardo M.  and
      Aiton, Grant  and
      Hatcher, Richard J.  and
      Prud'hommeaux, Emily  and
      Kumar, Ritesh  and
      Hulden, Mans  and
      Barta, Botond  and
      Lakatos, Dorina  and
      Szolnok, G{\'a}bor  and
      {\'A}cs, Judit  and
      Raj, Mohit  and
      Yarowsky, David  and
      Cotterell, Ryan  and
      Ambridge, Ben  and
      Vylomova, Ekaterina",
    booktitle = "Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.sigmorphon-1.25",
    doi = "10.18653/v1/2021.sigmorphon-1.25",
    pages = "229--259"
}"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "unimorph_id"

_DESCRIPTION = """\
The UniMorph project, Indonesian chapter.
Due to sparsity of UniMorph original parsing, raw source is used instead.
Original parsing can be found on https://huggingface.co/datasets/universal_morphologies/blob/2.3.2/universal_morphologies.py
"""

_HOMEPAGE = "https://github.com/unimorph/ind"

_LICENSE = "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)"

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/unimorph/ind/main/ind",
}

_SUPPORTED_TASKS = [Tasks.MORPHOLOGICAL_INFLECTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class UnimorphIdDataset(datasets.GeneratorBasedBuilder):
    """The UniMorph project, Indonesian chapter."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    label_classes = [
        "",
        "1",
        "2",
        "3",
        "ACT",
        "ADJ",
        "ADV",
        "APPL",
        "CAUS",
        "DEF",
        "FEM",
        "FOC",
        "ITER",
        "MASC",
        "N",
        "NEG",
        "NEUT",
        "PASS",
        "POS",
        "PSS1S",
        "PSS2S",
        "PSS3S",
        "SG",
        "SPRL",
        "TR",
        "V",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_pairs_multi",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} Nusantara schema",
            schema="nusantara_pairs_multi",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "lemma": datasets.Value("string"),
                    "form": datasets.Value("string"),
                    "tag": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "nusantara_pairs_multi":
            features = schemas.pairs_multi_features(self.label_classes)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf8") as f:
            dataset = list(map(lambda l: l.rstrip("\r\n").split("\t"), f))

        _assert = set(map(len, dataset))
        if _assert != {3}:
            raise AssertionError(f"Expecting exactly 3 fields (lemma, form, tag/category), but found: {_assert}")

        def _raw2schema(line):
            return {
                "lemma": line[0],
                "form": line[1],
                "tag": line[2].split(";"),
            }

        dataset = list(map(_raw2schema, dataset))

        if self.config.schema == "source":
            for key, example in enumerate(dataset):
                yield key, example

        elif self.config.schema == "nusantara_pairs_multi":
            for key, ex in enumerate(dataset):
                yield key, {
                    "id": str(key),
                    "text_1": ex["lemma"],
                    "text_2": ex["form"],
                    "label": ex["tag"],
                }

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
