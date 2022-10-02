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

from itertools import chain, repeat
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_ud_data, load_ud_data_as_nusantara_kb
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@conference{2f8c7438a7f44f6b85b773586cff54e8,
    title = "A gold standard dependency treebank for Indonesian",
    author = "Ika Alfina and Arawinda Dinakaramani and Fanany, {Mohamad Ivan} and Heru Suhartanto",
    note = "Publisher Copyright: {\textcopyright} 2019 Proceedings of the 33rd Pacific Asia Conference on Language, Information and Computation, PACLIC 2019. All rights reserved.; \
33rd Pacific Asia Conference on Language, Information and Computation, PACLIC 2019 ; Conference date: 13-09-2019 Through 15-09-2019",
    year = "2019",
    month = jan,
    day = "1",
    language = "English",
    pages = "1--9",
}

@article{DBLP:journals/corr/abs-2011-00677,
    author    = {Fajri Koto and
                 Afshin Rahimi and
                 Jey Han Lau and
                 Timothy Baldwin},
    title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
                 Model for Indonesian {NLP}},
    journal   = {CoRR},
    volume    = {abs/2011.00677},
    year      = {2020},
    url       = {https://arxiv.org/abs/2011.00677},
    eprinttype = {arXiv},
    eprint    = {2011.00677},
    timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "indolem_ud_id_pud"

_DESCRIPTION = """\
1 of 8 sub-datasets of IndoLEM, a comprehensive dataset encompassing 7 NLP tasks (Koto et al., 2020).
This dataset is part of [Parallel Universal Dependencies (PUD)](http://universaldependencies.org/conll17/) project.
This is based on the first corrected version by Alfina et al. (2019), contains 1,000 sentences.
"""

_HOMEPAGE = "https://indolem.github.io/"

_LICENSE = "Creative Commons Attribution 4.0"

_FOLDS = list(range(5))
_DEFAULT_FOLD = _FOLDS[0]
_URLS = {
    f"{_DATASETNAME}_{fold}": {
        "train": f"https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_PUD/folds/train{fold}.conllu",
        "validation": f"https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_PUD/folds/dev{fold}.conllu",
        "test": f"https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_PUD/folds/test{fold}.conllu",
    }
    for fold in _FOLDS
}

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IndolemUDIDPUDDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = list(
        chain(
            (
                NusantaraConfig(
                    name=f"{_DATASETNAME}_source",
                    version=SOURCE_VERSION,
                    description=f"{_DATASETNAME} default fold ('{_DEFAULT_FOLD}') of source schema",
                    schema="source",
                    subset_id=f"{_DATASETNAME}_{_DEFAULT_FOLD}",
                ),
                NusantaraConfig(
                    name=f"{_DATASETNAME}_nusantara_kb",
                    version=NUSANTARA_VERSION,
                    description=f"{_DATASETNAME} default fold ('{_DEFAULT_FOLD}') of Nusantara KB schema",
                    schema="nusantara_kb",
                    subset_id=f"{_DATASETNAME}_{_DEFAULT_FOLD}",
                ),
            ),
            *(
                (
                    NusantaraConfig(
                        name=f"{_DATASETNAME}_{fold}_source",
                        version=ver_src,
                        description=f"{_DATASETNAME} fold '{fold}' of source schema",
                        schema="source",
                        subset_id=f"{_DATASETNAME}_{fold}",
                    ),
                    NusantaraConfig(
                        name=f"{_DATASETNAME}_{fold}_nusantara_kb",
                        version=ver_nusa,
                        description=f"{_DATASETNAME} fold '{fold}' of Nusantara KB schema",
                        schema="nusantara_kb",
                        subset_id=f"{_DATASETNAME}_{fold}",
                    ),
                )
                for fold, ver_src, ver_nusa in zip(_FOLDS, repeat(SOURCE_VERSION), repeat(NUSANTARA_VERSION))
            ),
        )
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_en": datasets.Value("string"),
                    # tokens
                    "id": [datasets.Value("string")],
                    "form": [datasets.Value("string")],
                    "lemma": [datasets.Value("string")],
                    "upos": [datasets.Value("string")],  # Alternatively, use ClassLabel (https://huggingface.co/datasets/universal_dependencies/blob/main/universal_dependencies.py#L1211)
                    "xpos": [datasets.Value("string")],
                    "feats": [datasets.Value("string")],
                    "head": [datasets.Value("string")],
                    "deprel": [datasets.Value("string")],
                    "deps": [datasets.Value("string")],
                    "misc": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "nusantara_kb":
            features = schemas.kb_features

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
        urls = _URLS[self.config.subset_id]
        data_dir = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

        try:
            generator_fn = {
                "source": load_ud_data,
                "nusantara_kb": load_ud_data_as_nusantara_kb,
            }[self.config.schema]
        except KeyError:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        for key, example in enumerate(generator_fn(filepath)):
            yield key, example


# if __name__ == "__main__":
#     datasets.load_dataset(__file__, name=f"indolem_ud_id_pud_source")
#     datasets.load_dataset(__file__, name=f"indolem_ud_id_pud_nusantara_kb")
#     for fold in _FOLDS:
#         datasets.load_dataset(__file__, name=f"indolem_ud_id_pud_{fold}_source")
#         datasets.load_dataset(__file__, name=f"indolem_ud_id_pud_{fold}_nusantara_kb")
