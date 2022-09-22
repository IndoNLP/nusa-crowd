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
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_CITATION = """\
@INPROCEEDINGS{
7065828,
author={Trisedya, Bayu Distiawan and Inastra, Dyah},
booktitle={2014 International Conference on Advanced Computer Science and Information System},
title={Creating Indonesian-Javanese parallel corpora using wikipedia articles},
year={2014},
volume={},
number={},
pages={239-245},
doi={10.1109/ICACSIS.2014.7065828}}
"""

_DATASETNAME = "id_wiki_parallel"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "jav", "min", "sun"]
_LOCAL = False

_DESCRIPTION = """\
This dataset is designed for machine translation task, specifically jav->ind, min->ind, sun->ind, and vice versa. The data are taken
from sentences in Wikipedia.

(from the publication abstract)
Parallel corpora are necessary for multilingual researches especially in information retrieval (IR) and natural language processing (NLP). However, such corpora are hard to find, specifically for low-resources languages like ethnic
languages. Parallel corpora of ethnic languages were usually collected manually. On the other hand, Wikipedia as a free online encyclopedia is supporting more and more languages each year, including ethnic languages in Indonesia. It has
become one of the largest multilingual sites in World Wide Web that provides free distributed articles. In this paper, we explore a few sentence alignment methods which have been used before for another domain. We want to check whether
Wikipedia can be used as one of the resources for collecting parallel corpora of Indonesian and Javanese, an ethnic language in Indonesia. We used two approaches of sentence alignment by treating Wikipedia as both parallel corpora and
comparable corpora. In parallel corpora case, we used sentence length based and word correspondence methods. Meanwhile,
we used the characteristics of hypertext links from Wikipedia in comparable corpora case. After the experiments, we can
see that Wikipedia is useful enough for our purpose because both approaches gave positive results.
"""

_HOMEPAGE = "https://github.com/dindainastra/indowikiparalelcorpora"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: {
        "jav": "https://raw.githubusercontent.com/dindainastra/indowikiparalelcorpora/main/manualsets/indojv-parallel.csv",
        "min": "https://raw.githubusercontent.com/dindainastra/indowikiparalelcorpora/main/manualsets/indomin-parallel.csv",
        "sun": "https://raw.githubusercontent.com/dindainastra/indowikiparalelcorpora/main/manualsets/indosun-parallel.csv",
    }
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdWikiParallel(datasets.GeneratorBasedBuilder):
    """
    This dataset is designed for machine translation task, specifically jav->ind, min->ind, sun->ind, and vice versa. The data are
    taken from sentences in Wikipedia."""

    ETHNIC_LANGUAGES = [lang for lang in _LANGUAGES if lang != "ind"]
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="{dataset_name}_{src}_ind_source".format(dataset_name=_DATASETNAME, src=src),
            version=datasets.Version(_SOURCE_VERSION),
            description="ID Wiki Parallel source schema for {src} to ind and ind to {src}".format(src=src),
            schema="source",
            subset_id="{dataset_name}_{src}_ind".format(dataset_name=_DATASETNAME, src=src),
        )
        for src in ETHNIC_LANGUAGES
    ] + [
        NusantaraConfig(
            name="{dataset_name}_{src}_ind_nusantara_t2t".format(dataset_name=_DATASETNAME, src=src),
            version=datasets.Version(_NUSANTARA_VERSION),
            description="ID Wiki Parallel  Nusantara schema for {src} to ind and ind to {src}".format(src=src),
            schema="nusantara_t2t",
            subset_id="{dataset_name}_{src}_ind".format(dataset_name=_DATASETNAME, src=src),
        )
        for src in ETHNIC_LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = "id_wiki_parallel_jav_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text_1": datasets.Value("string"), "text_2": datasets.Value("string")})
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        split_config_name = self.config.name.split("_")
        src = split_config_name[3]
        data_file = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME][src]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_file,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        split_config_name = self.config.name.split("_")
        src = split_config_name[3]
        df = pd.read_csv(filepath, encoding="utf8").reset_index()

        for id, row in df.iterrows():
            src_txt = row[1]
            tgt_txt = row[2]
            if self.config.schema == "source":
                yield id, {"id": str(id), "text_1": src_txt, "text_2": tgt_txt}
            elif self.config.schema == "nusantara_t2t":
                yield id, {"id": str(id), "text_1": src_txt, "text_2": tgt_txt, "text_1_name": src, "text_2_name": "ind"}
