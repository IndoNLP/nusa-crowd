import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas
import json

_CITATION = """\
@misc{quoraFirstQuora,
	author = {},
	title = {{F}irst {Q}uora {D}ataset {R}elease: {Q}uestion {P}airs --- quoradata.quora.com},
	howpublished = {https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs},
	year = 2017,
	note = {Online},
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "id_qqp"

_DESCRIPTION = """\
Quora Question Pairs (QQP) dataset consists of over 400,000 question pairs, 
and each question pair is annotated with a binary value indicating whether 
the two questions are paraphrase of each other. This dataset is translated 
version of QQP to Indonesian Language.
"""

_HOMEPAGE = "https://github.com/louisowen6/quora_paraphrasing_id"

_LICENSE = "Apache License, Version 2.0"

_URLS = {
    _DATASETNAME: [
        "https://github.com/louisowen6/quora_paraphrasing_id/raw/main/ID_Quora_Paraphrasing_train.json",
        "https://github.com/louisowen6/quora_paraphrasing_id/raw/main/ID_Quora_Paraphrasing_val.json",
    ]
}

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IdQuoraQuestionPairs(datasets.GeneratorBasedBuilder):
    """
    Quora Question Pairs (QQP) dataset consists of over 400,000 question pairs, 
    and each question pair is annotated with a binary value indicating whether 
    the two questions are paraphrase of each other. This dataset is translated 
    version of QQP to Indonesian Language.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_qqp_source",
            version=SOURCE_VERSION,
            description="ID QQP source schema",
            schema="source",
            subset_id="id_qqp",
        ),
        NusantaraConfig(
            name="id_qqp_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="ID QQP Nusantara schema",
            schema="nusantara_t2t",
            subset_id="id_qqp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_qqp_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
               {
                   "id": datasets.Value("string"),
                   "question_1": datasets.Value("string"),
                   "question_2": datasets.Value("string")
               }
            )

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
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir[0],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir[1],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        
        with open(filepath, "r") as f:
            lines = f.readlines()

        if self.config.schema == "source":
            
            for i, line in enumerate(lines):
                line = json.loads(line.strip())
                
                sample = {
                    "id": str(i),
                    "question_1": line["question_1"],
                    "question_2": line["question_2"]
                }
                yield i, sample

        elif self.config.schema == "nusantara_t2t":
            
            for i, line in enumerate(lines):
                line = json.loads(line.strip())
                
                sample = {
                    "id": str(i),
                    "text_1": line["question_1"],
                    "text_2": line["question_2"],
                    "text_1_name": "question_1",
                    "text_2_name": "question_2"
                }
                yield i, sample
