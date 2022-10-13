import json
from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "squad_id"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{muis2020sequence,
  title={Sequence-to-sequence learning for indonesian automatic question generator},
  author={Muis, Ferdiant Joshua and Purwarianti, Ayu},
  booktitle={2020 7th International Conference on Advance Informatics: Concepts, Theory and Applications (ICAICTA)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
    This dataset contains Indonesian SQuAD v2.0 dataset (Google-translated).
    The dataset can be used for automatic question generation (AQG),
    or machine reading comphrehension(MRC) task.
"""

_HOMEPAGE = "https://github.com/FerdiantJoshua/question-generator"

_LICENSE = "TBD"

_URLs = {"train": "https://drive.google.com/uc?id=1LP0iB0Xe6nkbnSxMeclxexUfqCE9e5qH&export=download", "val": "https://drive.google.com/uc?id=1KZE92j3Cnf7N6o0qrVplBqXV2XlGxnvo&export=download"}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class SQuADIdDataset(datasets.GeneratorBasedBuilder):
    """SQuADID dataset contains the Indonisian SQuAD 2.0 data (translated by google)."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="squad_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="SQUAD_ID source schema",
            schema="source",
            subset_id="squad_id",
        ),
        NusantaraConfig(
            name="squad_id_nusantara_qa",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="SQUAD_ID Nusantara schema",
            schema="nusantara_qa",
            subset_id="squad_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "squad_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "context": datasets.Value("string"), "question": datasets.Value("string"), "answer": datasets.Sequence(datasets.Value("string"))})
        elif self.config.schema == "nusantara_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        val_path = Path(dl_manager.download_and_extract(_URLs["val"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": val_path},
            ),
        ]

    def _generate_examples(self, filepath: Path):

        count = 0
        if self.config.schema == "nusantara_qa" or self.config.schema == "source":
            with open(filepath, "r") as f:
                data = json.load(f)
                paragraphs = data["paragraphs"]
                for k, v in paragraphs.items():
                    for each_data in v:
                        qas_list = each_data["qas"]
                        for each_qa in qas_list:
                            if "indonesian_plausible_answers" in each_qa.keys():
                                answers = each_qa["indonesian_plausible_answers"]
                            elif "indonesian_answers" in each_qa.keys():
                                answers = each_qa["indonesian_answers"]
                            if self.config.schema == "nusantara_qa":
                                yield count, {
                                    "id": each_qa["id"],
                                    "question_id": each_qa["id"],
                                    "document_id": k,
                                    "question": each_qa["question"],
                                    "type": "extractive",
                                    "choices": [],
                                    "context": each_data["context"],
                                    "answer": answers,
                                }

                            else:
                                yield count, {
                                    "id": each_qa["id"],
                                    "context": each_data["context"],
                                    "question": each_qa["question"],
                                    "answer": answers,
                                }
                            count += 1
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
