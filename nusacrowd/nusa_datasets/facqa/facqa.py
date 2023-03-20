import ast
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.nusa_datasets.facqa.utils.facqa_utils import (getAnswerString, listToString)
from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """
@inproceedings{purwarianti2007machine,
  title={A Machine Learning Approach for Indonesian Question Answering System},
  author={Ayu Purwarianti, Masatoshi Tsuchiya, and Seiichi Nakagawa},
  booktitle={Proceedings of Artificial Intelligence and Applications },
  pages={573--578},
  year={2007}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "facqa"

_DESCRIPTION = """
FacQA: The goal of the FacQA dataset is to find the answer to a question from a provided short passage from a news article.
Each row in the FacQA dataset consists of a question, a short passage, and a label phrase, which can be found inside the
corresponding short passage. There are six categories of questions: date, location, name,
organization, person, and quantitative.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    _DATASETNAME: {
        "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/facqa_qa-factoid-itb/test_preprocess.csv",
        "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/facqa_qa-factoid-itb/train_preprocess.csv",
        "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/facqa_qa-factoid-itb/valid_preprocess.csv",
    }
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class FacqaDataset(datasets.GeneratorBasedBuilder):
    """FacQA dataset is a labeled dataset for indonesian question answering task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="facqa_source",
            version=SOURCE_VERSION,
            description="FacQA source schema",
            schema="source",
            subset_id="facqa",
        ),
        NusantaraConfig(
            name="facqa_nusantara_qa",
            version=NUSANTARA_VERSION,
            description="FacQA Nusantara schema",
            schema="nusantara_qa",
            subset_id="facqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "facqa_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("int64"),
                    "question": [datasets.Value("string")],
                    "passage": [datasets.Value("string")],
                    "seq_label": [datasets.Value("string")],
                }
            )
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
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        train_csv_path = Path(dl_manager.download_and_extract(urls["train"]))
        validation_csv_path = Path(dl_manager.download_and_extract(urls["validation"]))
        test_csv_path = Path(dl_manager.download_and_extract(urls["test"]))
        data_files = {
            "train": train_csv_path,
            "validation": validation_csv_path,
            "test": test_csv_path,
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()
        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {"index": row.index, "question": ast.literal_eval(row.question), "passage": ast.literal_eval(row.passage), "seq_label": ast.literal_eval(row.seq_label)}
                yield row.index, entry

        elif self.config.schema == "nusantara_qa":
            for row in df.itertuples():
                entry = {
                    "id": str(row.index),
                    "question_id": str(row.index),
                    "document_id": str(row.index),
                    "question": listToString(ast.literal_eval(row.question)),
                    "type": "extractive",
                    "choices": [],
                    "context": listToString(ast.literal_eval(row.passage)),
                    "answer": [getAnswerString(ast.literal_eval(row.passage), ast.literal_eval(row.seq_label))],
                }
                yield row.index, entry
