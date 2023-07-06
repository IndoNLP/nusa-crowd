import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.nusa_datasets.id_short_answer_grading.utils.id_short_answer_grading_utils import \
    create_saintek_and_soshum_dataset
from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{
    JLK,
    author = {Muh Haidir and Ayu Purwarianti},
    title = { Short Answer Grading Using Contextual Word Embedding and Linear Regression},
    journal = {Jurnal Linguistik Komputasional},
    volume = {3},
    number = {2},
    year = {2020},
    keywords = {},
    abstract = {Abstract—One of the obstacles in an efficient MOOC is the evaluation of student answers, including the short answer grading which requires large effort from instructors to conduct it manually.
                Thus, NLP research in short answer grading has been conducted in order to support the automation, using several techniques such as rule
                and machine learning based. Here, we’ve conducted experiments on deep learning based short answer grading to compare the answer
                representation and answer assessment method. In the answer representation, we compared word embedding and sentence embedding models
                such as BERT, and its modification. In the answer assessment method, we use linear regression. There are 2 datasets that we used, available
                English short answer grading dataset with 80 questions and 2442 to get the best configuration for model and Indonesian short answer grading
                dataset with 36 questions and 9165 short answers as testing data. Here, we’ve collected Indonesian short answers for Biology and Geography
                subjects from 534 respondents where the answer grading was done by 7 experts. The best root mean squared error for both dataset was achieved
                by using BERT pretrained, 0.880 for English dataset dan 1.893 for Indonesian dataset.},
    issn = {2621-9336},	pages = {54--61},	doi = {10.26418/jlk.v3i2.38},
    url = {https://inacl.id/journal/index.php/jlk/article/view/38}
}\
"""
_DATASETNAME = "id_short_answer_grading"

_DESCRIPTION = """\
Indonesian short answers for Biology and Geography subjects from 534 respondents where the answer grading was done by 7 experts.\
"""

_HOMEPAGE = "https://github.com/AgeMagi/tugas-akhir"
_LOCAL = False
_LANGUAGES = ["ind"]

_LICENSE = "Unknown"

_URLS = {
    "saintek": {
        "train": {
            "question": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/question-saintek.csv",
            "score": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/score-saintek.csv",
        },
        "test": {
            "question": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/question-saintek-test.csv",
            "score": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/score-saintek-test.csv",
        },
    },
    "soshum": {
        "train": {
            "question": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/question-soshum.csv",
            "score": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/score-soshum.csv",
        },
        "test": {
            "question": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/question-soshum-test.csv",
            "score": "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data/score-soshum-test.csv",
        },
    },
}

_SUPPORTED_TASKS = [Tasks.SHORT_ANSWER_GRADING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IdShortAnswerGrading(datasets.GeneratorBasedBuilder):
    """Indonesian short answers for Biology and Geography subjects from 534 respondents where the answer grading was done by 7 experts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_short_answer_grading_source",
            version=SOURCE_VERSION,
            description="id_short_answer_grading source schema",
            schema="source",
            subset_id="id_short_answer_grading",
        ),
        NusantaraConfig(
            name="id_short_answer_grading_nusantara_pairs_score",
            version=NUSANTARA_VERSION,
            description="id_short_answer_grading Nusantara schema",
            schema="nusantara_pairs_score",
            subset_id="id_short_answer_grading",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_short_answer_grading_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("int64"),
                    "type-problem": datasets.Value("int64"),
                    "pertanyaan": datasets.Value("string"),
                    "kunci-jawaban": datasets.Value("string"),
                    "jawaban": datasets.Value("string"),
                    "score": datasets.Value("int64"),
                }
            )
        elif self.config.schema == "nusantara_pairs_score":
            features = schemas.pairs_features([0, 1, 2, 3, 4, 5])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        saintek_question = Path(dl_manager.download_and_extract(_URLS["saintek"]["train"]["question"]))
        saintek_score = Path(dl_manager.download_and_extract(_URLS["saintek"]["train"]["score"]))
        saintek_question_test = Path(dl_manager.download_and_extract(_URLS["saintek"]["test"]["question"]))
        saintek_score_test = Path(dl_manager.download_and_extract(_URLS["saintek"]["test"]["score"]))

        soshum_question = Path(dl_manager.download_and_extract(_URLS["soshum"]["train"]["question"]))
        soshum_score = Path(dl_manager.download_and_extract(_URLS["soshum"]["train"]["score"]))
        soshum_question_test = Path(dl_manager.download_and_extract(_URLS["soshum"]["test"]["question"]))
        soshum_score_test = Path(dl_manager.download_and_extract(_URLS["soshum"]["test"]["score"]))

        data_files = {
            "saintek_question": saintek_question,
            "saintek_score": saintek_score,
            "saintek_question_test": saintek_question_test,
            "saintek_score_test": saintek_score_test,
            "soshum_question": soshum_question,
            "soshum_score": soshum_score,
            "soshum_question_test": soshum_question_test,
            "soshum_score_test": soshum_score_test,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "saintek_question": os.path.join(data_files["saintek_question"]),
                    "soshum_question": os.path.join(data_files["soshum_question"]),
                    "saintek_score": os.path.join(data_files["saintek_score"]),
                    "soshum_score": os.path.join(data_files["soshum_score"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "saintek_question": os.path.join(data_files["saintek_question_test"]),
                    "soshum_question": os.path.join(data_files["soshum_question_test"]),
                    "saintek_score": os.path.join(data_files["saintek_score_test"]),
                    "soshum_score": os.path.join(data_files["soshum_score_test"]),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, saintek_question: Path, soshum_question: Path, saintek_score: Path, soshum_score: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = create_saintek_and_soshum_dataset(saintek_question, soshum_question, saintek_score, soshum_score)
        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {
                    "index": row.index,
                    "type-problem": row.type_problem,
                    "pertanyaan": row.pertanyaan,
                    "kunci-jawaban": row.kunci_jawaban,
                    "jawaban": row.jawaban,
                    "score": row.score,
                }
                yield row.index, entry

        elif self.config.schema == "nusantara_pairs_score":
            for row in df.itertuples():
                entry = {
                    "id": str(row.index),
                    "text_1": row.pertanyaan,
                    "text_2": row.jawaban,
                    "label": row.score,
                }
                yield row.index, entry
