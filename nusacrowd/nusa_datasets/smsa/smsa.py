from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "smsa"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_CITATION = """\
@INPROCEEDINGS{8904199,
    author={Purwarianti, Ayu and Crisdayanti, Ida Ayu Putu Ari},
    booktitle={2019 International Conference of Advanced Informatics: Concepts, Theory and Applications (ICAICTA)},
    title={Improving Bi-LSTM Performance for Indonesian Sentiment Analysis Using Paragraph Vector},
    year={2019},
    pages={1-5},
    doi={10.1109/ICAICTA.2019.8904199}
}

@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Wilie, Bryan and Vincentio, Karissa and Winata, Genta Indra and Cahyawijaya, Samuel and Li, Xiaohong and Lim, Zhi Yuan and Soleman, Sidik and Mahendra, Rahmad and Fung, Pascale and Bahar, Syafri and others},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  pages={843--857},
  year={2020}
}
"""

_DESCRIPTION = """\
SmSA is a sentence-level sentiment analysis dataset (Purwarianti and Crisdayanti, 2019) is a collection of comments and reviews
in Indonesian obtained from multiple online platforms. The text was crawled and then annotated by several Indonesian linguists
to construct this dataset. There are three possible sentiments on the SmSA dataset: positive, negative, and neutral
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://github.com/IndoNLP/indonlu/raw/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv",
    "validation": "https://github.com/IndoNLP/indonlu/raw/master/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv",
    "test": "https://github.com/IndoNLP/indonlu/raw/master/dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class SMSA(datasets.GeneratorBasedBuilder):
    """SMSA is a sentiment analysis dataset consisting of 3 labels (positive, neutral, and negative) which comes from comments and reviews collected from multiple online platforms."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="smsa_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="SMSA source schema",
            schema="source",
            subset_id="smsa",
        ),
        NusantaraConfig(
            name="smsa_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="SMSA Nusantara schema",
            schema="nusantara_text",
            subset_id="smsa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "smsa_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "sentence": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["negative", "neutral", "positive"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        validation_tsv_path = Path(dl_manager.download_and_extract(_URLs["validation"]))
        test_tsv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "train": train_tsv_path,
            "validation": validation_tsv_path,
            "test": test_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_csv(filepath, sep="\t", header=None).reset_index()
        df.columns = ["id", "sentence", "label"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": str(row.id), "sentence": row.sentence, "label": row.label}
                yield row.id, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.sentence,
                    "label": row.label
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
