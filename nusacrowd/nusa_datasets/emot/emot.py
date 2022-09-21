from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "emot"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 langauge code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{saputri2018emotion,
  title={Emotion classification on indonesian twitter dataset},
  author={Saputri, Mei Silviana and Mahendra, Rahmad and Adriani, Mirna},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  pages={90--95},
  year={2018},
  organization={IEEE}
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
EmoT is an emotion classification dataset collected from the social media platform Twitter. The dataset consists of around 4000 Indonesian colloquial language tweets, covering five different emotion labels: anger, fear, happiness, love, and sadness.
EmoT dataset is splitted into 3 sets with 3521 train, 440 validation, 442 test data.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/valid_preprocess.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/test_preprocess.csv",
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class EmoT(datasets.GeneratorBasedBuilder):
    """SMSA is a sentiment analysis dataset consisting of 3 labels (positive, neutral, and negative) which comes from comments and reviews collected from multiple online platforms."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="emot_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="EmoT source schema",
            schema="source",
            subset_id="emot",
        ),
        NusantaraConfig(
            name="emot_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="EmoT Nusantara schema",
            schema="nusantara_text",
            subset_id="emot",
        ),
    ]

    DEFAULT_CONFIG_NAME = "emot_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tweet": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["happy", "love", "fear", "anger", "sadness"])

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
        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()
        df.columns = ["id", "label", "tweet"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": str(row.id), "tweet": row.tweet, "label": row.label}
                yield row.id, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {"id": str(row.id), "text": row.tweet, "label": row.label}
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
