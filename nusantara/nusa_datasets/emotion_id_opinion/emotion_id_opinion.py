from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "emotion_id_opinion"
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
Emotion ID Opinion is a dataset of Indonesian-language tweets conveying public opinion on a variety of topics.
It comtains 7080 indunesian tweets and a person's emotion response towards each tweet.
The data is annotated with six emotional labels, namely anger, fear, joy, love, sad, and neutral.
"""

_HOMEPAGE = "https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "anger": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/AngerData.csv",
    "fear": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/FearData.csv",
    "joy": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/JoyData.csv",
    "love": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/LoveData.csv",
    "sad": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/SadData.csv",
    "neutral": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/NeutralData.csv"
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class EmoIdOpinion(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="emotion_id_opinion_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="EmoIdOpinion source schema",
            schema="source",
            subset_id="emotion_id_opinion",
        ),
        NusantaraConfig(
            name="emotion_id_opinion_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="EmoIdOpinion Nusantara schema",
            schema="nusantara_text",
            subset_id="emotion_id_opinion",
        ),
    ]

    DEFAULT_CONFIG_NAME = "emotion_id_opinion_source"

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
            features = schemas.text_features(["Joy", "Love", "Fear", "Anger", "Sad", "Neutral"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        anger_tsv_path = Path(dl_manager.download_and_extract(_URLs["anger"]))
        fear_tsv_path = Path(dl_manager.download_and_extract(_URLs["fear"]))
        joy_tsv_path = Path(dl_manager.download_and_extract(_URLs["joy"]))
        love_tsv_path = Path(dl_manager.download_and_extract(_URLs["love"]))
        neutral_tsv_path = Path(dl_manager.download_and_extract(_URLs["neutral"]))
        sad_tsv_path = Path(dl_manager.download_and_extract(_URLs["sad"]))

        data_files = {
            "anger": anger_tsv_path,
            "fear": fear_tsv_path,
            "joy": joy_tsv_path,
            "love": love_tsv_path,
            "neutral": neutral_tsv_path,
            "sad": sad_tsv_path
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": [
                    data_files["anger"],
                    data_files["fear"],
                    data_files["joy"],
                    data_files["love"],
                    data_files["neutral"],
                    data_files["sad"]
                ]},
            ),
        ]

    def _generate_examples(self, filepath: List[Path]):
        increment = 0
        for i, fp in enumerate(filepath):
        # df = pd.concat([pd.read_csv(fp, sep="\t", header="infer").reset_index() for fp in filepath])
            df = pd.read_csv(fp, sep="\t", header="infer").reset_index()
            df.columns = ["id", "Tweet", "Label"]

            if self.config.schema == "source":
                for row in df.itertuples():
                    ex = {"index": str(increment + row.id), "tweet": row.Tweet, "label": row.Label}
                    yield increment + row.id, ex
            elif self.config.schema == "nusantara_text":
                for row in df.itertuples():
                    ex = {"id": str(increment + row.id), "text": row.Tweet, "label": row.Label}
                    yield increment + row.id, ex
            else:
                raise ValueError(f"Invalid config: {self.config.name}")

            increment += row.id + 1
