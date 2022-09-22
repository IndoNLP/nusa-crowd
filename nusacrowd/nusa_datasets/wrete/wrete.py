from pathlib import Path
from sys import stdout
from tempfile import tempdir
from typing import List
from unicodedata import category

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "wrete"
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
WReTe, The Wiki Revision Edits Textual Entailment dataset (Setya and Mahendra, 2018) consists of 450 sentence pairs constructed from Wikipedia revision history. The dataset contains pairs of sentences and binary semantic relations between the pairs. The data are labeled as entailed when the meaning of the second sentence can be derived from the first one, and not entailed otherwise
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://github.com/IndoNLP/indonlu/raw/master/dataset/wrete_entailment-ui/train_preprocess.csv",
    "validation": "https://github.com/IndoNLP/indonlu/raw/master/dataset/wrete_entailment-ui/valid_preprocess.csv",
    "test": "https://github.com/IndoNLP/indonlu/raw/master/dataset/wrete_entailment-ui/test_preprocess_masked_label.csv",
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class WReTe(datasets.GeneratorBasedBuilder):
    """WReTe consists of premise, hypothesis, category, and label. The Data are labeled as entailed when the meaning of the second sentence can be derived from the first one, and not entailed otherwise"""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="wrete_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="WReTe source schema",
            schema="source",
            subset_id="wrete",
        ),
        NusantaraConfig(
            name="wrete_nusantara_pairs",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="WReTe Nusantara schema",
            schema="nusantara_pairs",
            subset_id="wrete",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wrete_source"
    labels = ["NotEntail", "Entail_or_Paraphrase"]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("int32"), 
                                        "sent_A": datasets.Value("string"), 
                                        "sent_B": datasets.Value("string"),
                                        "category" : datasets.Value("string"),
                                        "label": datasets.Value("string")})
        elif self.config.schema == "nusantara_pairs":
            features = schemas.pairs_features(self.labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_csv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        validation_csv_path = Path(dl_manager.download_and_extract(_URLs["validation"]))
        test_csv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "train": train_csv_path,
            "validation": validation_csv_path,
            "test": test_csv_path,
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
        df = pd.read_csv(filepath, sep=",", quotechar='"').reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": row.index,
                     "sent_A" : row.sent_A,
                     "sent_B" : row.sent_B,
                     "category" : row.category, 
                     "label" : row.label
                }
                yield row.index, ex
        elif self.config.schema == "nusantara_pairs":
            for row in df.itertuples():
                ex = {
                    "id": row.index,
                    "text_1": row.sent_A,
                    "text_2": row.sent_B,
                    "label": row.label
                }
                yield row.index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
