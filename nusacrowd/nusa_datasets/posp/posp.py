from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{hoesen2018investigating,
  title={Investigating Bi-LSTM and CRF with POS Tag Embedding for Indonesian Named Entity Tagger},
  author={Devin Hoesen and Ayu Purwarianti},
  booktitle={Proceedings of the 2018 International Conference on Asian Language Processing (IALP)},
  pages={35--38},
  year={2018},
  organization={IEEE}
}

@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "posp"

_DESCRIPTION = """\
POSP is a POS Tagging dataset containing 8400 sentences, collected from Indonesian news website with 26 POS tag classes.
The POS tag labels follow the Indonesian Association of Computational Linguistics (INACL) POS Tagging Convention.
POSP dataset is splitted into 3 sets with 6720 train, 840 validation, and 840 test data.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/posp_pos-prosa/train_preprocess.txt",
        "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/posp_pos-prosa/valid_preprocess.txt",
        "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/posp_pos-prosa/test_preprocess.txt",
    }
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class POSPDataset(datasets.GeneratorBasedBuilder):
    """POSP is a POS Tagging dataset containing 8400 sentences, collected from Indonesian news website with 26 POS tag classes."""

    label_classes = [
        "B-PPO",
        "B-KUA",
        "B-ADV",
        "B-PRN",
        "B-VBI",
        "B-PAR",
        "B-VBP",
        "B-NNP",
        "B-UNS",
        "B-VBT",
        "B-VBL",
        "B-NNO",
        "B-ADJ",
        "B-PRR",
        "B-PRK",
        "B-CCN",
        "B-$$$",
        "B-ADK",
        "B-ART",
        "B-CSN",
        "B-NUM",
        "B-SYM",
        "B-INT",
        "B-NEG",
        "B-PRI",
        "B-VBE",
    ]

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="posp_source",
            version=SOURCE_VERSION,
            description="POSP source schema",
            schema="source",
            subset_id="posp",
        ),
        NusantaraConfig(
            name="posp_nusantara_seq_label",
            version=NUSANTARA_VERSION,
            description="POSP Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="posp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "posp_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "pos_tags": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.label_classes)

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
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        conll_dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "index": str(i),
                    "tokens": row["sentence"],
                    "pos_tags": row["label"],
                }
                yield i, ex

        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "id": str(i),
                    "tokens": row["sentence"],
                    "labels": row["label"],
                }
                yield i, ex
