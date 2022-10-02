from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "keps"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]
_LOCAL = False
_CITATION = """\
@inproceedings{mahfuzh2019improving,
  title={Improving Joint Layer RNN based Keyphrase Extraction by Using Syntactical Features},
  author={Miftahul Mahfuzh, Sidik Soleman, and Ayu Purwarianti},
  booktitle={Proceedings of the 2019 International Conference of Advanced Informatics: Concepts, Theory and Applications (ICAICTA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
The KEPS dataset (Mahfuzh, Soleman and Purwarianti, 2019) consists of text from Twitter
discussing banking products and services and is written in the Indonesian language. A phrase
containing important information is considered a keyphrase. Text may contain one or more
keyphrases since important phrases can be located at different positions.
- tokens: a list of string features.
- seq_label: a list of classification labels, with possible values including O, B, I.
The labels use Inside-Outside-Beginning (IOB) tagging.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/keps_keyword-extraction-prosa/train_preprocess.txt",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/keps_keyword-extraction-prosa/valid_preprocess.txt",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/keps_keyword-extraction-prosa/test_preprocess.txt",
}

_SUPPORTED_TASKS = [Tasks.KEYWORD_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class KepsDataset(datasets.GeneratorBasedBuilder):
    """KEPS is an keyphrase extraction dataset contains about (train=800,valid=200,test=247) sentences, with 3 classes."""

    label_classes = ["B", "I", "O"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="keps_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="KEPS source schema",
            schema="source",
            subset_id="keps",
        ),
        NusantaraConfig(
            name="keps_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="KEPS Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="keps",
        ),
    ]

    DEFAULT_CONFIG_NAME = "keps_source"

    def _info(self):
        print(datasets)
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "ke_tag": [datasets.Value("string")]})
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
        conll_dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for i, row in enumerate(conll_dataset):
                ex = {"index": str(i), "tokens": row["sentence"], "ke_tag": row["label"]}
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {"id": str(i), "tokens": row["sentence"], "labels": row["label"]}
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")