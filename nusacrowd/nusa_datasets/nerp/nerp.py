from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "nerp"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]
_LOCAL = False
_CITATION = """\
@inproceedings{hoesen2018investigating,
  title={Investigating bi-lstm and crf with pos tag embedding for indonesian named entity tagger},
  author={Hoesen, Devin and Purwarianti, Ayu},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  pages={35--38},
  year={2018},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
The NERP dataset (Hoesen and Purwarianti, 2018) contains texts collected from several Indonesian news websites with five labels
- PER (name of person)
- LOC (name of location)
- IND (name of product or brand)
- EVT (name of the event)
- FNB (name of food and beverage).
NERP makes use of the IOB chunking format, just like the TermA dataset.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nerp_ner-prosa/train_preprocess.txt",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nerp_ner-prosa/valid_preprocess.txt",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nerp_ner-prosa/test_preprocess_masked_label.txt",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class NerpDataset(datasets.GeneratorBasedBuilder):
    """NERP is an NER tagging dataset contains about (train=6720,valid=840,test=840) sentences, with 11 classes."""

    label_classes = ["B-PPL", "B-PLC", "B-EVT", "B-IND", "B-FNB", "I-PPL", "I-PLC", "I-EVT", "I-IND", "I-FNB", "O"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="nerp_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="NERP source schema",
            schema="source",
            subset_id="nerp",
        ),
        NusantaraConfig(
            name="nerp_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="NERP Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="nerp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "nerp_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "ner_tag": [datasets.Value("string")]})
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
                ex = {"index": str(i), "tokens": row["sentence"], "ner_tag": row["label"]}
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {"id": str(i), "tokens": row["sentence"], "labels": row["label"]}
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
