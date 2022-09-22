from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "term_a"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]
_LOCAL = False
_CITATION = """\
@article{winatmoko2019aspect,
  title={Aspect and opinion term extraction for hotel reviews using transfer learning and auxiliary labels},
  author={Winatmoko, Yosef Ardhito and Septiandri, Ali Akbar and Sutiono, Arie Pratama},
  journal={arXiv preprint arXiv:1909.11879},
  year={2019}
}
@inproceedings{fernando2019aspect,
  title={Aspect and opinion terms extraction using double embeddings and attention mechanism for indonesian hotel reviews},
  author={Fernando, Jordhy and Khodra, Masayu Leylia and Septiandri, Ali Akbar},
  booktitle={2019 International Conference of Advanced Informatics: Concepts, Theory and Applications (ICAICTA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
TermA is a span-extraction dataset collected from the hotel aggregator platform, AiryRooms
(Septiandri and Sutiono, 2019; Fernando et al.,
2019) consisting of thousands of hotel reviews,each containing a span label for aspect
and sentiment words representing the opinion of the reviewer on the corresponding aspect.
The labels use Inside-Outside-Beginning tagging (IOB) with two kinds of tags, aspect and
sentiment.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

_LICENSE = "Creative Common Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/train_preprocess.txt",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/valid_preprocess.txt",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/test_preprocess_masked_label.txt",
}

_SUPPORTED_TASKS = [Tasks.KEYWORD_TAGGING]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class BaPOSDataset(datasets.GeneratorBasedBuilder):
    """TermA is a span-extraction dataset containing 3k, 1k, 1k colloquial sentences in train, valid & test respectively of hotel domain with a total of 5 tags."""

    label_classes = ["B-ASPECT", "I-ASPECT", "B-SENTIMENT", "I-SENTIMENT", "O"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="term_a_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="TermA source schema",
            schema="source",
            subset_id="term_a",
        ),
        NusantaraConfig(
            name="term_a_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="TermA Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="term_a",
        ),
    ]

    DEFAULT_CONFIG_NAME = "term_a_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "token_tag": [datasets.Value("string")]})
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
                ex = {"index": str(i), "tokens": row["sentence"], "token_tag": row["label"]}
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {"id": str(i), "tokens": row["sentence"], "labels": row["label"]}
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
