import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas
import jsonlines
from nltk.tokenize.treebank import TreebankWordDetokenizer

_CITATION = """\
@inproceedings{hasan2021xl,
  title={XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages},
  author={Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md Saiful and Mubasshir, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and Rahman, M Sohel and Shahriyar, Rifat},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={4693--4703},
  year={2021}
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "xl_sum"

_DESCRIPTION = """\
XL-Sum is a large-scale multilingual summarization dataset that covers 45 languages including Indonesian text summarization.
The dataset is based on article-summary pairs from BBC, is highly abstractive, concise, and of high quality, as indicated by human and intrinsic evaluation.
"""

_HOMEPAGE = "https://github.com/csebuetnlp/xl-sum"

_LICENSE = "CC-BY-NC-SA 4.0"

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/indonesian_XLSum_v2.0.tar.bz2",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "2.0.0"

_NUSANTARA_VERSION = "1.0.0"

class XLSum(datasets.GeneratorBasedBuilder):
    """XL-Sum is a large-scale multilingual summarization dataset that covers 45 languages including Indonesian text summarization. The dataset is based on article-summary pairs from BBC, is highly abstractive, concise, and of high quality, as indicated by human and intrinsic evaluation."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="xl_sum_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="xl_sum source schema",
            schema="source",
            subset_id="xl_sum",
        ),
        NusantaraConfig(
            name="xl_sum_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="xl_sum Nusantara schema",
            schema="nusantara_t2t",
            subset_id="xl_sum",
        ),
    ]

    DEFAULT_CONFIG_NAME = "xl_sum_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                    "id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "summary": datasets.Value("string")
               }
            )
        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        data_files = {
            "train": "indonesian_train.jsonl",
            "validation": "indonesian_val.jsonl",
            "test": "indonesian_test.jsonl",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["validation"]),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_files["test"]),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {
                        "id": each_data["id"],
                        "url": each_data["url"],
                        "title": each_data["title"],
                        "text": each_data["text"],
                        "summary": each_data["summary"],
                    }
                    yield each_data["id"], ex

        elif self.config.schema == "nusantara_t2t":
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {
                        "id": each_data["id"],
                        "text_1": each_data["text"],
                        "text_2": each_data["summary"],
                        "text_1_name": each_data["title"],
                        "text_2_name": "summary"
                    }
                    yield each_data["id"], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
