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
@INPROCEEDINGS{8629109,
  author={Kurniawan, Kemal and Louvan, Samuel},
  booktitle={2018 International Conference on Asian Language Processing (IALP)}, 
  title={Indosum: A New Benchmark Dataset for Indonesian Text Summarization}, 
  year={2018},
  volume={},
  number={},
  pages={215-220},
  doi={10.1109/IALP.2018.8629109}}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indosum"

_DESCRIPTION = """\
INDOSUM is a new benchmark dataset for Indonesian text summarization. 
The dataset consists of news articles and manually constructed summaries.
"""

_HOMEPAGE = "https://github.com/kata-ai/indosum"

_LICENSE = "Apache License, Version 2.0"

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IndoSUM(datasets.GeneratorBasedBuilder):
    """INDOSUM is a new benchmark dataset for Indonesian text summarization. The dataset consists of news articles and manually constructed summaries."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = (
        [
            NusantaraConfig(
                name="indosum_fold{fold_number}_source".format(fold_number=i),
                version=_SOURCE_VERSION,
                description="indosum source schema",
                schema="source",
                subset_id="indosum_fold{fold_number}".format(fold_number=i),
            ) for i in range(5)
        ]
        +
        [
            NusantaraConfig(
            name="indosum_fold{fold_number}_nusantara_t2t".format(fold_number=i),
            version=_NUSANTARA_VERSION,
            description="indosum Nusantara schema",
            schema="nusantara_t2t",
            subset_id="indosum_fold{fold_number}".format(fold_number=i),
        ) for i in range(5)
        ]
    )

    DEFAULT_CONFIG_NAME = "indosum_fold0_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
               {
                   "document": datasets.Value("string"),
                   "id": datasets.Value("string"),
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

    def _get_fold_index(self):
        try:
            subset_id = self.config.subset_id
            idx_fold = subset_id.index("_fold")
            file_id = subset_id[(idx_fold + 5):]
            return int(file_id)
        except:
            return 0

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        idx = self._get_fold_index()

        urls = _URLS[_DATASETNAME]

        data_dir = Path(dl_manager.download_and_extract(urls))

        location = {
            "train": "indosum/train.0{fold_number}.jsonl",
            "test": "indosum/test.0{fold_number}.jsonl",
            "dev": "indosum/dev.0{fold_number}.jsonl"
        }

        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["train"].format(fold_number=idx+1)),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["test"].format(fold_number=idx+1)),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["dev"].format(fold_number=idx+1)),
                    "split": "dev",
                },
            ),
        ]

    def _get_full_paragraph_and_summary(self, data: Dict) -> Tuple[str, str]:
        detokenizer = TreebankWordDetokenizer()
        paragraph = ""
        summary = ""
        begin_paragraph = True
        begin_summary = True

        for each_paragraph in data["paragraphs"]:
            for each_sentence in each_paragraph:
                detokenized_sentence = detokenizer.detokenize(each_sentence)
                if begin_paragraph:
                    paragraph+=detokenized_sentence
                    begin_paragraph = False
                else:
                    paragraph = "{} {}".format(paragraph, detokenized_sentence)
        
        for each_summary in data["summary"]:
            detokenized_sentence = detokenizer.detokenize(each_summary)
            if begin_summary:
                summary+=detokenized_sentence
                begin_summary = False
            else:
                summary = "{} {}".format(summary, detokenized_sentence)

        return paragraph, summary

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    full_paragraph, full_summary = self._get_full_paragraph_and_summary(each_data)
                    ex = {
                        "id": each_data["id"],
                        "document": full_paragraph,
                        "summary": full_summary
                    }
                    yield i, ex
                    i+=1

        elif self.config.schema == "nusantara_t2t":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    full_paragraph, full_summary = self._get_full_paragraph_and_summary(each_data)
                    ex = {
                        "id": each_data["id"],
                        "text_1": full_paragraph,
                        "text_2": full_summary,
                        "text_1_name": "document",
                        "text_2_name": "summary"
                    }
                    yield i, ex
                    i+=1
