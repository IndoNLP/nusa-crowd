import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas
import json

_CITATION = """\
@inproceedings{koto2020liputan6,
  title={Liputan6: A Large-scale Indonesian Dataset for Text Summarization},
  author={Koto, Fajri and Lau, Jey Han and Baldwin, Timothy},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  pages={598--608},
  year={2020}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "liputan6"

_DESCRIPTION = """
A large-scale Indonesian summarization dataset consisting of harvested articles from Liputan6.com, an online news portal, resulting in 215,827 document-summary pairs.
"""

_HOMEPAGE = "https://github.com/fajri91/sum_liputan6"

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    _DATASETNAME: "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class Liputan6(datasets.GeneratorBasedBuilder):
    """A large-scale Indonesian summarization dataset consisting of harvested articles from Liputan6.com, an online news portal, resulting in 215,827 document-summary pairs."""


    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    
    TYPE_LIST = ['canonical', 'xtreme']
    BUILDER_CONFIGS = (
        [
            NusantaraConfig(
                name="liputan6_{fold_name}_source".format(fold_name=i),
                version=_SOURCE_VERSION,
                description="liputan6 source schema",
                schema="source",
                subset_id="liputan6_{fold_name}".format(fold_name=i),
            ) for i in TYPE_LIST
        ]
        +
        [
            NusantaraConfig(
            name="liputan6_{fold_name}_nusantara_t2t".format(fold_name=i),
            version=_NUSANTARA_VERSION,
            description="liputan6 Nusantara schema",
            schema="nusantara_t2t",
            subset_id="liputan6_{fold_name}".format(fold_name=i),
        ) for i in TYPE_LIST
        ]
    )
    DEFAULT_CONFIG_NAME = "liputan6_canonical_source"

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

    def _get_fold_name(self):
        subset_id = self.config.subset_id
        idx_fold = subset_id.index("_")
        file_id = subset_id[(idx_fold + 1):]
        return file_id

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        fold_name = self._get_fold_name()

        urls = _URLS[_DATASETNAME]

        data_dir = Path(dl_manager.download_and_extract(urls))

        location = {
            "train": "IndoNLG_downstream_tasks/liputan6/{fold_name}_train.json",
            "test": "IndoNLG_downstream_tasks/liputan6/{fold_name}_test.json",
            "dev": "IndoNLG_downstream_tasks/liputan6/{fold_name}_dev.json"
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["train"].format(fold_name=fold_name)),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["test"].format(fold_name=fold_name)),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, location["dev"].format(fold_name=fold_name)),
                    "split": "dev",
                },
            ),
        ]
    

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            
            if "xtreme_train.json" in filepath:
                with open(filepath) as f:
                    lines = f.read().split("{")
                    LEN = len(lines)
                    for i, line in enumerate(lines):
                        if 0 < i < LEN-1:
                            idx = line.index("}")
                            line = "{"+line[:idx+1]
                            each_data = json.loads(line)
                            ex = {
                                "id": each_data["id"],
                                "document": each_data['text'],
                                "summary": each_data['label']
                            }
                            yield each_data["id"], ex

            else:
                with open(filepath) as f:
                    data =  json.load(f)
                    for i, each_data in enumerate(data):
                        ex = {
                            "id": each_data["id"],
                            "document": each_data['text'],
                            "summary": each_data['label']
                        }
                        yield each_data["id"], ex

        elif self.config.schema == "nusantara_t2t":
            if "xtreme_train.json" in filepath:
                with open(filepath) as f:
                    lines = f.read().split("{")
                    LEN = len(lines)
                    for i, line in enumerate(lines):
                        if 0 < i < LEN-1:
                            idx = line.index("}")
                            line = "{"+line[:idx+1]
                            each_data = json.loads(line)
                            ex = {
                                "id": each_data["id"],
                                "text_1": each_data['text'],
                                "text_2": each_data['label'],
                                "text_1_name": "document",
                                "text_2_name": "summary"
                            }
                            yield each_data["id"], ex

            else:
                with open(filepath) as f:
                    data =  json.load(f)
                    for i, each_data in enumerate(data):
                        ex = {
                            "id": each_data["id"],
                            "text_1": each_data['text'],
                            "text_2": each_data['label'],
                            "text_1_name": "document",
                            "text_2_name": "summary"
                        }
                        yield each_data["id"], ex
            
            
            
