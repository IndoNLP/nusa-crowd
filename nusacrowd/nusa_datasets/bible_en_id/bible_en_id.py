from pathlib import Path
from typing import List

import datasets
import json

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "bible_en_id"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{cahyawijaya-etal-2021-indonlg,
    title = "{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation",
    author = "Cahyawijaya, Samuel  and
      Winata, Genta Indra  and
      Wilie, Bryan  and
      Vincentio, Karissa  and
      Li, Xiaohong  and
      Kuncoro, Adhiguna  and
      Ruder, Sebastian  and
      Lim, Zhi Yuan  and
      Bahar, Syafri  and
      Khodra, Masayu  and
      Purwarianti, Ayu  and
      Fung, Pascale",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.699",
    doi = "10.18653/v1/2021.emnlp-main.699",
    pages = "8875--8898",
    abstract = "Natural language generation (NLG) benchmarks provide an important avenue to measure progress and develop better NLG systems. Unfortunately, the lack of publicly available NLG benchmarks for low-resource languages poses a challenging barrier for building NLG systems that work well for languages with limited amounts of data. Here we introduce IndoNLG, the first benchmark to measure natural language generation (NLG) progress in three low-resource{---}yet widely spoken{---}languages of Indonesia: Indonesian, Javanese, and Sundanese. Altogether, these languages are spoken by more than 100 million native speakers, and hence constitute an important use case of NLG systems today. Concretely, IndoNLG covers six tasks: summarization, question answering, chit-chat, and three different pairs of machine translation (MT) tasks. We collate a clean pretraining corpus of Indonesian, Sundanese, and Javanese datasets, Indo4B-Plus, which is used to pretrain our models: IndoBART and IndoGPT. We show that IndoBART and IndoGPT achieve competitive performance on all tasks{---}despite using only one-fifth the parameters of a larger multilingual model, mBART-large (Liu et al., 2020). This finding emphasizes the importance of pretraining on closely related, localized languages to achieve more efficient learning and faster inference at very low-resource languages like Javanese and Sundanese.",
}
"""

_DESCRIPTION = """\
Bible En-Id is a machine translation dataset containing Indonesian-English parallel sentences collected from the bible. We also add a Bible dataset to the English Indonesian translation task. Specifically, we collect an Indonesian and an English language Bible and generate a verse-aligned parallel corpus for the English-Indonesian machine translation task. We split the dataset and use 75% as the training set, 10% as the validation set, and 15% as the test set. Each of the datasets is evaluated in both directions, i.e., English to Indonesian (En → Id) and Indonesian to English (Id → En) translations.
"""

_HOMEPAGE = "https://github.com/IndoNLP/indonlg"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {"indonlg": "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip"}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class BibleEnId(datasets.GeneratorBasedBuilder):
    """Bible En-Id is a machine translation dataset containing Indonesian-English parallel sentences collected from the bible.."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="bible_en_id_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Bible En-Id source schema",
            schema="source",
            subset_id="bible_en_id",
        ),
        NusantaraConfig(
            name="bible_en_id_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Bible En-Id Nusantara schema",
            schema="nusantara_t2t",
            subset_id="bible_en_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bible_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string")})
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
        base_path = Path(dl_manager.download_and_extract(_URLs["indonlg"])) / "IndoNLG_downstream_tasks" / "MT_ENGKJV_INZNTV"
        data_files = {
            "train": base_path / "train_preprocess.json",
            "validation": base_path / "valid_preprocess.json",
            "test": base_path / "test_preprocess.json",
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
        data = json.load(open(filepath, "r"))
        if self.config.schema == "source":
            for row in data:
                ex = {"id": row["id"], "text": row["text"], "label": row["label"]}
                yield row["id"], ex
        elif self.config.schema == "nusantara_t2t":
            for row in data:
                ex = {
                    "id": row["id"],
                    "text_1": row["text"],
                    "text_2": row["label"],
                    "text_1_name": "eng",
                    "text_2_name": "ind",
                }
                yield row["id"], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
