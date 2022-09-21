from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@INPROCEEDINGS{8275098,
  author={Gultom, Yohanes and Wibowo, Wahyu Catur},
  booktitle={2017 International Workshop on Big Data and Information Security (IWBIS)},
  title={Automatic open domain information extraction from Indonesian text},
  year={2017},
  volume={},
  number={},
  pages={23-30},
  doi={10.1109/IWBIS.2017.8275098}}

@article{DBLP:journals/corr/abs-2011-00677,
  author    = {Fajri Koto and
               Afshin Rahimi and
               Jey Han Lau and
               Timothy Baldwin},
  title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
               Model for Indonesian {NLP}},
  journal   = {CoRR},
  volume    = {abs/2011.00677},
  year      = {2020},
  url       = {https://arxiv.org/abs/2011.00677},
  eprinttype = {arXiv},
  eprint    = {2011.00677},
  timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indolem_nerui"

_DESCRIPTION = """\
NER UI is a Named Entity Recognition dataset that contains 2,125 sentences obtained via an annotation assignment in an NLP course at the University of Indonesia in 2016.
The corpus has three named entity classes: location, organisation, and person with training/dev/test distribution: 1,530/170/42 and based on 5-fold cross validation.
"""

_HOMEPAGE = "https://indolem.github.io/"

_LICENSE = "Creative Commons Attribution 4.0"

_URLS = {
    _DATASETNAME: [
        {
            "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.01.tsv",
            "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.01.tsv",
            "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.01.tsv",
        },
        {
            "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.02.tsv",
            "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.02.tsv",
            "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.02.tsv",
        },
        {
            "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.03.tsv",
            "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.03.tsv",
            "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.03.tsv",
        },
        {
            "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.04.tsv",
            "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.04.tsv",
            "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.04.tsv",
        },
        {
            "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.05.tsv",
            "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.05.tsv",
            "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.05.tsv",
        },
    ]
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndolemNERUIDataset(datasets.GeneratorBasedBuilder):
    """NER UI contains 2,125 sentences obtained via an annotation assignment in an NLP course at the University of Indonesia. The corpus has three named entity classes: location, organisation, and person; and based on 5-fold cross validation."""

    label_classes = [
        "O",
        "B-LOCATION",
        "B-ORGANIZATION",
        "B-PERSON",
        "I-LOCATION",
        "I-ORGANIZATION",
        "I-PERSON",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"indolem_nerui_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Indolem NER UI source schema",
            schema="source",
            subset_id=f"indolem_nerui",
        ),
        NusantaraConfig(
            name=f"indolem_nerui_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Indolem NER UI Nusantara schema",
            schema="nusantara_seq_label",
            subset_id=f"indolem_nerui",
        )
    ] + [
        NusantaraConfig(
            name=f"indolem_nerui_fold{i}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Indolem NER UI source schema",
            schema="source",
            subset_id=f"indolem_nerui_fold{i}",
        )
        for i in range(5)
    ] + [
        NusantaraConfig(
            name=f"indolem_nerui_fold{i}_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Indolem NER UI Nusantara schema",
            schema="nusantara_seq_label",
            subset_id=f"indolem_nerui_fold{i}",
        )
        for i in range(5)
    ]

    DEFAULT_CONFIG_NAME = "indolem_nerui_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "tags": [datasets.Value("string")],
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
        idx = self._get_fold_index()
        urls = _URLS[_DATASETNAME][idx]
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
        conll_dataset = load_conll_data(filepath)

        if self.config.schema == "source":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "index": str(i),
                    "tokens": row["sentence"],
                    "tags": row["label"],
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

    def _get_fold_index(self):
        try:
            subset_id = self.config.subset_id
            idx_fold = subset_id.index("_fold")
            file_id = subset_id[(idx_fold + 5):]
            return int(file_id)
        except:
            # get default: fold0 (index 0)
            return 0
