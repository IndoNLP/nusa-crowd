from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{koto-etal-2020-indolem,
    title = "{I}ndo{LEM} and {I}ndo{BERT}: A Benchmark Dataset and Pre-trained Language Model for {I}ndonesian {NLP}",
    author = "Koto, Fajri  and
      Rahimi, Afshin  and
      Lau, Jey Han  and
      Baldwin, Timothy",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.66",
    doi = "10.18653/v1/2020.coling-main.66",
    pages = "757--770"
}
@phdthesis{fachri2014pengenalan,
  title     = {Pengenalan Entitas Bernama Pada Teks Bahasa Indonesia Menggunakan Hidden Markov Model},
  author    = {FACHRI, MUHAMMAD},
  year      = {2014},
  school    = {Universitas Gadjah Mada}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indolem_ner_ugm"

_DESCRIPTION = """\
NER UGM is a Named Entity Recognition dataset that comprises 2,343 sentences from news articles, and was constructed at the University of Gajah Mada based on five named entity classes: person, organization, location, time, and quantity.
"""

_HOMEPAGE = "https://indolem.github.io/"

_LICENSE = "Creative Commons Attribution 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerugm/train.0{fold_number}.tsv",
        "validation": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerugm/dev.0{fold_number}.tsv",
        "test": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerugm/test.0{fold_number}.tsv"
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class IndolemNERUGM(datasets.GeneratorBasedBuilder):
    """NER UGM comprises 2,343 sentences from news articles, and was constructed at the University of Gajah Mada based on five named entity classes: person, organization, location, time, and quantity; and based on 5-fold cross validation"""

    label_classes = ["B-PERSON", "B-LOCATION", "B-ORGANIZATION", "B-TIME", "B-QUANTITY", "I-PERSON", "I-LOCATION", "I-ORGANIZATION", "I-TIME", "I-QUANTITY", "O"]

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = ( 
    [
        NusantaraConfig(
            name="indolem_ner_ugm_fold{fold_number}_source".format(fold_number=i),
            version=_SOURCE_VERSION,
            description="indolem_ner_ugm source schema",
            schema="source",
            subset_id="indolem_ner_ugm_fold{fold_number}".format(fold_number=i),
        ) for i in range(5)
    ] 
    +   [
            NusantaraConfig(
                name="indolem_ner_ugm_fold{fold_number}_nusantara_seq_label".format(fold_number=i),
                version=_NUSANTARA_VERSION,
                description="indolem_ner_ugm Nusantara schema",
                schema="nusantara_seq_label",
                subset_id="indolem_ner_ugm_fold{fold_number}".format(fold_number=i),
            ) for i in range(5)
        ]
    )

    DEFAULT_CONFIG_NAME = "indolem_ner_ugm_fold0_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "tags": [datasets.Value("string")]
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

        for key in urls:
            urls[key] = urls[key].format(fold_number=idx+1)

        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
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
                    "tags": row["label"]
                }
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {
                    "id": str(i),
                    "tokens": row["sentence"],
                    "labels": row["label"]
                }
                yield i, ex
