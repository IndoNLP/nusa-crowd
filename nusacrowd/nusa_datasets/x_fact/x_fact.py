from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.nusa_datasets.x_fact.utils.x_fact_utils import \
    load_x_fact_dataset
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{gupta2021xfact,
      title={{X-FACT: A New Benchmark Dataset for Multilingual Fact Checking}},
      author={Gupta, Ashim and Srikumar, Vivek},
      booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
      month = jul,
      year = "2021",
      address = "Online",
      publisher = "Association for Computational Linguistics",
}
"""
_DATASETNAME = "x_fact"

_DESCRIPTION = """\
X-FACT: the largest publicly available multilingual dataset for factual verification of naturally existing realworld claims.
"""

_HOMEPAGE = "https://github.com/utahnlp/x-fact"

_LANGUAGES = [
    'ara', 'aze', 'ben', 'deu', 'spa', 
    'fas', 'fra', 'guj', 'hin', 'ind', 
    'ita', 'kat', 'mar', 'nor', 'nld', 
    'pan', 'pol', 'por', 'ron', 'rus',
    'sin', 'srp', 'sqi', 'tam', 'tur'
]
_LOCAL = False

_LICENSE = "MIT"

_URLS = {
    "train": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/train.all.tsv",
    "validation": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/dev.all.tsv",
    "test": {
        "in_domain": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/test.all.tsv",
        "out_domain": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/ood.tsv",
    },
}

_SUPPORTED_TASKS = [Tasks.FACT_CHECKING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class XFact(datasets.GeneratorBasedBuilder):
    """X-FACT: the largest publicly available multilingual dataset for factual verification of naturally existing realworld claims."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="x_fact_source",
            version=SOURCE_VERSION,
            description="x_fact source schema",
            schema="source",
            subset_id="x_fact",
        ),
    ]

    DEFAULT_CONFIG_NAME = "x_fact_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "language": datasets.Value("string"),
                    "site": datasets.Value("string"),
                    "evidence_1": datasets.Value("string"),
                    "evidence_2": datasets.Value("string"),
                    "evidence_3": datasets.Value("string"),
                    "evidence_4": datasets.Value("string"),
                    "evidence_5": datasets.Value("string"),
                    "link_1": datasets.Value("string"),
                    "link_2": datasets.Value("string"),
                    "link_3": datasets.Value("string"),
                    "link_4": datasets.Value("string"),
                    "link_5": datasets.Value("string"),
                    "claimDate": datasets.Value("string"),
                    "reviewDate": datasets.Value("string"),
                    "claimant": datasets.Value("string"),
                    "claim": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": _URLS["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": _URLS["validation"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.splits.NamedSplit("TEST_IN_DOMAIN"),
                gen_kwargs={
                    "filepath": _URLS["test"]["in_domain"],
                    "split": "test_in_domain",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.splits.NamedSplit("TEST_OUT_DOMAIN"),
                gen_kwargs={
                    "filepath": _URLS["test"]["out_domain"],
                    "split": "test_out_domain",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        df = load_x_fact_dataset(filepath)
        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {
                    "language": row.language,
                    "site": row.site,
                    "evidence_1": row.evidence_1,
                    "evidence_2": row.evidence_2,
                    "evidence_3": row.evidence_3,
                    "evidence_4": row.evidence_4,
                    "evidence_5": row.evidence_5,
                    "link_1": row.link_1,
                    "link_2": row.link_2,
                    "link_3": row.link_3,
                    "link_4": row.link_4,
                    "link_5": row.link_5,
                    "claimDate": row.claimDate,
                    "reviewDate": row.reviewDate,
                    "claimant": row.claimant,
                    "claim": row.claim,
                    "label": row.label,
                }
                yield row.index, entry
