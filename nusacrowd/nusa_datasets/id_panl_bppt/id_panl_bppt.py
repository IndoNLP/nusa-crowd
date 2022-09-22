import os
from typing import Dict, List, Tuple

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{id_panl_bppt,
  author    = {PAN Localization - BPPT},
  title     = {Parallel Text Corpora, English Indonesian},
  year      = {2009},
  url       = {http://digilib.bppt.go.id/sampul/p92-budiono.pdf},
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "id_panl_bppt"
_DESCRIPTION = """\
Parallel Text Corpora for Multi-Domain Translation System created by BPPT (Indonesian Agency for the Assessment and
Application of Technology) for PAN Localization Project (A Regional Initiative to Develop Local Language Computing
Capacity in Asia). The dataset contains about 24K sentences in English and Bahasa Indonesia from 4 different topics
(Economy, International Affairs, Science & Technology, and Sports).
"""
_HOMEPAGE = "http://digilib.bppt.go.id/sampul/p92-budiono.pdf"
_LICENSE = ""
_URLS = {
    _DATASETNAME: "https://github.com/cahya-wirawan/indonesian-language-models/raw/master/data/BPPTIndToEngCorpusHalfM.zip",
}
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
# Source has no versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdPanlBppt(datasets.GeneratorBasedBuilder):
    """\
    Dataset contains about ~24K sentences in English and Bahasa Indonesia from 4 different topics (Economy,
    International Affairs, Science & Technology, and Sports)
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    class Topic(TypedDict):
        name: Literal["Economy", "International", "Science", "Sport"]
        # seems to be the number of words in the file
        words: Literal["150K", "100K"]

    TOPICS: List[Topic] = [{"name": "Economy", "words": "150K"}, {"name": "International", "words": "150K"}, {"name": "Science", "words": "100K"}, {"name": "Sport", "words": "100K"}]

    SOURCE_LANGUAGE = "en"
    TARGET_LANGUAGE = "id"

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_panl_bppt_source",
            version=SOURCE_VERSION,
            description="PANL BPPT source schema",
            schema="source",
            subset_id="id_panl_bppt",
        ),
        NusantaraConfig(
            name="id_panl_bppt_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="PANL BPPT Nusantara schema",
            schema="nusantara_t2t",
            subset_id="id_panl_bppt",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_panl_bppt_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "translation": datasets.features.Translation(languages=[self.SOURCE_LANGUAGE, self.TARGET_LANGUAGE]),
                    "topic": datasets.features.ClassLabel(names=list(map(lambda topic: topic["name"], self.TOPICS))),
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
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "dir": os.path.join(data_dir, "plain"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, dir: str, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        id = 0
        for topic in self.TOPICS:
            src_path = f"PANL-BPPT-{topic['name'][:3].upper()}-{self.SOURCE_LANGUAGE.upper()}-{topic['words']}w.txt"
            tgt_path = f"PANL-BPPT-{topic['name'][:3].upper()}-{self.TARGET_LANGUAGE.upper()}-{topic['words']}w.txt"
            with open(os.path.join(dir, src_path), encoding="utf-8") as f1, open(os.path.join(dir, tgt_path), encoding="utf-8") as f2:
                src = f1.read().split("\n")[:-1]
                tgt = f2.read().split("\n")[:-1]
                for s, t in zip(src, tgt):
                    if self.config.schema == "source":
                        yield id, {
                            "id": str(id),
                            "translation": {self.SOURCE_LANGUAGE: s, self.TARGET_LANGUAGE: t},
                            "topic": topic["name"],
                        }
                    elif self.config.schema == "nusantara_t2t":
                        # Schema does not have topics or any other fields to have the topics
                        yield id, {
                            "id": str(id),
                            "text_1": s,
                            "text_2": t,
                            "text_1_name": self.SOURCE_LANGUAGE,
                            "text_2_name": self.TARGET_LANGUAGE,
                        }

                    id += 1
