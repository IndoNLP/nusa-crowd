from pathlib import Path
from typing import Dict, List, Tuple

import json
import datasets
from nusacrowd.utils import schemas

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
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
_DATASETNAME = "indolem_ntp"

_DESCRIPTION = """\
NTP (Next Tweet prediction) is one of the comprehensive Indonesian benchmarks that given a list of tweets and an option, we predict if the option is the next tweet or not.
This task is similar to the next sentence prediction (NSP) task used to train BERT (Devlin et al., 2019).
In NTP, each instance consists of a Twitter thread (containing 2 to 4 tweets) that we call the premise, and four possible options for the next tweet, one of which is the actual response from the original thread.

Train: 5681 threads
Development: 811 threads
Test: 1890 threads
"""

_HOMEPAGE = "https://indolem.github.io/"

_LICENSE = "Creative Commons Attribution 4.0"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/indolem/indolem/main/next_tweet_prediction/data/train.json",
        "validation": "https://raw.githubusercontent.com/indolem/indolem/main/next_tweet_prediction/data/dev.json",
        "test": "https://raw.githubusercontent.com/indolem/indolem/main/next_tweet_prediction/data/test.json",
    }
}

_SUPPORTED_TASKS = [Tasks.NEXT_SENTENCE_PREDICTION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndolemNTPDataset(datasets.GeneratorBasedBuilder):
    """NTP (Next Tweet prediction) is based on next sentence prediction (NSP), consists of a Twitter thread (containing  2 to 4 tweets) and four possible options for the next tweet, one of which is the actual response from the original thread."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indolem_ntp_source",
            version=SOURCE_VERSION,
            description="Indolem NTP source schema",
            schema="source",
            subset_id="indolem_ntp",
        ),
        NusantaraConfig(
            name="indolem_ntp_nusantara_pairs",
            version=NUSANTARA_VERSION,
            description="Indolem NTP Nusantara schema",
            schema="nusantara_pairs",
            subset_id="indolem_ntp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indolem_ntp_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tweets": datasets.Value("string"),
                    "next_tweet": datasets.Value("string"),
                    "label": datasets.Value("int8"),
                }
            )
        elif self.config.schema == "nusantara_pairs":
            features = schemas.pairs_features([0, 1])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
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
        data = self._read_data(filepath)
        if self.config.schema == "source":
            for i, row in enumerate(data):
                ex = {
                    "id": str(i),
                    "tweets": row[0],
                    "next_tweet": row[1],
                    "label": row[2],
                }
                yield i, ex

        elif self.config.schema == "nusantara_pairs":
            for i, row in enumerate(data):
                ex = {
                    "id": str(i),
                    "text_1": row[0],
                    "text_2": row[1],
                    "label": row[2],
                }
                yield i, ex

    def _read_data(self, fname):
        data = json.load(open(fname, "r"))
        results = []
        for datum in data:
            tweets = " ".join(datum["tweets"])
            for key, option in datum["next_tweet"]:
                results.append((tweets, option, key))
        return results
