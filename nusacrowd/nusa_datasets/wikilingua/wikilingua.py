import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{
    ladhak-wiki-2020,
    title={WikiLingua: A New Benchmark Dataset for Multilingual Abstractive Summarization},
    author={Faisal Ladhak, Esin Durmus, Claire Cardie and Kathleen McKeown},
    booktitle={Findings of EMNLP, 2020},
    year={2020}
}
"""

_DATASETNAME = "wikilingua"

_DESCRIPTION = """\
We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of crosslingual abstractive 
summarization systems. We extract article and summary pairs in 18 languages from WikiHow12, a high quality, 
collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard 
article summary alignments across languages by aligning the images that are used to describe each how-to step in an 
article.
"""

_HOMEPAGE = "https://github.com/esdurmus/Wikilingua"

_LANGUAGES = ["ind"]

_LICENSE = "CC-BY-NC-SA 3.0"

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://drive.google.com/u/0/uc?id=1PGa8j1_IqxiGTc3SU6NMB38sAzxCPS34&export=download"
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class Wikilingua(datasets.GeneratorBasedBuilder):
    """
    The dataset includes 47,511 articles from WikiHow. Extracted gold-standard article-summary alignments across
    languages by aligning the images that are used to describe each how-to step in an article.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="wikilingua_source",
            version=SOURCE_VERSION,
            description="wikilingua source schema",
            schema="source",
            subset_id="wikilingua",
        ),
        NusantaraConfig(
            name="wikilingua_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="wikilingua Nusantara schema",
            schema="nusantara_t2t",
            subset_id="wikilingua",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wikilingua_source"

    def _info(self) -> datasets.DatasetInfo:
        features = []
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "link": datasets.Value("string"),
                    "main_point": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "document": datasets.Value("string"),
                    "english_section_name": datasets.Value("string"),
                    "english_url": datasets.Value("string"),
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
                    "filepath": os.path.join(data_dir),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "rb") as file:
                indonesian_docs = pickle.load(file)

            _id = 1
            for key_link, articles in indonesian_docs.items():
                for main_point, items in articles.items():
                    example = {"id": _id, "link": key_link, "main_point": main_point, "summary": items["summary"], "document": items["document"], "english_section_name": items["english_section_name"], "english_url": items["english_url"]}
                    yield _id, example
                    _id += 1
        elif self.config.schema == "nusantara_t2t":
            with open(filepath, "rb") as file:
                indonesian_docs = pickle.load(file)

            _id = 1
            for key_link, articles in indonesian_docs.items():
                for main_point, items in articles.items():
                    example = {"id": _id, "text_1": items["document"], "text_2": items["summary"], "text_1_name": "document", "text_2_name": "summary"}
                    yield _id, example
                    _id += 1
