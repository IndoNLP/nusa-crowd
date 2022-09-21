from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """
"""

_DATASETNAME = "id_sts"

_DESCRIPTION = """\
SemEval is a series of international natural language processing (NLP) research workshops whose mission is
to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a
range of increasingly challenging problems in natural language semantics. This is a translated version of SemEval Dataset
from 2012-2016 for Semantic Textual Similarity Task to Indonesian language.
"""

_HOMEPAGE = "https://github.com/ahmadizzan/sts-indo"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/ahmadizzan/sts-indo/master/data/final-data/train.tsv",
        "test": "https://raw.githubusercontent.com/ahmadizzan/sts-indo/master/data/final-data/test.tsv",
    }
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class IdSts(datasets.GeneratorBasedBuilder):
    """id_sts, translated version of SemEval Dataset
    from 2012-2016 for Semantic Textual Similarity Task to Indonesian language"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_sts_source",
            version=SOURCE_VERSION,
            description="ID_STS source schema",
            schema="source",
            subset_id="id_sts",
        ),
        NusantaraConfig(
            name="id_sts_nusantara_pairs_score",
            version=NUSANTARA_VERSION,
            description="ID_STS Nusantara schema",
            schema="nusantara_pairs_score",
            subset_id="id_sts",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_sts_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("float64"),
                }
            )
        elif self.config.schema == "nusantara_pairs_score":
            features = schemas.pairs_features_score()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        train_data_path = Path(dl_manager.download(urls["train"]))
        test_data_path = Path(dl_manager.download(urls["test"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_data_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset does not have id, using row index as id
        df = pd.read_csv(filepath, delimiter="\t").reset_index()
        df.columns = ["id", "score", "original_text_1", "original_text_2", "source", "text_1", "text_2"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"text_1": row.text_1, "text_2": row.text_2, "label": row.score}
                yield row.id, ex

        elif self.config.schema == "nusantara_pairs_score":
            for row in df.itertuples():
                ex = {"id": str(row.id), "text_1": row.text_1, "text_2": row.text_2, "label": row.score}
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
