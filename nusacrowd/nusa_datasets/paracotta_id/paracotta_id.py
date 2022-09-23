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
@article{aji2022paracotta,
  title={ParaCotta: Synthetic Multilingual Paraphrase Corpora from the Most Diverse Translation Sample Pair},
  author={Aji, Alham Fikri and Fatyanosa, Tirana Noor and Prasojo, Radityo Eko and Arthur, Philip and Fitriany, Suci and Qonitah, Salma and Zulfa, Nadhifa and Santoso, Tomi and Data, Mahendra},
  journal={arXiv preprint arXiv:2205.04651},
  year={2022}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "paracotta_id"

_DESCRIPTION = """\
ParaCotta is a synthetic parallel paraphrase corpus across 17 languages: Arabic, Catalan, Czech, German, English, Spanish, Estonian, French, Hindi, Indonesian, Italian, Dutch, Ro- manian, Russian, Swedish, Vietnamese, and Chinese.
"""

_HOMEPAGE = "https://github.com/afaji/paracotta-paraphrase"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?id=1QPyD4lOKxbXGUypA5ke6Y9_i9utq-QSQ",
}

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]

# Dataset does not have versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class ParaCotta(datasets.GeneratorBasedBuilder):
    """ParaCotta is a synthetic parallel paraphrase corpus across 17 languages: Arabic, Catalan, Czech, German, English, Spanish, Estonian, French, Hindi, Indonesian, Italian, Dutch, Ro- manian, Russian, Swedish, Vietnamese, and Chinese.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="paracotta_id_source",
            version=SOURCE_VERSION,
            description="paracotta_id source schema",
            schema="source",
            subset_id="paracotta_id",
        ),
        NusantaraConfig(
            name="paracotta_id_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="paracotta_id Nusantara schema",
            schema="nusantara_t2t",
            subset_id="paracotta_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "paracotta_id_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
          features = datasets.Features(
            {
              "id": datasets.Value("string"),
              "src": datasets.Value("string"),
              "tgt": datasets.Value("string"),
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

        data_dir = Path(dl_manager.download(urls))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            with open(filepath, 'r') as f:
                data = f.readlines()
                id = 0
                for each_data in data:
                    each_data = each_data.strip('\n')
                    ex = {
                        "id": id,
                        "src": each_data.split('\t')[1],
                        "tgt": each_data.split('\t')[2],
                    }
                    id += 1
                    yield id, ex

        elif self.config.schema == "nusantara_t2t":
            with open(filepath, 'r') as f:
                data = f.readlines()
                id = 0
                for each_data in data:
                    each_data = each_data.strip('\n')
                    ex = {
                        "id": id,
                        "text_1": each_data.split('\t')[1],
                        "text_2": each_data.split('\t')[2],
                        "text_1_name": "src",
                        "text_2_name": "tgt"
                    }
                    id += 1
                    yield id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")