from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{hidayatullah2020attention,
  title={Attention-based cnn-bilstm for dialect identification on javanese text},
  author={Hidayatullah, Ahmad Fathan and Cahyaningtyas, Siwi and Pamungkas, Rheza Daffa},
  journal={Kinetik: Game Technology, Information System, Computer Network, Computing, Electronics, and Control},
  pages={317--324},
  year={2020}
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "jadi_ide"

_DESCRIPTION = """\
The JaDi-Ide dataset is a Twitter dataset for Javanese dialect identification, containing 16,498 
data samples. The dialect is classified into `Standard Javanese`, `Ngapak Javanese`, and `East 
Javanese` dialects.
"""

_HOMEPAGE = "https://github.com/fathanick/Javanese-Dialect-Identification-from-Twitter-Data"
_LICENSE = "Unknown"
_URLS = {
    _DATASETNAME: "https://github.com/fathanick/Javanese-Dialect-Identification-from-Twitter-Data/raw/main/Update 16K_Dataset.xlsx",
}
# TODO check supported tasks
_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class JaDi_Ide(datasets.GeneratorBasedBuilder):
    """The JaDi-Ide dataset is a Twitter dataset for Javanese dialect identification, containing 16,498 
    data samples."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="jadi_ide_source",
            version=SOURCE_VERSION,
            description="JaDi-Ide source schema",
            schema="source",
            subset_id="jadi_ide",
        ),
        NusantaraConfig(
            name="jadi_ide_nusantara_text",
            version=NUSANTARA_VERSION,
            description="JaDi-Ide Nusantara schema",
            schema="nusantara_text",
            subset_id="jadi_ide",
        ),
    ]

    DEFAULT_CONFIG_NAME = "jadi_ide_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"), 
                    "text": datasets.Value("string"), 
                    "label": datasets.Value("string")
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["Jawa Timur", "Jawa Standar", "Jawa Ngapak"])
   

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # Dataset does not have predetermined split, putting all as TRAIN
        urls = _URLS[_DATASETNAME]
        base_dir = Path(dl_manager.download(urls))
        data_files = {"train": base_dir}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_excel(filepath)
        df.columns = ["id", "text", "label"]

        if self.config.schema == "source":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text": row.text,
                    "label": row.label,
                }
                yield idx, ex

        elif self.config.schema == "nusantara_text":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text": row.text,
                    "label": row.label,
                }
                yield idx, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
