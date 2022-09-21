from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{ibrohim-budi-2019-multi,
    title = "Multi-label Hate Speech and Abusive Language Detection in {I}ndonesian {T}witter",
    author = "Ibrohim, Muhammad Okky  and
      Budi, Indra",
    booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-3506",
    doi = "10.18653/v1/W19-3506",
    pages = "46--57",
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "id_multilabel_hs"

_DESCRIPTION = """\
The ID_MULTILABEL_HS dataset is collection of 13,169 tweets in Indonesian language,
designed for hate speech detection NLP task. This dataset is combination from previous research and newly crawled data from Twitter.
This is a multilabel dataset with label details as follows:
-HS : hate speech label;
-Abusive : abusive language label;
-HS_Individual : hate speech targeted to an individual;
-HS_Group : hate speech targeted to a group;
-HS_Religion : hate speech related to religion/creed;
-HS_Race : hate speech related to race/ethnicity;
-HS_Physical : hate speech related to physical/disability;
-HS_Gender : hate speech related to gender/sexual orientation;
-HS_Gender : hate related to other invective/slander;
-HS_Weak : weak hate speech;
-HS_Moderate : moderate hate speech;
-HS_Strong : strong hate speech.
"""

_HOMEPAGE = "https://aclanthology.org/W19-3506/"
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv",
}
_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdAbusive(datasets.GeneratorBasedBuilder):
    """The ID_MULTILABEL_HS dataset is multi-label hate speech and abusive language detection in Indonesian tweets"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_multilabel_hs_source",
            version=SOURCE_VERSION,
            description="ID Multilabel HS source schema",
            schema="source",
            subset_id="id_multilabel_hs",
        ),
        NusantaraConfig(
            name="id_multilabel_hs_nusantara_text_multi",
            version=NUSANTARA_VERSION,
            description="ID Multilabel HS Nusantara schema",
            schema="nusantara_text_multi",
            subset_id="id_multilabel_hs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_multilabel_hs_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({
                "tweet": datasets.Value("string"), 
                "HS": datasets.Value("bool"),
                "Abusive": datasets.Value("bool"), 
                "HS_Individual": datasets.Value("bool"), 
                "HS_Group": datasets.Value("bool"), 
                "HS_Religion": datasets.Value("bool"), 
                "HS_Race": datasets.Value("bool"), 
                "HS_Physical": datasets.Value("bool"), 
                "HS_Gender": datasets.Value("bool"), 
                "HS_Other": datasets.Value("bool"), 
                "HS_Weak": datasets.Value("bool"), 
                "HS_Moderate": datasets.Value("bool"), 
                "HS_Strong": datasets.Value("bool"),
                })
        elif self.config.schema == "nusantara_text_multi":
            features = schemas.text_multi_features([0, 1])

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
        base_dir = Path(dl_manager.download_and_extract(urls))
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
        # Dataset does not have id, using row index as id
        label_cols = ["HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion", "HS_Race", "HS_Physical", "HS_Gender", "HS_Other", "HS_Weak", "HS_Moderate", "HS_Strong"]
        df = pd.read_csv(filepath, encoding="ISO-8859-1").reset_index()
        df.columns = ["id", "tweet"] + label_cols

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "tweet": row.tweet,
                }
                for label in label_cols:
                    ex[label] = getattr(row, label)
                yield row.id, ex

        elif self.config.schema == "nusantara_text_multi":
            for row in df.itertuples():
                ex = {
                    "id": str(row.id),
                    "text": row.tweet,
                    "labels": [label for label in row[3:]],
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
