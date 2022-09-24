import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@INPROCEEDINGS{
9297929,  
author={Putra, Oddy Virgantara and Wasmanson, Fathin Muhammad and Harmini, Triana and Utama, Shoffin Nahwa},  
booktitle={2020 International Conference on Computer Engineering, Network, and Intelligent Multimedia (CENIM)},   
title={Sundanese Twitter Dataset for Emotion Classification},   
year={2020},  
volume={},  
number={},  
pages={391-395},  
doi={10.1109/CENIM51130.2020.9297929}
}
"""

_DATASETNAME = "sunda_twitter_emotions"

_DESCRIPTION = """\
This dataset is designed for Emotion Classification NLP task.
"""
_HOMEPAGE = ""

_LICENSE = "UNKNOWN"

_URLS = {
	"datasets": "https://raw.githubusercontent.com/virgantara/sundanese-twitter-dataset/master/newdataset.csv"
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]


_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="sunda_twitter_emotions_source",
            version=SOURCE_VERSION,
            description="Sundanese Twitter Dataset for Emotion source schema",
            schema="source",
            subset_id="sunda_twitter_emotions",
        ),
        NusantaraConfig(
            name="sunda_twitter_emotions_nusantara_text",
            version=NUSANTARA_VERSION,
            description="Sundanese Twitter Dataset for Emotion Nusantara schema",
            schema="nusantara_text",
            subset_id="sunda_twitter_emotions",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sunda_twitter_emotions_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({
                "index": datasets.Value("string"),
                "data": datasets.Value("string"), 
                "label": datasets.Value("string")})

        # For example nusantara_kb, nusantara_t2t
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["anger", "joy", "fear", "sadness"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS
        data_dir = Path(dl_manager.download_and_extract(urls['datasets']))
        data_files = {"train":data_dir}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files['train'],
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()
        df.columns = ["index","label", "data"]

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": str(row.index), "data": row.data, "label": row.label}
                yield row.index, ex
        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                ex = {"id": str(row.index), "text": row.data, "label": row.label}
                yield row.index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
        
if __name__ == "__main__":
    datasets.load_dataset(__file__)
