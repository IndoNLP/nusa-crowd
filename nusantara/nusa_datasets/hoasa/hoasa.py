import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks
from nusantara.utils import schemas

_CITATION = """
@inproceedings{azhar2019multi,
  title={Multi-label Aspect Categorization with Convolutional Neural Networks and Extreme Gradient Boosting},
  author={A. N. Azhar, M. L. Khodra, and A. P. Sutiono}
  booktitle={Proceedings of the 2019 International Conference on Electrical Engineering and Informatics (ICEEI)},
  pages={35--40},
  year={2019}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "hoasa"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """
HoASA: An aspect-based sentiment analysis dataset consisting of hotel reviews collected from the hotel aggregator platform, AiryRooms. The dataset covers ten different aspects of hotel quality. Similar to the CASA dataset, each review is labeled with a single sentiment label for each aspect. There are four possible sentiment classes for each sentiment label: positive, negative, neutral, and positive-negative. The positivenegative label is given to a review that contains multiple sentiments of the same aspect but for different objects (e.g., cleanliness of bed and toilet).
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://github.com/IndoNLP/indonlu"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = "CC-BY-SA 4.0"

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/train_preprocess.csv",
    "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/valid_preprocess.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/test_preprocess.csv",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    # You will be able to load the "source" or "nusantara" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_nusantara = datasets.load_dataset('my_dataset', name='nusantara')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_nusantara = datasets.load_dataset('my_dataset', name='nusantara', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and Nusantara;
    #  If dataset contains more than one subset (see nusantara/nusa_datasets/smsa.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. smsa_(source|nusantara)_[nusantara_schema_name]
    #   - version: option = (SOURCE_VERSION|NUSANTARA_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|nusantara_[nusantara_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. smsa)
    #  where [nusantara_schema_name] = (kb, pairs, qa, text, t2t)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="hoasa_source",
            version=SOURCE_VERSION,
            description="HoASA source schema",
            schema="source",
            subset_id="hoasa",
        ),
        NusantaraConfig(
            name="hoasa_nusantara_text",
            version=NUSANTARA_VERSION,
            description="HoASA Nusantara schema",
            schema="nusantara_text",
            subset_id="hoasa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "hoasa_source"
    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({
                "index": datasets.Value("int64"),
                "review": datasets.Value("string"),
                "ac": datasets.Value("string"),
                "air_panas": datasets.Value("string"),
                "bau": datasets.Value("string"),
                "general": datasets.Value("string"),
                "kebersihan": datasets.Value("string"),
                "linen": datasets.Value("string"),
                "service": datasets.Value("string"),
                "sunrise_meal": datasets.Value("string"),
                "tv": datasets.Value("string"),
                "wifi": datasets.Value("string"),
            })

        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(["pos", "neut", "neg", "neg_pos"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_csv_path = Path(dl_manager.download_and_extract(_URLS["train"]))
        validation_csv_path = Path(dl_manager.download_and_extract(_URLS["validation"]))
        test_csv_path = Path(dl_manager.download_and_extract(_URLS["test"]))

        data_dir = {
            "train" : train_csv_path,
            "validation" : validation_csv_path,
            "test" : test_csv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir['train'],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir['test'],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir['validation'],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {
                    "index": row.index,
                    "review": row.review,
                    "ac": row.ac,
                    "air_panas": row.air_panas,
                    "bau": row.bau,
                    "general": row.general,
                    "kebersihan": row.kebersihan,
                    "linen": row.linen,
                    "service": row.service,
                    "sunrise_meal": row.sunrise_meal,
                    "tv": row.tv,
                    "wifi": row.wifi,
                }
                yield row.index, entry

        elif self.config.schema == "nusantara_text":
            for row in df.itertuples():
                entry = {
                    "id": str(row.index),
                    "text": row.review,
                    "label": row.ac,
                }
                yield row.index, entry

if __name__ == "__main__":
    datasets.load_dataset(__file__)