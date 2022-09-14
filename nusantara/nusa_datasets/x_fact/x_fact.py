import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusantara.nusa_datasets.x_fact.utils.x_fact_utils import load_x_fact_dataset
from nusantara.utils.configs import NusantaraConfig

# TODO: Add BibTeX citation
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

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "x_fact"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
X-FACT: the largest publicly available multilingual dataset for factual verification of naturally existing realworld claims.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://github.com/utahnlp/x-fact"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = "MIT"

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "train": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/train.all.tsv",
    "validation": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/dev.all.tsv",
    "test": {
        "in_domain": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/test.all.tsv",
        "out_domain": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/ood.tsv",
        "zero_shot": "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/zeroshot.tsv",
    }
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class XFact(datasets.GeneratorBasedBuilder):
    """X-FACT: the largest publicly available multilingual dataset for factual verification of naturally existing realworld claims."""

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
            name="x_fact_source",
            version=SOURCE_VERSION,
            description="x_fact source schema",
            schema="source",
            subset_id="x_fact",
        ),
        NusantaraConfig(
            name="x_fact_nusantara_[nusantara_schema_name]",
            version=NUSANTARA_VERSION,
            description="x_fact Nusantara schema",
            schema="nusantara_[nusantara_schema_name]",
            subset_id="x_fact",
        ),
    ]

    DEFAULT_CONFIG_NAME = "x_fact_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features({
                "language" : datasets.Value("string"),
                "site" : datasets.Value("string"),
                "evidence_1" : datasets.Value("string"),
                "evidence_2" : datasets.Value("string"),
                "evidence_3" : datasets.Value("string"),
                "evidence_4" : datasets.Value("string"),
                "evidence_5" : datasets.Value("string"),
                "link_1" : datasets.Value("string"),
                "link_2" : datasets.Value("string"),
                "link_3" : datasets.Value("string"),
                "link_4" : datasets.Value("string"),
                "link_5" : datasets.Value("string"),
                "claimDate" : datasets.Value("string"),
                "reviewDate" : datasets.Value("string"),
                "claimant" : datasets.Value("string"),
                "claim" : datasets.Value("string"),
                "label" : datasets.Value("string"),
            })

        # Choose the appropriate nusantara schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple nusantara configs with a nusantara_[nusantara_schema_name] format.

        # For example nusantara_kb, nusantara_t2t
        elif self.config.schema == "nusantara_[nusantara_schema_name]":
            # e.g. features = schemas.kb_features
            # TODO: Choose your nusantara schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "nusantara" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in thing:
                yield key, example

        elif self.config.schema == "nusantara_[nusantara_schema_name]":
            # TODO: yield (key, example) tuples in the nusantara schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)