# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This template serves as a starting point for contributing a dataset to the Nusantara Dataset repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[nusantara_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""
from base64 import encode
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_conll_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

# TODO: Add BibTeX citation
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

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "indolem_sentiment"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
IndoLEM (Indonesian Language Evaluation Montage) is a comprehensive Indonesian benchmark that comprises of seven tasks for the Indonesian language. This benchmark is categorized into three pillars of NLP tasks: morpho-syntax, semantics, and discourse.

This dataset is based on binary classification (positive and negative), with distribution:
* Train: 3638 sentences
* Development: 399 sentences
* Test: 1011 sentences

The data is sourced from 1) Twitter [(Koto and Rahmaningtyas, 2017)](https://www.researchgate.net/publication/321757985_InSet_Lexicon_Evaluation_of_a_Word_List_for_Indonesian_Sentiment_Analysis_in_Microblogs)
and 2) [hotel reviews](https://github.com/annisanurulazhar/absa-playground/).

The experiment is based on 5-fold cross validation.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://indolem.github.io/"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and nusantara config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
        'train': 'https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/train0.csv',
        'dev': 'https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/dev0.csv',
        'test': 'https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/test0.csv'
    }
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class IndolemSentimentDataset(datasets.GeneratorBasedBuilder):

    label_classes = ['negative','positive']
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    # You will be able to load the "source" or "nusanrata" configurations with
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
            name="indolem_sentiment_source",
            version=SOURCE_VERSION,
            description="indolem_sentiment source schema",
            schema="source",
            subset_id="indolem_sentiment",
        ),
        NusantaraConfig(
            name="indolem_sentiment_nusantara_text",
            version=NUSANTARA_VERSION,
            description="indolem_sentiment Nusantara schema",
            schema="nusantara_text",
            subset_id="indolem_sentiment",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indolem_sentiment_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.
        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features({"sentence":datasets.Value("string"), "sentiment": datasets.Value("int32")})
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(self.label_classes)

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
        train_data = Path(dl_manager.download(urls['train']))
        test_data = Path(dl_manager.download(urls['test']))
        dev_data = Path(dl_manager.download(urls['dev']))
        
        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train_data,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_data,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_data,
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

        with filepath.open('r', encoding='utf-8') as f:
            line = f.readline()
            id = 0
            while line:
                line = f.readline().strip()
                if len(line) == 0: break
                
                ex = {}
                id += 1
                sentence = line[:-2].strip('"')
                sentiment = int(line[-1])
                if self.config.schema == 'source':
                    ex = {'sentence': sentence, 'sentiment': sentiment}
                elif self.config.schema == 'nusantara_text':
                    ex = {'id': str(id), 'text': str(sentence), 'label': self.label_classes[sentiment]}
                else:
                    raise ValueError(f"Invalid config: {self.config.name}")
                
                yield id, ex

# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py