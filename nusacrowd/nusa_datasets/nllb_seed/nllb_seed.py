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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "nllb_seed"
_LANGUAGES = ["ace", "bjn", "bug", "eng"]
_LANGUAGE_MAP = {"ace": "Aceh", "bjn": "Banjar", "bug": "Bugis"}
_LANG_CODE_MAP = {"Aceh": "ace", "Banjar": "bjn", "Bugis": "bug"}
_LANGUAGE_PAIR = [("ace", "eng"), ("eng", "ace"), ("bjn", "eng"), ("eng", "bjn"), ("bug", "eng"), ("eng", "bug")]

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
No Language Left Behind Seed Data
NLLB Seed is a set of professionally-translated sentences in the Wikipedia domain. Data for NLLB-Seed was sampled from Wikimedia’s List of articles every Wikipedia should have, a collection of topics in different fields of knowledge and human activity. NLLB-Seed consists of around six thousand sentences in 39 languages. NLLB-Seed is meant to be used for training rather than model evaluation. Due to this difference, NLLB-Seed does not go through the human quality assurance process present in FLORES-200.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://github.com/facebookresearch/flores/tree/main/nllb_seed"

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
    _DATASETNAME: "https://tinyurl.com/NLLBSeed",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"
_LOCAL = False


def nusantara_config_constructor(lang, schema, version):
    if lang == "":
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source" and schema != "nusantara_t2t":
        raise ValueError(f"Invalid schema: {schema}")

    return NusantaraConfig(
        name="nllb_seed_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="nllb_seed {schema} schema for {lang} language".format(lang=_LANGUAGE_MAP[lang], schema=schema),
        schema=schema,
        subset_id="nllb_seed",
    )


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class NLLBSeed(datasets.GeneratorBasedBuilder):
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

    BUILDER_CONFIGS = [nusantara_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGE_MAP] + [nusantara_config_constructor(lang, "nusantara_t2t", _NUSANTARA_VERSION) for lang in _LANGUAGE_MAP]
    """
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="nllb_seed_source",
            version=SOURCE_VERSION,
            description="nllb_seed source schema",
            schema="source",
            subset_id="nllb_seed",
        ),
        NusantaraConfig(
            name="nllb_seed_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="nllb_seed Nusantara schema",
            schema="nusantara_t2t",
            subset_id="nllb_seed",
        ),
    ]
    """
    DEFAULT_CONFIG_NAME = "nllb_seed_ace_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            # TODO: Create your source schema here
            # raise NotImplementedError()

            # EX: Arbitrary NER type dataset
            # features = datasets.Features(
            #    {
            #        "doc_id": datasets.Value("string"),
            #        "text": datasets.Value("string"),
            #        "entities": [
            #            {
            #                "offsets": [datasets.Value("int64")],
            #                "text": datasets.Value("string"),
            #                "type": datasets.Value("string"),
            #                "entity_id": datasets.Value("string"),
            #            }
            #        ],
            #    }
            # )
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": [datasets.Value("string")],
                    "tgt": [datasets.Value("string")],
                }
            )

        # Choose the appropriate nusantara schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple nusantara configs with a nusantara_[nusantara_schema_name] format.

        # For example nusantara_kb, nusantara_t2t
        elif self.config.schema == "nusantara_t2t":
            # e.g. features = schemas.kb_features
            features = schemas.text2text_features
            # TODO: Choose your nusantara schema here
            # raise NotImplementedError()

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
        data_dir = Path(dl_manager.download_and_extract(urls)) / "NLLB-Seed"
        data_subdir = {"ace": os.path.join(data_dir, "ace_Latn-eng_Latn"), "bjn": os.path.join(data_dir, "bjn_Latn-eng_Latn"), "bug": os.path.join(data_dir, "bug_Latn-eng_Latn")}
        lang = self.config.name.split("_")[2]

        """
        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.
        """

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": {lang: os.path.join(data_subdir[lang], lang + "_Latn"), lang + "_eng": os.path.join(data_subdir[lang], "eng_Latn"), "split": "train"}},
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files
        lang = self.config.name.split("_")[2]
        lang_code = lang

        eng_text = open(filepath[lang_code + "_eng"], "r").readlines()
        lang_text = open(filepath[lang_code], "r").readlines()

        eng_text = list(map(str.strip, eng_text))
        lang_text = list(map(str.strip, lang_text))
        """
        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in thing:
                yield key, example

        elif self.config.schema == "nusantara_t2t":
            # TODO: yield (key, example) tuples in the nusantara schema
            for key, example in thing:
                yield key, example
        """
        if self.config.schema == "source":
            for id, (src, tgt) in enumerate(zip(lang_text, eng_text)):
                row = {
                    "id": str(id),
                    "src": [src],
                    "tgt": [tgt],
                }
                yield id, row

        elif self.config.schema == "nusantara_t2t":
            for id, (src, tgt) in enumerate(zip(lang_text, eng_text)):
                row = {
                    "id": str(id),
                    "text_1": src,
                    "text_2": tgt,
                    "text_1_name": lang_code,
                    "text_2_name": "eng",
                }
                yield id, row
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
