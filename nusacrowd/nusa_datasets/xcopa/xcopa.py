"""This code is partially taken from https://github.com/huggingface/datasets/blob/main/datasets/xcopa/xcopa.py."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks


_HOMEPAGE = "https://github.com/cambridgeltl/xcopa"

_CITATION = """\
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
@inproceedings{roemmele2011choice,
  title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
  booktitle={2011 AAAI Spring Symposium Series},
  year={2011},
  url={https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF},
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "xcopa"

_DESCRIPTION = """\
  XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across
languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around
the globe. The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages. All the details about the
creation of XCOPA and the implementation of the baselines are available in the paper.
"""

_HOMEPAGE = "https://github.com/cambridgeltl/xcopa"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: [
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/id/val.id.jsonl",
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/id/test.id.jsonl",
    ]
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"



class Xcopa(datasets.GeneratorBasedBuilder):
    """The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across
    languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around
    the globe."""
    
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="xcopa_source",
            version=SOURCE_VERSION,
            description="XCOPA source schema",
            schema="source",
            subset_id="xcopa",
        ),
        NusantaraConfig(
            name="xcopa_nusantara_qa",
            version=NUSANTARA_VERSION,
            description="XCOPA Nusantara schema",
            schema="nusantara_qa",
            subset_id="xcopa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "xcopa_source"
    
    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "choice1": datasets.Value("string"),
                    "choice2": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "idx": datasets.Value("int32"),
                    "changed": datasets.Value("bool"),
                }
            )
        elif self.config.schema == "nusantara_qa":
            features = schemas.qa_features
            
            
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir[0],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir[1],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    idx = data["idx"]
                    yield idx, data
                    
        elif self.config.schema == "nusantara_qa":
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    idx = data["idx"]
                    
                    sample = {
                        "id": str(idx),
                        "question_id": str(idx),
                        "document_id": str(idx),
                        "question": data["question"],
                        "type": "multiple_choice",
                        "choices": [data["choice1"], data["choice2"]],
                        "context": data["premise"],
                        "answer": [data["choice1"] if data["label"] == 0 else data["choice2"]],
                    }
                    yield idx, sample
            
        else:
            raise ValueError(f"Invalid config: {self.config.name}")