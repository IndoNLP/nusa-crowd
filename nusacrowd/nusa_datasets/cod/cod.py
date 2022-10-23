import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{majewska2022cross,
  title={Cross-lingual dialogue dataset creation via outline-based generation},
  author={Majewska, Olga and Razumovskaia, Evgeniia and Ponti, Edoardo Maria and Vuli{\'c}, Ivan and Korhonen, Anna},
  journal={arXiv preprint arXiv:2201.13405},
  year={2022}
}
"""

_LANGUAGES = ["ind"]
_LOCAL = False

_DATASETNAME = "cod"

_DESCRIPTION = """\
Cross-lingual Outline-based Dialogue (COD) is a dataset comprised of manually generated, localized, and cross-lingually aligned Task-Oriented-Dialogue (TOD) data that served as the source of dialogue prompts.
COD enables natural language understanding, dialogue state tracking, and end-to-end dialogue modeling and evaluation.
Majewska et al. (2022) create COD using a novel outline-based annotation pipeline for multilingual TOD by Majewska et al. (2022).
English Schema-Guided Dialogue (SGD; Shah et al., 2018; Rastogi et al., 2020) dataset is automatically sampled and mapped into outlines. The outlines are then paraphrased and adapted to the local target domain by human subjects.
"""

_HOMEPAGE = "https://github.com/cambridgeltl/COD"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: {
        "validation": "https://raw.githubusercontent.com/cambridgeltl/COD/main/id_dev.json",
        "test": "https://raw.githubusercontent.com/cambridgeltl/COD/main/id_test.json",
    },
}

_SUPPORTED_TASKS = [Tasks.DIALOGUE_SYSTEM]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """Cross-lingual Outline-based Dialogue (COD) is a dataset comprises manually generated, localised, and cross-lingually aligned Task-Oriented-Dialogue (TOD) data which served as the source of dialogue prompts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="cod_source",
            version=SOURCE_VERSION,
            description="Cross-lingual Outline-based Dialogue (COD) source schema",
            schema="source",
            subset_id="cod",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cod_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "dialogue_id": datasets.Value("string"),
                    "services": [datasets.Value("string")],
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "utterance": datasets.Value("string"),
                            "frames": [
                                {
                                    "actions": [
                                        {
                                            "act": datasets.Value("string"),
                                            "slot": datasets.Value("string"),
                                            "values": [datasets.Value("string")],
                                        }
                                    ],
                                    "service": datasets.Value("string"),
                                    "slots": [
                                        {
                                            "exclusive_end": datasets.Value("int32"),
                                            "slot": datasets.Value("string"),
                                            "start": datasets.Value("int32"),
                                        }
                                    ],
                                    "state": {
                                        "active_intent": datasets.Value("string"),
                                        "requested_slots": [datasets.Value("string")],
                                        "slot_values": [
                                            {"slot": datasets.Value("string"), "values": [datasets.Value("string")]},
                                        ],
                                    },
                                }
                            ],
                        }
                    ],
                }
            )
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        with open(filepath, "r+") as fw:
            data = json.loads(fw.read())

        if self.config.schema == "source":
            for idx, example in enumerate(data):
                example["index"] = str(idx)
                for turn in example["turns"]:
                    for frame in turn["frames"]:
                        if "state" not in frame:
                            continue
                        ls_slot_values = []
                        for slot in frame["state"]["slot_values"]:
                            ls_slot_values.append({"slot": slot, "values": frame["state"]["slot_values"][slot]})
                        frame["state"]["slot_values"] = ls_slot_values

                yield str(idx), example
