import os
from pathlib import Path
from typing import Dict, List, Tuple
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import datasets
import json

from nusacrowd.utils.configs import NusantaraConfig

_CITATION = """\
@article{lin2020xpersona,
  title={XPersona: Evaluating multilingual personalized chatbot},
  author={Lin, Zhaojiang and Liu, Zihan and Winata, Genta Indra and Cahyawijaya, Samuel and Madotto, Andrea and Bang, Yejin and Ishii, Etsuko and Fung, Pascale},
  journal={arXiv preprint arXiv:2003.07568},
  year={2020}
}
@inproceedings{cahyawijaya-etal-2021-indonlg,
    title = "{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation",
    author = "Cahyawijaya, Samuel  and
      Winata, Genta Indra  and
      Wilie, Bryan  and
      Vincentio, Karissa  and
      Li, Xiaohong  and
      Kuncoro, Adhiguna  and
      Ruder, Sebastian  and
      Lim, Zhi Yuan  and
      Bahar, Syafri  and
      Khodra, Masayu  and
      Purwarianti, Ayu  and
      Fung, Pascale",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.699",
    doi = "10.18653/v1/2021.emnlp-main.699",
    pages = "8875--8898"
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "xpersona_id"

_DESCRIPTION = """\
XPersona is a multi-lingual extension of Persona-Chat. 
XPersona dataset includes persona conversations in six different languages other than English for building and evaluating multilingual personalized agents.
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY-SA 4.0"

_URLS = {
    _DATASETNAME: "https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip",
}

_SUPPORTED_TASKS = [Tasks.DIALOGUE_SYSTEM]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

class XPersonaID(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="xpersona_id_source",
            version=SOURCE_VERSION,
            description="XPersona ID source schema",
            schema="source",
            subset_id="xpersona_id",
        ),
        NusantaraConfig(
            name="xpersona_id_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="XPersona ID Nusantara schema",
            schema="nusantara_t2t",
            subset_id="xpersona_id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "xpersona_id_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "persona": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "dialogue": datasets.Sequence(
                        datasets.Sequence(
                            datasets.Value("string")
                        )
                    )
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
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        data_dir = os.path.join(data_dir, "IndoNLG_downstream_tasks/xpersona")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Id_persona_train_corrected.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Id_persona_split_test_human_annotated.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Id_persona_split_valid_human_annotated.json"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        data = json.load(open(filepath, "r"))

        if self.config.schema == "source":
            key = 0
            for each_data in data:
                example = {
                    "persona": each_data["persona"],
                    "dialogue": each_data["dialogue"]
                }
                yield key, example
                key+=1

        elif self.config.schema == "nusantara_t2t":
            id = 0
            key = 0
            for each_data in data:
                persona = " | ".join(each_data["persona"])
                for i in range(len(each_data["dialogue"]) - 1):
                    example = {
                        "text_1_name": persona,
                        "text_2_name": "response"
                    }

                    # for first turn

                    if i == 0:
                        example["id"] = "{}_{}".format(id, i)
                        example["text_1"] = "U: {}".format(each_data["dialogue"][i][0])
                        example["text_2"] = each_data["dialogue"][i][1]
                        yield key, example
                        key+=1

                    # for second turn and other until last turn

                    example["id"] = "{}_{}".format(id, i+1)
                    example["text_1"] = "U: {} | S: {} | U: {}".format(each_data["dialogue"][i][0], each_data["dialogue"][i][1], each_data["dialogue"][i+1][0])
                    example["text_2"] = each_data["dialogue"][i+1][1]
                    yield key, example
                    key+=1
                id+=1
                

