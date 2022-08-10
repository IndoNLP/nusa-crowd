import json
from pathlib import Path
from typing import List

import datasets
import pandas as pd

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks

_CITATION = """\
@INPROCEEDINGS{8629144,  
  author={R. {Jannati} and R. {Mahendra} and C. W. {Wardhana} and M. {Adriani}},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  title={Stance Classification Towards Political Figures on Blog Writing},
  year={2018},
  volume={},
  number={},
  pages={96-101},
}
"""
_DATASETNAME = "id_stance"
_DESCRIPTION = "Stance Classification Towards Political Figures on Blog Writing"
_HOMEPAGE = "https://github.com/reneje/id_stance_dataset_article-Stance-Classification-Towards-Political-Figures-on-Blog-Writing"
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"
_URLs = {
    _DATASETNAME: "https://raw.githubusercontent.com/reneje/id_stance_dataset_article-Stance-Classification-Towards-Political-Figures-on-Blog-Writing/master/dataset_stance_2_label_2018_building_by_rini.csv"
}
_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


def parse_list(content):
    if (not content):
        return []
    try:
        return json.loads(content)
    except:
        return json.loads("[\"" + content[1:-1].replace("\"", "\\\"") + "\"]")


class IdStance(datasets.GeneratorBasedBuilder):
    """-"""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_stance_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="IdStance source schema",
            schema="source",
            subset_id="id_stance",
        ),
        NusantaraConfig(
            name="id_stance_nusantara_pairs",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="IdStance Nusantara schema",
            schema="nusantara_pairs",
            subset_id="id_stance",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_stance_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "person": datasets.Value("string"),
                    "event": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "content": datasets.Value("string"),
                    "stance_final": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_pairs":
            features = schemas.pairs_features(["for", "against", "againts", "no"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_path = Path(dl_manager.download_and_extract(_URLs[_DATASETNAME]))
        data_files = {
            "train": data_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_csv(filepath, sep=";", header="infer", keep_default_na=False).reset_index()
        df.columns = ["id", "Tokoh", "Event", "Judul_Artikel", "Isi_Artikel", "stance_final", ""]
        df.Isi_Artikel = df.Isi_Artikel.apply(parse_list)

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "person": row.Tokoh,
                    "event": row.Event,
                    "title": row.Judul_Artikel,
                    "content": ". ".join(row.Isi_Artikel),
                    "stance_final": row.stance_final
                }
                yield row.id, ex
        elif self.config.schema == "nusantara_pairs":
            for row in df.itertuples():
                ex = {
                    "id": row.id,
                    "text_1": row.Tokoh + " | " + row.Event,
                    "text_2": " ".join([row.Judul_Artikel] + row.Isi_Artikel),
                    "label": row.stance_final
                }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
