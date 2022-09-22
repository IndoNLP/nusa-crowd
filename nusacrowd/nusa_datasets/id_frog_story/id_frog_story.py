import os
from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{FrogStorytelling,
  author="Moeljadi, David",
  title="Usage of Indonesian Possessive Verbal Predicates : A Statistical Analysis Based on Storytelling Survey",
  journal="Tokyo University Linguistic Papers",
  ISSN="1345-8663",
  publisher="東京大学大学院人文社会系研究科・文学部言語学研究室",
  year="2014",
  month="sep",
  volume="35",
  number="",
  pages="155-176",
  URL="https://ci.nii.ac.jp/naid/120005525793/en/",
  DOI="info:doi/10.15083/00027472",
}
"""
_DATASETNAME = "id_frog_story"
_DESCRIPTION = """\
Indonesian Frog Storytelling Corpus
Indonesian written and spoken corpus, based on the twenty-eight pictures. (http://compling.hss.ntu.edu.sg/who/david/corpus/pictures.pdf)
"""
_HOMEPAGE = "https://github.com/matbahasa/corpus-frog-storytelling"
_LANGUAGES = ["ind"]
_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
_LOCAL = False
_URLS = {
    _DATASETNAME: "https://github.com/matbahasa/corpus-frog-storytelling/archive/refs/heads/master.zip",
}
_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdFrogStory(datasets.GeneratorBasedBuilder):
    """IdFrogStory contains 13 spoken datasets and 11 written datasets"""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="id_frog_story_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="IdFrogStory source schema",
            schema="source",
            subset_id="id_frog_story",
        ),
        NusantaraConfig(
            name="id_frog_story_nusantara_ssp",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="IdFrogStory Nusantara schema",
            schema="nusantara_ssp",
            subset_id="id_frog_story",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_frog_story_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "nusantara_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        base_path = Path(dl_manager.download_and_extract(urls)) / "corpus-frog-storytelling-master" / "data"
        spoken_path = base_path / "spoken"
        written_path = base_path / "written"

        data = []
        for spoken_file_name in sorted(os.listdir(spoken_path)):
            spoken_file_path = spoken_path / spoken_file_name
            if os.path.isfile(spoken_file_path):
                with open(spoken_file_path, "r") as fspoken:
                    data.extend(fspoken.read().strip("\n").split("\n\n"))

        for written_file_name in sorted(os.listdir(written_path)):
            written_file_path = written_path / written_file_name
            if os.path.isfile(written_file_path):
                with open(written_file_path, "r") as fwritten:
                    data.extend(fwritten.read().strip("\n").split("\n\n"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data: List, split: str):
        if self.config.schema == "source":
            for index, row in enumerate(data):
                ex = {
                    "id": index,
                    "text": row
                }
                yield index, ex
        elif self.config.schema == "nusantara_ssp":
            for index, row in enumerate(data):
                ex = {
                    "id": index,
                    "text": row
                }
                yield index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
