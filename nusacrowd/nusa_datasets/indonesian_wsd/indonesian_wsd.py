import os
from pathlib import Path
from typing import Dict, List, Tuple
from nusacrowd.utils.constants import Tasks
from nusacrowd.utils import schemas

import datasets
import json

from nusacrowd.utils.configs import NusantaraConfig

_CITATION = """\
@inproceedings{mahendra-etal-2018-cross,
    title = "Cross-Lingual and Supervised Learning Approach for {I}ndonesian Word Sense Disambiguation Task",
    author = "Mahendra, Rahmad  and
      Septiantri, Heninggar  and
      Wibowo, Haryo Akbarianto  and
      Manurung, Ruli  and
      Adriani, Mirna",
    booktitle = "Proceedings of the 9th Global Wordnet Conference",
    month = jan,
    year = "2018",
    address = "Nanyang Technological University (NTU), Singapore",
    publisher = "Global Wordnet Association",
    url = "https://aclanthology.org/2018.gwc-1.28",
    pages = "245--250",
    abstract = "Ambiguity is a problem we frequently face in Natural Language Processing. Word Sense Disambiguation (WSD) is a task to determine the correct sense of an ambiguous word. However, research in WSD for Indonesian is still rare to find. The availability of English-Indonesian parallel corpora and WordNet for both languages can be used as training data for WSD by applying Cross-Lingual WSD method. This training data is used as an input to build a model using supervised machine learning algorithms. Our research also examines the use of Word Embedding features to build the WSD model.",
}
"""

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "indonesian_wsd"

_DESCRIPTION = """\
Word Sense Disambiguation (WSD) is a task to determine the correct sense of an ambiguous word.
The training data was collected from news websites and manually annotated. The words in training data were processed using the morphological analysis to obtain lemma.
The features being used were some words around the target word (including the words before and after the target word), the nearest verb from the
target word, the transitive verb around the target word, and the document context. 
"""

_HOMEPAGE = ""

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://github.com/rmahendra/Indonesian-WSD/raw/master/dataset-clwsd-ina.zip",
}

_SUPPORTED_TASKS = [Tasks.WORD_SENSE_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_LABELS = [
    {
        "name": "atas",
        "file_ext": ""
    },
    {
        "name": "perdana",
        "file_ext": ".tab"
    },
    {
        "name": "alam",
        "file_ext": ".tab"
    },
    {
        "name": "dasar",
        "file_ext": ".tab"
    },
    {
        "name": "anggur",
        "file_ext": ".tab"
    },
    {
        "name": "kayu",
        "file_ext": ""
    }
]

class IndonesianWSD(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indonesian_wsd_source",
            version=SOURCE_VERSION,
            description="Indonesian WSD source schema",
            schema="source",
            subset_id="indonesian_wsd",
        ),
        NusantaraConfig(
            name="indonesian_wsd_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="Indonesian WSD Nusantara schema",
            schema="nusantara_t2t",
            subset_id="indonesian_wsd",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indonesian_wsd_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
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

        data_dir = os.path.join(data_dir, "dataset")

        datas = []

        for label in _LABELS:
            file_name = f"{label['name']}_t01"
            if label["file_ext"] != "":
                file_name = f"{file_name}{label['file_ext']}"
            
            parsed_data = self._parse_file(os.path.join(data_dir, file_name))
            datas = datas + parsed_data

        path_dumped_file = os.path.join(data_dir, "data.json")
        
        with open(path_dumped_file, 'w') as f:
            f.write(json.dumps(datas))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path_dumped_file,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        data = json.load(open(filepath, "r"))

        if self.config.schema == "source":
            key = 0
            for each_data in data:
                example = {
                    "label": each_data["label"], # sense_id as label
                    "text": each_data["text"]
                }
                yield key, example
                key+=1

        elif self.config.schema == "nusantara_t2t":
            key = 0
            for each_data in data:
                example = {
                    "id": str(key+1),
                    "text_1": each_data["sense_id"],
                    "text_1_name": "label",
                    "text_2": each_data[""],
                    "text_2_name": "text"
                }
                yield key, example
                key+=1

    def _parse_file(self, file_path):
        parsed_lines = open(file_path, "r").readlines()
        data = []
        for line in parsed_lines:
            if len(line.strip()) > 0:
                _, sense_id, text = line[:-1].split("\t")    
                data.append({
                    "sense_id": sense_id,
                    "text": text
                })
        return data
                

