from pathlib import Path
from typing import List

import datasets
import json

import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "bible_en_id"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "abs", "btk", "bew", "bhp", "jav", "mad", "mak", "min", "mui", "rej", "sun"]
_LOCAL = False
_CITATION = """\

"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/IndoNLP"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
	"abs": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-ambon.csv",
	"btk": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-batak.csv",
	"bew": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-betawi.csv",
	"bhp": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-bima.csv",
	"jav": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-javanese.csv",
	"mad": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-madurese.csv",
	"mak": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-makassarese.csv",
	"min": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-minangkabau.csv",
	"mui": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-palembang.csv",
	"rej": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-rejang_lebong.csv",
	"sun": "https://huggingface.co/datasets/indonlp/nusa_translasi/raw/main/mt-sundanese.csv",
}

CODE2LANG = {
    "abs": "Ambon",
    "btk": "Batak",
    "bew": "Betawi",
    "bhp": "Bima",
    "jav": "Javanese",
    "mad": "Madurese",
    "mak": "Makassarese",
    "min": "Minangkabau",
    "mui": "Palembang / Musi",
    "rej": "Rejang (Lebong)",
    "sun": "Sundanese"
}

_SUPPORTED_TASKS = [
    Tasks.MACHINE_TRANSLATION
]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"

class NusaTranslasi(datasets.GeneratorBasedBuilder):
    """NusaTranslasi is a translation dataset from Indonesian to 11 other Indonesian local langauges coversing Ambon, Batak, Betawi, Bima, Javanese, Madurese, Makassarese, Minangkabau, Palembang/Musi, Rejang (Lebong), and Sundanese"""


    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"nusa_translasi_ind_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"NusaTranslasi Indonesian-{CODE2LANG[subset]} source schema",
            schema="source",
            subset_id=f"nusa_translasi",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"nusa_translasi_ind_{subset}_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaTranslasi Indonesian-{CODE2LANG[subset]} Nusantara schema",
            schema="nusantara_t2t",
            subset_id=f"nusa_translasi",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"nusa_translasi_{subset}_ind_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"NusaTranslasi {CODE2LANG[subset]}-Indonesian source schema",
            schema="source",
            subset_id=f"nusa_translasi",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"nusa_translasi_{subset}_ind_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"NusaTranslasi {CODE2LANG[subset]}-Indonesian Nusantara schema",
            schema="nusantara_t2t",
            subset_id=f"nusa_translasi",
        )
        for subset in _LANGUAGES[1:]
    ]

    DEFAULT_CONFIG_NAME = "nusa_translasi_ind_jav_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("string"), "ind_text": datasets.Value("string"), 
                "tgt_text": datasets.Value("string"), "cls_label": datasets.Value("string")
            })
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
        src_lang, tgt_lang = self.config.name.split("_")[2:4]
        print('src_lang, tgt_lang', src_lang, tgt_lang)
        if src_lang == 'ind':
            url = _URLs[tgt_lang]
        else: # if tgt_lang == 'ind':
            url = _URLs[src_lang]
        path = dl_manager.download(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path
                },
            ),
        ]

    def _generate_examples(self, filepath: Path):
        src_lang, tgt_lang = self.config.name.split("_")[2:4]
        
        df = pd.read_csv(filepath)
        if self.config.schema == "source":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": row.id, 
                    "ind_text": row.ind_text, 
                    "tgt_text": row.tgt_text, 
                    "cls_label": row.cls_label
                }
                yield row.id, ex
        elif self.config.schema == "nusantara_t2t":
            for idx, row in enumerate(df.itertuples()):
                if src_lang == 'ind':
                    ex = {
                        "id": row.id,
                        "text_1": row.ind_text,
                        "text_2": row.tgt_text,
                        "text_1_name": src_lang,
                        "text_2_name": tgt_lang
                    }
                else: # tgt_lang == 'ind'
                    ex = {
                        "id": row.id,
                        "text_1": row.tgt_text,
                        "text_2": row.ind_text,
                        "text_1_name": tgt_lang,
                        "text_2_name": src_lang
                    }
                yield row.id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
