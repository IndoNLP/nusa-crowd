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

"""\
Data loader implementation for IDENTICv1.0 dataset.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.common_parser import load_ud_data
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{larasati-2012-identic,
    title = "{IDENTIC} Corpus: Morphologically Enriched {I}ndonesian-{E}nglish Parallel Corpus",
    author = "Larasati, Septina Dian",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/644_Paper.pdf",
    pages = "902--906",
    abstract = "This paper describes the creation process of an Indonesian-English parallel corpus (IDENTIC).
    The corpus contains 45,000 sentences collected from different sources in different genres.
    Several manual text preprocessing tasks, such as alignment and spelling correction, are applied to the corpus
    to assure its quality. We also apply language specific text processing such as tokenization on both sides and
    clitic normalization on the Indonesian side. The corpus is available in two different formats: plain',
    stored in text format and morphologically enriched', stored in CoNLL format. Some parts of the corpus are
    publicly available at the IDENTIC homepage.",
}
"""

_DATASETNAME = "identic"

_DESCRIPTION = """\
IDENTIC is an Indonesian-English parallel corpus for research purposes.
The corpus is a bilingual corpus paired with English. The aim of this work is to build and provide
researchers a proper Indonesian-English textual data set and also to promote research in this language pair.
The corpus contains texts coming from different sources with different genres.
Additionally, the corpus contains tagged texts that follows MorphInd tagset (Larasati et. al., 2011).
"""

_HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-BF85-F"

_LICENSE = "CC BY-NC-SA 3.0"

_URLS = {
    _DATASETNAME: "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0005-BF85-F/IDENTICv1.0.zip?sequence=1&isAllowed=y",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION, Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"

_LANGUAGES = ["ind", "eng"]

_LOCAL = False

SOURCE_VARIATION = ["raw", "tokenized", "noclitic"]

tagsets_map = {
    # ind
    "07<c>_CO-$": "CO-",
    "176<c>_CO-$": "CO-",
    "F--.^com.<f>_F--$": "X--",
    "F--.^xi<x>_X--$.^b<x>_X--$.^2.<c>_CC-$": "X--",
    "X--.^0.<c>_CC-$": "X--",
    "X--.^a.<x>_X--$": "X--",
    "X--.^b.<x>_X--$": "X--",
    "X--.^c.<x>_X--$": "X--",
    "X--.^com.<f>_F--$": "X--",
    "X--.^gammima<x>_X--$.^ag.<f>_F--$": "X--",
    "X--.^h.<x>_X--$": "X--",
    "X--.^i.<x>_X--$": "X--",
    "X--.^j.<x>_X--$": "X--",
    "X--.^m.<f>_F--$": "X--",
    "X--.^n.<x>_X--$": "X--",
    "X--.^net.<x>_X--$": "X--",
    "X--.^okezone<x>_X--$.^com.<f>_F--$": "X--",
    "X--.^p<x>_X--$.^k.<x>_X--$": "X--",
    "X--.^r.<x>_X--$": "X--",
    "X--.^s.<x>_X--$": "X--",
    "X--.^w.<x>_X--$": "D--",
    "^ke+dua": "D--",
    "^ke+p": "D--",
    "^nya$": "D--",
    "duanya<c>_CO-$": "CO-",
}


def nusantara_config_constructor(version, variation=None, task="source", lang="id"):
    if variation not in SOURCE_VARIATION:
        raise NotImplementedError("'{var}' is not available".format(var=variation))

    ver = datasets.Version(version)

    if task == "seq_label":
        return NusantaraConfig(
            name="identic_{lang}_nusantara_seq_label".format(lang=lang),
            version=ver,
            description="IDENTIC {lang} source schema".format(lang=lang),
            schema="nusantara_seq_label",
            subset_id="identic",
        )
    else:
        return NusantaraConfig(
            name="identic_{var}_{task}".format(var=variation, task=task),
            version=ver,
            description="IDENTIC {var} source schema".format(var=variation),
            schema=task,
            subset_id="identic",
        )


def load_ud_data_as_pos_tag(filepath, lang):
    dataset_source = list(load_ud_data(filepath))

    if lang == "id":
        return [{"id": str(i + 1), "tokens": row["form"], "labels": [tagsets_map.get(pos_tag, pos_tag) for pos_tag in row["xpos"]]} for (i, row) in enumerate(dataset_source)]
    else:
        return [{"id": str(i + 1), "tokens": row["form"], "labels": row["xpos"]} for (i, row) in enumerate(dataset_source)]


class IdenticDataset(datasets.GeneratorBasedBuilder):
    """
    IDENTIC is an Indonesian-English parallel corpus for research purposes. This dataset is used for ind -> eng translation and vice versa, as well for POS-Tagging task.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    # Details of the tagsets in https://septinalarasati.com/morphind/
    TAGSETS = [
        # en
        "#",
        "$",
        "''",
        ",",
        ".",
        ":",
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNP",
        "NNS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
        "``",
        # id
        "APP",
        "ASP",
        "ASS",
        "B--",
        "CC-",
        "CD-",
        "CO-",
        "D--",
        "F--",
        "G--",
        "H--",
        "I--",
        "M--",
        "NPD",
        "NSD",
        "NSF",
        "NSM",
        "O--",
        "PP1",
        "PP3",
        "PS1",
        "PS2",
        "PS3",
        "R--",
        "S--",
        "T--",
        "VPA",
        "VPP",
        "VSA",
        "VSP",
        "W--",
        "X--",
        "Z--",
    ]

    BUILDER_CONFIGS = (
        [
            NusantaraConfig(
                name="identic_source",
                version=SOURCE_VERSION,
                description="identic source schema",
                schema="source",
                subset_id="identic",
            ),
            NusantaraConfig(
                name="identic_id_source",
                version=SOURCE_VERSION,
                description="identic source schema",
                schema="source",
                subset_id="identic",
            ),
            NusantaraConfig(
                name="identic_en_source",
                version=SOURCE_VERSION,
                description="identic source schema",
                schema="source",
                subset_id="identic",
            ),
            NusantaraConfig(
                name="identic_nusantara_t2t",
                version=NUSANTARA_VERSION,
                description="Identic Nusantara schema",
                schema="nusantara_t2t",
                subset_id="identic",
            ),
            NusantaraConfig(
                name="identic_nusantara_seq_label",
                version=NUSANTARA_VERSION,
                description="Identic Nusantara schema",
                schema="nusantara_seq_label",
                subset_id="identic",
            ),
        ]
        + [nusantara_config_constructor(_NUSANTARA_VERSION, var) for var in SOURCE_VARIATION]
        + [nusantara_config_constructor(_NUSANTARA_VERSION, var, "nusantara_t2t") for var in SOURCE_VARIATION]
        + [nusantara_config_constructor(_NUSANTARA_VERSION, "raw", task="seq_label", lang=lang) for lang in ["en", "id"]]
    )

    DEFAULT_CONFIG_NAME = "identic_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if self.config.name.endswith("id_source") or self.config.name.endswith("en_source"):
                features = datasets.Features(
                    {
                        "id": [datasets.Value("string")],
                        "form": [datasets.Value("string")],
                        "lemma": [datasets.Value("string")],
                        "upos": [datasets.Value("string")],
                        "xpos": [datasets.Value("string")],
                        "feats": [datasets.Value("string")],
                        "head": [datasets.Value("string")],
                        "deprel": [datasets.Value("string")],
                        "deps": [datasets.Value("string")],
                        "misc": [datasets.Value("string")],
                    }
                )
            else:
                features = datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "id_sentence": datasets.Value("string"),
                        "en_sentence": datasets.Value("string"),
                    }
                )

        elif self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features

        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.TAGSETS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        base_dir = dl_manager.download_and_extract(urls)

        name_split = self.config.name.split("_")

        lang = name_split[1] if name_split[1] in ["en", "id"] else None

        if name_split[-1] == "source":
            if len(name_split) == 2:
                data_dir = base_dir + "/IDENTICv1.0/identic.raw.npp.txt"
            else:
                if name_split[1] in ["en", "id"]:
                    data_dir = base_dir + "/IDENTICv1.0/identic.raw.npp.txt"
                else:
                    data_dir = base_dir + "/IDENTICv1.0/identic.{var}.npp.txt".format(var=name_split[1])
        elif name_split[-1] == "t2t":
            if len(name_split) == 3:
                data_dir = base_dir + "/IDENTICv1.0/identic.raw.npp.txt"
            else:
                data_dir = base_dir + "/IDENTICv1.0/identic.{var}.npp.txt".format(var=name_split[1])
        elif name_split[-1] == "label":
            data_dir = base_dir + "/IDENTICv1.0/identic.raw.npp.txt"
        else:
            raise NotImplementedError("The defined task is not implemented")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(data_dir), "split": datasets.Split.TRAIN, "lang": lang},
            )
        ]

    def _generate_examples(self, filepath: Path, split: str, lang=None) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = self._load_df_from_tsv(filepath)

        if self.config.schema == "source":
            if lang is None:
                # T2T source
                for id, row in df.iterrows():
                    yield id, {"id": row["id"], "id_sentence": row["id_sentence"], "en_sentence": row["en_sentence"]}
            else:
                # conll source
                path = filepath.parent / "{lang}.npp.conll".format(lang=lang)
                for key, example in enumerate(load_ud_data(path)):
                    yield key, example

        elif self.config.schema == "nusantara_t2t":
            for id, row in df.iterrows():
                yield id, {
                    "id": str(id),
                    "text_1": row["id_sentence"],
                    "text_2": row["en_sentence"],
                    "text_1_name": "ind",
                    "text_2_name": "eng",
                }

        elif self.config.schema == "nusantara_seq_label":
            if lang is None:
                lang = "id"
            path = filepath.parent / "{lang}.npp.conll".format(lang=lang)
            for key, example in enumerate(load_ud_data_as_pos_tag(path, lang=lang)):
                yield key, example

    @staticmethod
    def _load_df_from_tsv(path):
        return pd.read_csv(
            path,
            sep="\t",
            names=["id", "id_sentence", "en_sentence"],
            quoting=csv.QUOTE_NONE,
        )
