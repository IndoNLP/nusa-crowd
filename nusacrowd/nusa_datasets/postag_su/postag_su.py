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

import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@data{FK2/VTAHRH_2022,
    author = {ARDIYANTI SURYANI, ARIE and Widyantoro, Dwi Hendratmo and Purwarianti, Ayu and Sudaryat, Yayat},
    publisher = {Telkom University Dataverse},
    title = {{PoSTagged Sundanese Monolingual Corpus}},
    year = {2022},
    version = {DRAFT VERSION},
    doi = {10.34820/FK2/VTAHRH},
    url = {https://doi.org/10.34820/FK2/VTAHRH}
}

@INPROCEEDINGS{7437678,
  author={Suryani, Arie Ardiyanti and Widyantoro, Dwi Hendratmo and Purwarianti, Ayu and Sudaryat, Yayat},
  booktitle={2015 International Conference on Information Technology Systems and Innovation (ICITSI)},
  title={Experiment on a phrase-based statistical machine translation using PoS Tag information for Sundanese into Indonesian},
  year={2015},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICITSI.2015.7437678}
}
"""

_LANGUAGES = ["sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "postag_su"

_DESCRIPTION = """\
This dataset contains 3616 lines of Sundanese sentences taken from several online magazines (Mangle, Dewan Dakwah Jabar, and Balebat). \
Annotated with PoS Labels by several undergraduates of the Sundanese Language Education Study Program (PPBS), UPI Bandung.
"""

_HOMEPAGE = "https://dataverse.telkomuniversity.ac.id/dataset.xhtml?persistentId=doi:10.34820/FK2/VTAHRH"

_LICENSE = 'CC0 - "Public Domain Dedication"'

_URLS = {
    _DATASETNAME: "https://dataverse.telkomuniversity.ac.id/api/access/datafile/:persistentId?persistentId=doi:10.34820/FK2/VTAHRH/WQIFK8",
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.1.0"

_NUSANTARA_VERSION = "1.0.0"


class PosSunMonoDataset(datasets.GeneratorBasedBuilder):
    """PoSTagged Sundanese Monolingual Corpus"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    # Based on Wicaksono, A. F., & Purwarianti, A. (2010). HMM Based Part-of-Speech Tagger for Bahasa Indonesia. On Proceedings of 4th International MALINDO (Malay and Indonesian Language) Workshop.
    POS_TAGS = [
        "",
        "!",
        '"',
        "'",
        ")",
        ",",
        "-",
        ".",
        "...",
        "....",
        "/",
        ":",
        ";",
        "?",
        "C",
        "CBI",
        "CC",
        "CDC",
        "CDI",
        "CDO",
        "CDP",
        "CDT",
        "CP",
        "CRB",
        "CS",
        "DC",
        "DT",
        "FE",
        "FW",
        "GM",
        "IN",
        "J",
        "JJ",
        "KA",
        "KK",
        "MD",
        "MG",
        "MN",
        "N",
        "NEG",
        "NN",
        "NNA",
        "NNG",
        "NNN",
        "NNO",
        "NNP",
        "NNPP",
        "NP",
        "NPP",
        "OP",
        "PB",
        "PCDP",
        "PR",
        "PRL",
        "PRL|IN",
        "PRN",
        "PRP",
        "RB",
        "RBT",
        "RB|RP",
        "RN",
        "RP",
        "SC",
        "SCC",
        "SC|IN",
        "SYM",
        "UH",
        "VB",
        "VBI",
        "VBT",
        "VRB",
        "W",
        "WH",
        "WHP",
        "WRP",
        "`",
        "–",
        "—",
        "‘",
        "’",
        "“",
        "”",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        NusantaraConfig(
            name=f"{_DATASETNAME}_nusantara_seq_label",
            version=NUSANTARA_VERSION,
            description=f"{_DATASETNAME} Nusantara Seq Label schema",
            schema="nusantara_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"labeled_sentence": datasets.Value("string")})
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.POS_TAGS)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

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
        data_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        def __hotfix(line):
            if line.endswith(" taun|NN 1953.|."):
                return line.replace(" taun|NN 1953.|.", " taun|NN 1953|CDP .|.")
            elif line.endswith(" jeung|CC|CC sasab|RB .|."):
                return line.replace(" jeung|CC|CC sasab|RB .|.", " jeung|CC sasab|RB .|.")
            elif line.startswith("Kagiatan|NN éta|DT dihadiran|VBT kira|-kira "):
                return line.replace("Kagiatan|NN éta|DT dihadiran|VBT kira|-kira ", "Kagiatan|NN éta|DT dihadiran|VBT kira-kira|DT ")
            return line

        with open(filepath, "r", encoding="utf8") as ipt:
            raw = list(map(lambda l: __hotfix(l.rstrip("\n ")), ipt))

        pat_0 = r"(,\|,|\?\|\?|-\|-|!\|!)"
        repl_spc = r" \1 "

        pat_1 = r"([A-Z”])(\.\|\.)"
        pat_2 = r"(\.\|\.)([^. ])"
        repl_spl = r"\1 \2"

        pat_3 = r"([^ ]+\|[^ ]+)\| "
        repl_del = r"\1 "

        pat_4 = r"\|\|"
        repl_dup = r"|"

        def __apply_regex(txt):
            for pat, repl in [(pat_0, repl_spc), (pat_1, repl_spl), (pat_2, repl_spl), (pat_3, repl_del), (pat_4, repl_dup)]:
                txt = re.sub(pat, repl, txt)
            return txt

        def __cleanse_label(token):
            text, label = token
            return text, re.sub(r"([A-Z]+)[.,)]", r"\1", label.upper())

        if self.config.schema == "source":
            for key, example in enumerate(raw):
                yield key, {"labeled_sentence": example}

        elif self.config.schema == "nusantara_seq_label":
            spaced = list(map(__apply_regex, raw))
            data = list(map(lambda l: [__cleanse_label(tok.split("|", 1)) for tok in filter(None, l.split(" "))], spaced))

            for key, example in enumerate(data):
                tokens, labels = zip(*example)
                yield key, {"id": str(key), "tokens": tokens, "labels": labels}

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")


if __name__ == "__main__":
    datasets.load_dataset(__file__)
