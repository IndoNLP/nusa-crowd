from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME
from nusacrowd.utils.common_parser import load_conll_data

_DATASETNAME = "idn_tagged_corpus_csui"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{dinakaramani2014designing,
  title={Designing an Indonesian part of speech tagset and manually tagged Indonesian corpus},
  author={Dinakaramani, Arawinda and Rashel, Fam and Luthfi, Andry and Manurung, Ruli},
  booktitle={2014 International Conference on Asian Language Processing (IALP)},
  pages={66--69},
  year={2014},
  organization={IEEE}
}

@inproceedings{kurniawan2018towards,
  author={Kurniawan, Kemal and Aji, Alham Fikri},
  booktitle={2018 International Conference on Asian Language Processing (IALP)}, 
  title={Toward a Standardized and More Accurate Indonesian Part-of-Speech Tagging}, 
  year={2018},
  volume={},
  number={},
  pages={303-307},
  doi={10.1109/IALP.2018.8629236}}
"""

_DESCRIPTION = """\
Idn-tagged-corpus-CSUI is a POS tagging dataset contains about 10,000 sentences, collected from the PAN Localization Project tagged with 23 POS tag classes.
The POS tagset is created through a detailed study and analysis of existing tagsets and the manual tagging of an Indonesian corpus.
Idn-tagged-corpus-CSUI dataset is splitted into 3 sets with 8000 train, 1000 validation, 1029 test data.
"""

_HOMEPAGE = "https://bahasa.cs.ui.ac.id/postag/corpus"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLs = {
    "train": "https://raw.githubusercontent.com/ir-nlp-csui/idn-tagged-corpus-CSUI/master/IndoNLU-split/train_preprocess.conll",
    "validation": "https://raw.githubusercontent.com/ir-nlp-csui/idn-tagged-corpus-CSUI/master/IndoNLU-split/valid_preprocess.conll",
    "test": "https://raw.githubusercontent.com/ir-nlp-csui/idn-tagged-corpus-CSUI/master/IndoNLU-split/test_preprocess.conll",
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IdnTaggedCorpusCSUI(datasets.GeneratorBasedBuilder):
    """Idn-tagged-corpus-CSUI is a POS tagging dataset contains about 10,000 sentences, collected from the PAN Localization Project tagged with 23 POS tag classes."""
    label_classes=[
        "B-PR",
        "B-CD",
        "I-PR",
        "B-SYM",
        "B-JJ",
        "B-DT",
        "I-UH",
        "I-NND",
        "B-SC",
        "I-WH",
        "I-IN",
        "I-NNP",
        "I-VB",
        "B-IN",
        "B-NND",
        "I-CD",
        "I-JJ",
        "I-X",
        "B-OD",
        "B-RP",
        "B-RB",
        "B-NNP",
        "I-RB",
        "I-Z",
        "B-CC",
        "B-NEG",
        "B-VB",
        "B-NN",
        "B-MD",
        "B-UH",
        "I-NN",
        "B-PRP",
        "I-SC",
        "B-Z",
        "I-PRP",
        "I-OD",
        "I-SYM",
        "B-WH",
        "B-FW",
        "I-CC",
        "B-X",
    ]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="idn_tagged_corpus_csui_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Idn-tagged-corpus-CSUI source schema",
            schema="source",
            subset_id="idn_tagged_corpus_csui",
        ),
        NusantaraConfig(
            name="idn_tagged_corpus_csui_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="Idn-tagged-corpus-CSUI Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="idn_tagged_corpus_csui",
        ),
    ]

    DEFAULT_CONFIG_NAME = "idn_tagged_corpus_csui_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")], "pos_tag": [datasets.Value("string")]})
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(self.label_classes)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        validation_tsv_path = Path(dl_manager.download_and_extract(_URLs["validation"]))
        test_tsv_path = Path(dl_manager.download_and_extract(_URLs["test"]))
        data_files = {
            "train": train_tsv_path,
            "validation": validation_tsv_path,
            "test": test_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        conll_dataset = load_conll_data(filepath)  # [{'sentence': [T1, T2, ..., Tn], 'labels': [L1, L2, ..., Ln]}]

        if self.config.schema == "source":
            for i, row in enumerate(conll_dataset):
                ex = {"index": str(i), "tokens": row["sentence"], "pos_tag": row["label"]}
                yield i, ex
        elif self.config.schema == "nusantara_seq_label":
            for i, row in enumerate(conll_dataset):
                ex = {"id": str(i), "tokens": row["sentence"], "labels": row["label"]}
                yield i, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
