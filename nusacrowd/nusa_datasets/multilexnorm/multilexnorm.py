from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{multilexnorm,
  title= {MultiLexNorm: A Shared Task on Multilingual Lexical Normalization,
  author = "van der Goot, Rob and Ramponi et al.",
  booktitle = "Proceedings of the 7th Workshop on Noisy User-generated Text (W-NUT 2021)",
  year = "2021",
  publisher = "Association for Computational Linguistics",
  address = "Punta Cana, Dominican Republic"
}
"""

_DATASETNAME = "multilexnorm"

_DESCRIPTION = """\
MULTILEXNPRM is a new benchmark dataset for multilingual lexical normalization
including 12 language variants,
we here specifically work on the Indonisian-english language.
"""

_HOMEPAGE = "https://bitbucket.org/robvanderg/multilexnorm/src/master/"

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LICENSE = "CC-BY-NC-SA 4.0"

_URLS = {
    "train": "https://bitbucket.org/robvanderg/multilexnorm/raw/e92e5b8f111fea15c7c88aebd4c058f6a1ca8d74/data/iden/train.norm",
    "validation": "https://bitbucket.org/robvanderg/multilexnorm/raw/e92e5b8f111fea15c7c88aebd4c058f6a1ca8d74/data/iden/dev.norm",
    "test": "https://bitbucket.org/robvanderg/multilexnorm/raw/e92e5b8f111fea15c7c88aebd4c058f6a1ca8d74/data/iden/test.norm",
}

_SUPPORTED_TASKS = [Tasks.MULTILEXNORM]

_SOURCE_VERSION = "1.0.0"

_NUSANTARA_VERSION = "1.0.0"


class MultiLexNorm(datasets.GeneratorBasedBuilder):
    """MultiLexNorm is a new benchmark dataset for lexical normalization for indonisian English language. which is the translation
     of social media text to canonical text:
    new pix      comming tomoroe
    new pictures coming  tomorrow
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="multilexnorm_source",
            version=_SOURCE_VERSION,
            description="multilexnorm source schema",
            schema="source",
            subset_id="multilexnorm",
        ),
        NusantaraConfig(
            name="multilexnorm_nusantara_t2t",
            version=_NUSANTARA_VERSION,
            description="multilexnorm Nusantara schema",
            schema="nusantara_t2t",
            subset_id="multilexnorm",
        ),
    ]

    DEFAULT_CONFIG_NAME = "multilexnorm_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "src_sent": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "norm_sent": datasets.Value("string"),
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

        train_path = Path(dl_manager.download_and_extract(_URLS["train"]))
        validation_path = Path(dl_manager.download_and_extract(_URLS["validation"]))
        test_path = Path(dl_manager.download_and_extract(_URLS["test"]))
        data_files = {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        curSent = []
        print(filepath)
        if self.config.schema == "source":
            i = 0
            for line in open(filepath):
                tok = line.strip("\n").split("\t")

                if tok == [""] or tok == []:
                    ex = {"id": str(i), 
                          "src_sent": " ".join([x[0] for x in curSent]), 
                          "norm_sent": " ".join([x[1] for x in curSent])}
                    yield i, ex
                    i += 1
                    curSent = []

                else:
                    if len(tok) > 2:
                        print("erroneous input, line:\n" + line + "\n in file " + filepath + " contains more then two elements")
                    if len(tok) == 1:
                        tok.append("")
                    curSent.append(tok)

        elif self.config.schema == "nusantara_t2t":
            i = 0
            for line in open(filepath):
                tok = line.strip("\n").split("\t")

                if tok == [""] or tok == []:
                    ex = {"id": str(i), 
                          "text_1": " ".join([x[0] for x in curSent]), 
                          "text_2": " ".join([x[1] for x in curSent]), 
                          "text_1_name": "src_sent", 
                          "text_2_name": "norm_sent"}
                    yield i, ex
                    i += 1
                    curSent = []

                else:
                    if len(tok) > 2:
                        print("erroneous input, line:\n" + line + "\n in file " + filepath + " contains more then two elements")
                    if len(tok) == 1:
                        tok.append("")
                    curSent.append(tok)

