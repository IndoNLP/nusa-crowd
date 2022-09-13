from pathlib import Path
from typing import List

import datasets
import pandas as pd
import numpy as np

from nusantara.utils import schemas
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import DEFAULT_NUSANTARA_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Tasks

_DATASETNAME = "barasa"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 langauge code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{baccianella-etal-2010-sentiwordnet,
    title = "{S}enti{W}ord{N}et 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining",
    author = "Baccianella, Stefano  and
      Esuli, Andrea  and
      Sebastiani, Fabrizio",
    booktitle = "Proceedings of the Seventh International Conference on Language Resources and Evaluation ({LREC}'10)",
    month = may,
    year = "2010",
    address = "Valletta, Malta",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2010/pdf/769_Paper.pdf",
    abstract = "In this work we present SENTIWORDNET 3.0, a lexical resource explicitly devised for supporting sentiment classification and opinion mining applications. SENTIWORDNET 3.0 is an improved version of SENTIWORDNET 1.0, a lexical resource publicly available for research purposes, now currently licensed to more than 300 research groups and used in a variety of research projects worldwide. Both SENTIWORDNET 1.0 and 3.0 are the result of automatically annotating all WORDNET synsets according to their degrees of positivity, negativity, and neutrality. SENTIWORDNET 1.0 and 3.0 differ (a) in the versions of WORDNET which they annotate (WORDNET 2.0 and 3.0, respectively), (b) in the algorithm used for automatically annotating WORDNET, which now includes (additionally to the previous semi-supervised learning step) a random-walk step for refining the scores. We here discuss SENTIWORDNET 3.0, especially focussing on the improvements concerning aspect (b) that it embodies with respect to version 1.0. We also report the results of evaluating SENTIWORDNET 3.0 against a fragment of WORDNET 3.0 manually annotated for positivity, negativity, and neutrality; these results indicate accuracy improvements of about 20{\%} with respect to SENTIWORDNET 1.0.",
}
@misc{moeljadi_2016,
    title={Neocl/Barasa: Indonesian SentiWordNet},
    url={https://github.com/neocl/barasa},
    journal={GitHub},
    author={Moeljadi, David},
    year={2016}, month={Mar}
}
"""

_DESCRIPTION = """\
The Barasa dataset is an Indonesian SentiWordNet for sentiment analysis.
For each term, the pair (POS,ID) uniquely identifies a WordNet (3.0) synset and there are PosScore and NegScore to show the positivity and negativity of the term.
The objectivity score can be calculated as: ObjScore = 1 - (PosScore + NegScore).
"""

_HOMEPAGE = "https://github.com/neocl/barasa"

_LICENSE = "MIT"

_URLs = {
    "train": "https://github.com/neocl/barasa/raw/master/data/SentiWordNet_3.0.0_20130122.txt",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"


class Barasa(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="barasa_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Barasa source schema",
            schema="source",
            subset_id="barasa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "barasa_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "POS": datasets.Value("string"),
                    "PosScore": datasets.Value("float32"),
                    "NegScore": datasets.Value("float32"),
                    "SynsetTerms": datasets.Value("string"),
                    "Gloss": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        train_tsv_path = Path(dl_manager.download_and_extract(_URLs["train"]))
        data_files = {
            "train": train_tsv_path,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        df = pd.read_csv(
            filepath, sep="\t", header="infer",
            skiprows=list(range(26)),
            dtype={'ID': str}
        ).reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {
                    "index": str(row.ID),
                    "POS": row[1],
                    "PosScore": row.PosScore,
                    "NegScore": row.NegScore,
                    "SynsetTerms": row.SynsetTerms,
                    "Gloss": row.Gloss,
                }
                yield row.Index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
