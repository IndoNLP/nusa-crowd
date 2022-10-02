import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import datasets

from nusacrowd.nusa_datasets.indocoref.utils.text_preprocess import \
    TextPreprocess
from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{artari-etal-2021-multi,
  title        = {A Multi-Pass Sieve Coreference Resolution for {I}ndonesian},
  author       = {Artari, Valentina Kania Prameswara  and Mahendra, Rahmad  and Jiwanggi, Meganingrum Arista  and Anggraito, Adityo  and Budi, Indra},
  year         = 2021,
  month        = sep,
  booktitle    = {Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)},
  publisher    = {INCOMA Ltd.},
  address      = {Held Online},
  pages        = {79--85},
  url          = {https://aclanthology.org/2021.ranlp-1.10},
  abstract     = {Coreference resolution is an NLP task to find out whether the set of referring expressions belong to the same concept in discourse. A multi-pass sieve is a deterministic coreference model that implements several layers of sieves, where each sieve takes a pair of correlated mentions from a collection of non-coherent mentions. The multi-pass sieve is based on the principle of high precision, followed by increased recall in each sieve. In this work, we examine the portability of the multi-pass sieve coreference resolution model to the Indonesian language. We conduct the experiment on 201 Wikipedia documents and the multi-pass sieve system yields 72.74{\%} of MUC F-measure and 52.18{\%} of BCUBED F-measure.}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indocoref"
_DESCRIPTION = """\
Dataset contains articles from Wikipedia Bahasa Indonesia which fulfill these conditions:
- The pages contain many noun phrases, which the authors subjectively pick: (i) fictional plots, e.g., subtitles for films,
  TV show episodes, and novel stories; (ii) biographies (incl. fictional characters); and (iii) historical events or important events.
- The pages contain significant variation of pronoun and named-entity. We count the number of first, second, third person pronouns,
  and clitic pronouns in the document by applying string matching.We examine the number
of named-entity using the Stanford CoreNLP
NER Tagger (Manning et al., 2014) with a
model trained from the Indonesian corpus
taken from Alfina et al. (2016).
The Wikipedia texts have length of 500 to
2000 words.
We sample 201 of pages from subset of filtered
Wikipedia pages. We hire five annotators who are
undergraduate student in Linguistics department.
They are native in Indonesian. Annotation is carried out using the Script d’Annotation des Chanes
de Rfrence (SACR), a web-based Coreference resolution annotation tool developed by Oberle (2018).
From the 201 texts, there are 16,460 mentions
tagged by the annotators
"""

_HOMEPAGE = "https://github.com/valentinakania/indocoref/"
_LICENSE = "MIT"
_URLS = {
    _DATASETNAME: "https://github.com/valentinakania/indocoref/archive/refs/heads/main.zip",
}
_SUPPORTED_TASKS = [Tasks.COREFERENCE_RESOLUTION]
# Does not seem to have versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class Indocoref(datasets.GeneratorBasedBuilder):
    """A collection of 210 curated articles from Wikipedia Bahasa Indonesia with Coreference Annotations"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="indocoref_source",
            version=SOURCE_VERSION,
            description="Indocoref source schema",
            schema="source",
            subset_id="indocoref",
        ),
        NusantaraConfig(
            name="indocoref_nusantara_kb",
            version=NUSANTARA_VERSION,
            description="Indocoref Nusantara schema",
            schema="nusantara_kb",
            subset_id="indocoref",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indocoref_source"

    def _info(self) -> datasets.DatasetInfo:
        # The dataset does not really come with a schema, the features here come from the returned value
        # of the accompanying utils files.
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "passage": datasets.Value("string"),
                    "mentions": [
                        {
                            "id": datasets.Value("int64"),
                            # Two entities which share a label are coreferences
                            "labels": datasets.Sequence(datasets.Value("string")),
                            "class": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "pronoun": datasets.Value("bool"),
                            "proper": datasets.Value("bool"),
                            "sent": datasets.Value("int32"),
                            "cluster": datasets.Value("int32"),
                            "per": datasets.Value("bool"),
                            "org": datasets.Value("bool"),
                            "loc": datasets.Value("bool"),
                            "ner": datasets.Value("bool"),
                            # "offset" is only available after modifying the original util class
                            # "offset": datasets.Sequence(datasets.Value("int32"))
                            # POS tags were originally available but removed due to polyglot icu dependency
                            # polyglot.Text(passage, hint_language_code='id')
                        }
                    ],
                }
            )
        elif self.config.schema == "nusantara_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    class ReadPassage(TypedDict):
        passage: str
        annotated: str
        mentions: List[any]

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        base_path = Path(dl_manager.download_and_extract(urls)) / "indocoref-main" / "data"
        passage_path = base_path / "passage"
        annotated_path = base_path / "annotated"
        mentions_per_file = TextPreprocess(annotated_path).run(0)

        data: List[self.ReadPassage] = []
        for passage_file_name, annotated_file_name in zip(sorted(os.listdir(passage_path)), sorted(os.listdir(annotated_path))):
            passage_file_path, annotated_file_path = passage_path / passage_file_name, annotated_path / annotated_file_name

            if os.path.isfile(passage_file_path) and os.path.isfile(annotated_file_path):
                with open(passage_file_path, "r") as fpassage, open(annotated_file_path, "r") as fannotated:
                    data.append(self.ReadPassage(passage=fpassage.read(), annotated=fannotated.read(), mentions=mentions_per_file[annotated_file_name]))

        # Dataset has no predefined splits, using datasets.Split.TRAIN for all of the data.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data,
                    "split": "train",
                },
            ),
        ]

    class DisjointSet:
        parent = {}

        def __init__(self, items):
            for item in items:
                self.parent[item] = item

        def find(self, k):
            if self.parent[k] == k:
                return k
            return self.find(self.parent[k])

        def union(self, a, b):
            x = self.find(a)
            y = self.find(b)
            self.parent[x] = y

    def _generate_examples(self, data: List[ReadPassage], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for index, example in enumerate(data):
                passage, mentions = example["passage"], example["mentions"]
                row = {
                    "id": index,
                    "passage": passage,
                    "mentions": [
                        {
                            "id": mention["id"],
                            "labels": mention["labels"],
                            "class": mention["class"],
                            "text": mention["text"],
                            "pronoun": mention["pronoun"],
                            "proper": mention["proper"],
                            "sent": mention["sent"],
                            "cluster": mention["cluster"],
                            "per": mention["per"],
                            "org": mention["org"],
                            "loc": mention["loc"],
                            "ner": mention["ner"],
                        }
                        for mention in mentions
                    ],
                }
                yield index, row

        elif self.config.schema == "nusantara_kb":
            for index, example in enumerate(data):
                passage, mentions = example["passage"], example["mentions"]
                # Annotated text does not have any line breaks but the original passage does
                passage = passage.replace(" \n", " ")
                passage = passage.replace("\n", " ")
                all_labels = {label for mention in mentions for label in mention["labels"]}
                labels_disjoint_set = self.DisjointSet(all_labels)
                for mention in mentions:
                    for i in range(1, len(mention["labels"])):
                        labels_disjoint_set.union(mention["labels"][i], mention["labels"][i - 1])
                coreferences = {}
                for mention in mentions:
                    coreference_id = labels_disjoint_set.find(mention["labels"][0])
                    if coreference_id not in coreferences:
                        coreferences[coreference_id] = []
                    coreferences[coreference_id].append(str(mention["id"]))

                row_id = str(index)
                row = {
                    "id": row_id,
                    "passages": [{"id": "passage-" + row_id, "type": "text", "text": [passage], "offsets": [[0, len(passage)]]}],
                    "entities": [
                        {
                            "id": row_id + "-entity-" + str(mention["id"]),
                            "type": mention["class"],
                            "text": [mention["text"]],
                            "offsets": [list(mention["offset"])],
                            "normalized": [],
                        }
                        for mention in mentions
                    ],
                    "coreferences": [{"id": row_id + "-coreference-" + str(coref_id), "entity_ids": [row_id + "-entity-" + entity_id for entity_id in entity_ids]} for coref_id, entity_ids in enumerate(coreferences.values())],
                    "events": [],
                    "relations": [],
                }
                yield index, row
