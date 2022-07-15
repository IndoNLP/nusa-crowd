from pathlib import Path
from typing import List

import datasets

from nusantara.utils import schemas
from nusantara.utils.common_parser import load_conll_data
from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "wiki_ann"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind"]
_LOCAL = False
_CITATION = """\
@inproceedings{pan-etal-2017-cross,
    title = "Cross-lingual Name Tagging and Linking for 282 Languages",
    author = "Pan, Xiaoman  and
      Zhang, Boliang  and
      May, Jonathan  and
      Nothman, Joel  and
      Knight, Kevin  and
      Ji, Heng",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-1178",
    doi = "10.18653/v1/P17-1178",
    pages = "1946--1958",
    abstract = "The ambitious goal of this work is to develop a cross-lingual name tagging and linking framework for 282 languages that exist in Wikipedia. Given a document in any of these languages, our framework is able to identify name mentions, assign a coarse-grained or fine-grained type to each mention, and link it to an English Knowledge Base (KB) if it is linkable. We achieve this goal by performing a series of new KB mining methods: generating {``}silver-standard{''} annotations by transferring annotations from English to other languages through cross-lingual links and KB properties, refining annotations through self-training and topic selection, deriving language-specific morphology features from anchor links, and mining word translation pairs from cross-lingual links. Both name tagging and linking results for 282 languages are promising on Wikipedia data and on-Wikipedia data.",
}
@inproceedings{rahimi-etal-2019-massively,
    title = "Massively Multilingual Transfer for {NER}",
    author = "Rahimi, Afshin  and
      Li, Yuan  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1015",
    pages = "151--164",
}
"""

_DESCRIPTION = """\
The wiki_ann dataset contains NER tags with labels from O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4), B-LOC (5), I-LOC (6). The Indonesian subset is used.
"""

_HOMEPAGE = "https://github.com/afshinrahimi/mmner"

_LICENSE = "Apache-2.0 license"

_URLs = {
    "wiki_ann": "https://s3.amazonaws.com/datasets.huggingface.co/wikiann/1.1.0/panx_dataset.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.1.0"
_NUSANTARA_VERSION = "1.0.0"


class WikiAnnDataset(datasets.GeneratorBasedBuilder):
    """wiki_ann is an NER tagging dataset consisting of Wikipedia articles annotated with LOC, PER, and ORG tags"""

    label_classes = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="wiki_ann_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="wiki_ann source schema",
            schema="source",
            subset_id="wiki_ann",
        ),
        NusantaraConfig(
            name="wiki_ann_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="wiki_ann Nusantara schema",
            schema="nusantara_seq_label",
            subset_id="wiki_ann",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wiki_ann_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")],
                                          "ner_tag": [datasets.Value("string")]})
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
        wikiann_dl_dir = Path(dl_manager.download_and_extract(_URLs["wiki_ann"])) / "id.tar.gz"
        # paths = list(dl_manager.iter_archive(wikiann_dl_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "validtion", "filepath": dl_manager.iter_archive(wikiann_dl_dir)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test", "filepath": dl_manager.iter_archive(wikiann_dl_dir)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train", "filepath": dl_manager.iter_archive(wikiann_dl_dir)},
            ),
        ]

    def _tags_to_spans(self, tags):
        """Convert tags to spans. Based on https://github.com/huggingface/datasets/blob/main/datasets/wikiann/wikiann.py"""
        spans = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, string_tag in enumerate(tags):
            # Actual BIO tag.
            bio_tag = string_tag[0]
            assert bio_tag in ["B", "I", "O"], "Invalid Tag"
            conll_tag = string_tag[2:]
            if bio_tag == "O":
                # The span has ended.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "B":
                # We are entering a new span; reset indices and active tag to new span.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) We have IOB1 tagging scheme.
                # We'll process the previous span if it exists, but also include this
                # span. This is important, because otherwise, a model may get a perfect
                # F1 score whilst still including false positive ill-formed spans.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag:
            spans.add((active_conll_tag, (span_start, span_end)))
        # Return sorted list of spans
        return sorted(list(spans), key=lambda x: x[1][0])

    def _get_spans(self, tokens, tags):
        """Convert tags to textspans. Based on https://github.com/huggingface/datasets/blob/main/datasets/wikiann/wikiann.py"""
        spans = self._tags_to_spans(tags)
        text_spans = [x[0] + ": " + " ".join([tokens[i] for i in range(x[1][0], x[1][1] + 1)]) for x in spans]
        if not text_spans:
            text_spans = ["None"]
        return text_spans

    def _generate_examples(self, filepath: Path, split):
        """Based on https://github.com/huggingface/datasets/blob/main/datasets/wikiann/wikiann.py"""
        tokens = []
        ner_tags = []
        langs = []
        guid_index = 1
        for k, file in filepath:
            if k == split:
                for line in file:
                    line = line.decode("utf-8")
                    if line == "" or line == "\n":
                        if tokens:
                            spans = self._get_spans(tokens, ner_tags)
                            if self.config.schema == "source":
                                yield guid_index, {"index": str(guid_index), "tokens": tokens, "ner_tag": ner_tags}
                            elif self.config.schema == "nusantara_seq_label":
                                yield guid_index, {"id": str(guid_index), "tokens": tokens, "labels": ner_tags}
                            else:
                                raise ValueError(f"Invalid config: {self.config.name}")
                            guid_index += 1
                            tokens = []
                            ner_tags = []
                            langs = []
                    else:
                        # wikiann data is tab separated
                        splits = line.split("\t")
                        # strip out en: prefix
                        langs.append(splits[0].split(":")[0])
                        tokens.append(":".join(splits[0].split(":")[1:]))
                        if len(splits) > 1:
                            ner_tags.append(splits[-1].replace("\n", ""))
                        else:
                            # examples have no label in test set
                            ner_tags.append("O")
