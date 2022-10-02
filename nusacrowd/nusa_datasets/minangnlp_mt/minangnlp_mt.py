from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{koto-koto-2020-towards,
    title = "Towards Computational Linguistics in {M}inangkabau Language: Studies on Sentiment Analysis and Machine Translation",
    author = "Koto, Fajri  and
      Koto, Ikhwan",
    booktitle = "Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation",
    month = oct,
    year = "2020",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.paclic-1.17",
    pages = "138--148",
}
"""

_LOCAL = False
_LANGUAGES = ["min", "ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "minangnlp_mt"
_DESCRIPTION = """\
In this work, we create Minangkabau–Indonesian (MIN-ID) parallel corpus by using Wikipedia. We obtain 224,180 Minangkabau and
510,258 Indonesian articles, and align documents through title matching, resulting in 111,430 MINID document pairs.
After that, we do sentence segmentation based on simple punctuation heuristics and obtain 4,323,315 Minangkabau sentences. We
then use the bilingual dictionary to translate Minangkabau article (MIN) into Indonesian language (ID'). Sentence alignment is conducted using
ROUGE-1 (F1) score (unigram overlap) (Lin, 2004) between ID’ and ID, and we pair each MIN sentencewith an ID sentence based on the highest ROUGE1.
We then discard sentence pairs with a score of less than 0.5 to result in 345,146 MIN-ID parallel sentences.
We observe that the sentence pattern in the collection is highly repetitive (e.g. 100k sentences are about biological term definition). Therefore,
we conduct final filtering based on top-1000 trigram by iteratively discarding sentences until the frequency of each trigram equals to 100. Finally, we
obtain 16,371 MIN-ID parallel sentences and conducted manual evaluation by asking two native Minangkabau speakers to assess the adequacy and
fluency (Koehn and Monz, 2006). The human judgement is based on scale 1–5 (1 means poor quality and 5 otherwise) and conducted against 100 random
samples. We average the weights of two annotators before computing the overall score, and we achieve 4.98 and 4.87 for adequacy and fluency respectively.
This indicates that the resulting corpus is high-quality for machine translation training.
"""

_HOMEPAGE = "https://github.com/fajri91/minangNLP"
_LICENSE = "MIT"
_URLS = {
    _DATASETNAME: "https://github.com/fajri91/minangNLP/archive/refs/heads/master.zip",
}
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
# Dataset does not have versioning
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class MinangNLPmt(datasets.GeneratorBasedBuilder):
    """16,371-size parallel Minangkabau-Indonesian sentence pairs."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="minangnlp_mt_source",
            version=SOURCE_VERSION,
            description="MinangNLP Machine Translation source schema",
            schema="source",
            subset_id="minangnlp_mt",
        ),
        NusantaraConfig(
            name="minangnlp_mt_nusantara_t2t",
            version=NUSANTARA_VERSION,
            description="MinangNLP Machine Translation Nusantara schema",
            schema="nusantara_t2t",
            subset_id="minangnlp_mt",
        ),
    ]

    DEFAULT_CONFIG_NAME = "minangnlp_mt_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": datasets.Value("string"),
                    "tgt": datasets.Value("string"),
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
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls)) / "minangNLP-master" / "translation" / "wiki_data"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "src_filepath": data_dir / "src_train.txt",
                    "tgt_filepath": data_dir / "tgt_train.txt",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "src_filepath": data_dir / "src_test.txt",
                    "tgt_filepath": data_dir / "tgt_test.txt",
                    "split": "test",
                },
            ),
            # Dataset has a secondary test split
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "src_filepath": data_dir / "src_test_sent.txt",
            #         "tgt_filepath": data_dir / "tgt_test_sent.txt",
            #         "split": "test_sent",
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "src_filepath": data_dir / "src_dev.txt",
                    "tgt_filepath": data_dir / "tgt_dev.txt",
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, src_filepath: Path, tgt_filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(src_filepath, encoding="utf-8") as fsrc, open(tgt_filepath, encoding="utf-8") as ftgt:
            for idx, pair in enumerate(zip(fsrc, ftgt)):
                src, tgt = pair
                if self.config.schema == "source":
                    row = {
                        "id": str(idx),
                        "src": src,
                        "tgt": tgt,
                    }
                    yield idx, row

                elif self.config.schema == "nusantara_t2t":
                    row = {
                        "id": str(idx),
                        "text_1": src,
                        "text_2": tgt,
                        "text_1_name": "min",
                        "text_2_name": "id",
                    }
                    yield idx, row
