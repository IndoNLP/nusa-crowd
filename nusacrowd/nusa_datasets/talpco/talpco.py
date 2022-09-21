import os
from pathlib import Path
from typing import Dict, List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@article{published_papers/22434604,
  title = {TUFS Asian Language Parallel Corpus (TALPCo)},
  author = {Hiroki Nomoto and Kenji Okano and David Moeljadi and Hideo Sawada},
  journal = {言語処理学会 第24回年次大会 発表論文集},
  pages = {436--439},
  year = {2018}
}
@article{published_papers/22434603,
  title = {Interpersonal meaning annotation for Asian language corpora: The case of TUFS Asian Language Parallel Corpus (TALPCo)},
  author = {Hiroki Nomoto and Kenji Okano and Sunisa Wittayapanyanon and Junta Nomura},
  journal = {言語処理学会 第25回年次大会 発表論文集},
  pages = {846--849},
  year = {2019}
}
"""
_DATASETNAME = "talpco"
_DESCRIPTION = """\
The TUFS Asian Language Parallel Corpus (TALPCo) is an open parallel corpus consisting of Japanese sentences
and their translations into Korean, Burmese (Myanmar; the official language of the Republic of the Union of Myanmar),
Malay (the national language of Malaysia, Singapore and Brunei), Indonesian, Thai, Vietnamese and English.
"""
_HOMEPAGE = "https://github.com/matbahasa/TALPCo"
_LOCAL = False
_LANGUAGES = ["eng", "ind", "jpn", "kor", "myn", "tha", "vie", "zsm"]
_LICENSE = "CC-BY 4.0"
_URLS = {
    _DATASETNAME: "https://github.com/matbahasa/TALPCo/archive/refs/heads/master.zip",
}
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


def nusantara_config_constructor(lang_source, lang_target, schema, version):
    """Construct NusantaraConfig with talpco_{lang_source}_{lang_target}_{schema} as the name format"""
    if schema != "source" and schema != "nusantara_t2t":
        raise ValueError(f"Invalid schema: {schema}")

    if lang_source == "" and lang_target == "":
        return NusantaraConfig(
            name="talpco_{schema}".format(schema=schema),
            version=datasets.Version(version),
            description="talpco with {schema} schema for all 7 language pairs from / to ind language".format(schema=schema),
            schema=schema,
            subset_id="talpco",
        )
    else:
        return NusantaraConfig(
            name="talpco_{lang_source}_{lang_target}_{schema}".format(lang_source=lang_source, lang_target=lang_target, schema=schema),
            version=datasets.Version(version),
            description="talpco with {schema} schema for {lang_source} source language and  {lang_target} target language".format(lang_source=lang_source, lang_target=lang_target, schema=schema),
            schema=schema,
            subset_id="talpco",
        )


class TALPCo(datasets.GeneratorBasedBuilder):
    """TALPCo datasets contains 1372 datasets in 8 languages"""

    BUILDER_CONFIGS = (
        [nusantara_config_constructor(lang1, lang2, "source", _SOURCE_VERSION) for lang1 in _LANGUAGES for lang2 in _LANGUAGES if lang1 != lang2]
        + [nusantara_config_constructor(lang1, lang2, "nusantara_t2t", _NUSANTARA_VERSION) for lang1 in _LANGUAGES for lang2 in _LANGUAGES if lang1 != lang2]
        + [nusantara_config_constructor("", "", "source", _SOURCE_VERSION), nusantara_config_constructor("", "", "nusantara_t2t", _NUSANTARA_VERSION)]
    )

    DEFAULT_CONFIG_NAME = "talpco_jpn_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source" or self.config.schema == "nusantara_t2t":
            features = schemas.text2text_features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        base_path = Path(dl_manager.download_and_extract(urls)) / "TALPCo-master"
        data = {}
        for lang in _LANGUAGES:
            lang_file_name = "data_" + lang + ".txt"
            lang_file_path = base_path / lang / lang_file_name
            if os.path.isfile(lang_file_path):
                with open(lang_file_path, "r") as file:
                    data[lang] = file.read().strip("\n").split("\n")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data: Dict, split: str):
        if self.config.schema != "source" and self.config.schema != "nusantara_t2t":
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        if self.config.name == "talpco_source" or self.config.name == "talpco_nusantara_t2t":
            # load all 7 language pairs from / to ind language
            lang_target = "ind"
            for lang_source in _LANGUAGES:
                if lang_source == lang_target:
                    continue
                for language_pair_data in self.generate_language_pair_data(lang_source, lang_target, data):
                    yield language_pair_data

            lang_source = "ind"
            for lang_target in _LANGUAGES:
                if lang_source == lang_target:
                    continue
                for language_pair_data in self.generate_language_pair_data(lang_source, lang_target, data):
                    yield language_pair_data
        else:
            _, lang_source, lang_target = self.config.name.replace(f"_{self.config.schema}", "").split("_")
            for language_pair_data in self.generate_language_pair_data(lang_source, lang_target, data):
                yield language_pair_data

    def generate_language_pair_data(self, lang_source, lang_target, data):
        dict_source = {}
        for row in data[lang_source]:
            id, text = row.split("\t")
            dict_source[id] = text

        dict_target = {}
        for row in data[lang_target]:
            id, text = row.split("\t")
            dict_target[id] = text

        all_ids = set([k for k in dict_source.keys()] + [k for k in dict_target.keys()])
        dict_merged = {k: [dict_source.get(k), dict_target.get(k)] for k in all_ids}

        for id in sorted(all_ids):
            ex = {
                "id": lang_source + "_"  + lang_target + "_"  + id,
                "text_1": dict_merged[id][0],
                "text_2": dict_merged[id][1],
                "text_1_name": lang_source,
                "text_2_name": lang_target,
            }
            yield lang_source + "_" + lang_target + "_"  + id, ex
