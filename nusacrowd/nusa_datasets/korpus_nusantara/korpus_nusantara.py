from pathlib import Path
from typing import List

import re
import datasets
import pandas as pd

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks, DEFAULT_SOURCE_VIEW_NAME, DEFAULT_NUSANTARA_VIEW_NAME

_DATASETNAME = "korpus_nusantara"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_NUSANTARA_VIEW_NAME

_LANGUAGES = ["ind", "jav", "xdy", "bug", "sun", "mad", "bjn", "bbc", "khek", "msa", "min", "tiociu"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@article{sujaini2020improving,
  title={Improving the role of language model in statistical machine translation (Indonesian-Javanese)},
  author={Sujaini, Herry},
  journal={International Journal of Electrical and Computer Engineering},
  volume={10},
  number={2},
  pages={2102},
  year={2020},
  publisher={IAES Institute of Advanced Engineering and Science}
}
"""

_DESCRIPTION = """\
This parallel corpus was collected from several studies, assignments, and thesis of 
students of the Informatics Study Program, Tanjungpura University. Some of the corpus 
are used in the translation machine from Indonesian to local languages http://nustor.untan.ac.id/cammane/. 
This corpus can be used freely for research purposes by citing the paper 
https://ijece.iaescore.com/index.php/IJECE/article/download/20046/13738.

The dataset is a combination of multiple machine translation works from the author, 
Herry Sujaini, covering Indonesian to 25 local dialects in Indonesia. Since not all 
dialects have ISO639-3 standard coding, as agreed with Pak Herry , we decided to 
group the dataset into the closest language family, i.e.: Javanese, Dayak, Buginese, 
Sundanese, Madurese, Banjar, Batak Toba, Khek, Malay, Minangkabau, and Tiociu.
"""

_HOMEPAGE = "https://github.com/herrysujaini/korpusnusantara"
_LICENSE = "Unknown"
_URLS = {
    _DATASETNAME: "https://github.com/herrysujaini/korpusnusantara/raw/main/korpus nusantara.xlsx",
}
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


"""
A collection of all the dialects are: javanese, javanese kromo, javanese ngoko, dayak ahe, 
dayak iban, dayak pesaguan, dayak taman, buginese kelolau, buginese wajo, sundanese, 
madurese, banjar, batak toba, khek pontianak, kapuas hulu, melayu kembayan, melayu ketapang, 
melayu melawi, melayu pontianak, melayu putussibau, melayu sambas, melayu sintang, padang, 
tiociu pontianak.

In this project, we group the dialects into several subsets:

Javanese (jav)   : javanese, javanese kromo, javanese ngoko
Dayak (day)      : dayak ahe, dayak iban, dayak pesaguan, dayak taman
Buginese (bug)   : buginese kelolau, buginese wajo
Sundanese (sun)  : sundanese
Madurese (mad)   : madurese
Banjar (bjn)     : banjar
Batak Toba (bbc) : batak toba
Khek (khek)      : khek pontianak, kapuas hulu
Malay (msa)      : melayu kembayan, melayu ketapang, melayu melawi, melayu pontianak, melayu putussibau, melayu sambas, melayu sintang
Minangkabau (min): padang
Tiociu (tiociu)  : tiociu pontianak
"""

Domain2Subsets = {
    "jav": ['jawa', 'jawa kromo', 'jawa ngoko'],
    "xdy": ['dayak ahe', 'dayak iban', 'dayak pesaguan', 'dayak taman'],
    "bug": ['bugis kelolao', 'bugis wajo'],
    "sun": ['sunda'],
    "mad": ['madura'],
    "bjn": ['banjar'],
    "bbc": ['Batak'],
    "khek": ['kapuas hulu', 'Khek Pontianak'],
    "msa": ['melayu kembayan', 'melayu ketapang', 'melayu melawi', 'melayu pontianak', 'melayu putussibau', 'melayu sambas', 'melayu sintang'],
    "min": ['padang'],
    "tiociu": ['Tiociu Pontianak'],
}

class KorpusNusantara(datasets.GeneratorBasedBuilder):
    """Bible En-Id is a machine translation dataset containing Indonesian-English parallel sentences collected from the bible.."""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"korpus_nusantara_ind_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"Korpus_Nusantara ind2{subset} source schema",
            schema="source",
            subset_id=f"korpus_nusantara",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"korpus_nusantara_ind_{subset}_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"Korpus_Nusantara ind2{subset} Nusantara schema",
            schema="nusantara_t2t",
            subset_id=f"korpus_nusantara",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"korpus_nusantara_{subset}_ind_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"Korpus_Nusantara {subset}2ind source schema",
            schema="source",
            subset_id=f"korpus_nusantara",
        )
        for subset in _LANGUAGES[1:]
    ] + \
    [
        NusantaraConfig(
            name=f"korpus_nusantara_{subset}_ind_nusantara_t2t",
            version=datasets.Version(_NUSANTARA_VERSION),
            description=f"Korpus_Nusantara {subset}2ind Nusantara schema",
            schema="nusantara_t2t",
            subset_id=f"korpus_nusantara",
        )
        for subset in _LANGUAGES[1:]
    ]

    DEFAULT_CONFIG_NAME = "korpus_nusantara_jav_ind_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string")})
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
        # Dataset does not have predetermined split, putting all as TRAIN
        urls = _URLS[_DATASETNAME]
        base_dir = Path(dl_manager.download(urls))
        data_files = {"train": base_dir}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                },
            ),
        ]
    
    def _merge_subsets(self, dfs, subsets, revert=False):
        if not subsets: return None
        df = None
        for subset in subsets:
            sub_df = dfs[subset]
            orig_columns = sub_df.columns.tolist()
            sub_df.columns = ["label", "text"]+orig_columns[2:] if revert else ["text", "label"]+orig_columns[2:]
            if df is None:
                df = sub_df
            else:
                df = pd.concat([df, sub_df], axis=0, sort=False)
        return df
        
    def get_domain_data(self, dfs):
        domain = self.config.name
        matched_domain = re.findall(r"korpus_nusantara_.*?_.*?_", domain)
        
        assert len(matched_domain) == 1
        domain = matched_domain[0][:-1].replace("korpus_nusantara_", "").split("_")
        src_lang, tgt_lang = domain[0], domain[1]
        
        subsets = Domain2Subsets.get(src_lang if src_lang != "ind" else tgt_lang, None)
        return src_lang, tgt_lang, self._merge_subsets(dfs, subsets, revert=(src_lang != "ind"))

    def _generate_examples(self, filepath: Path):
        """Yields examples as (key, example) tuples."""
        dfs = pd.read_excel(filepath, sheet_name=None, header=None)
        src_lang, tgt_lang, df = self.get_domain_data((dfs))
        
        if self.config.schema == "source":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text": row.text,
                    "label": row.label,
                }
                yield idx, ex
                
        elif self.config.schema == "nusantara_t2t":
            for idx, row in enumerate(df.itertuples()):
                ex = {
                    "id": str(idx),
                    "text_1": row.text,
                    "text_2": row.label,
                    "text_1_name": src_lang,
                    "text_2_name": tgt_lang,
                }
                yield idx, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
