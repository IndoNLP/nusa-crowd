"""
Utility for filtering and loading Nusantara datasets.
"""
from collections import Counter
from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Dict

from dataclasses import dataclass
from dataclasses import field
import datasets

from .utils.configs import NusantaraConfig
from .utils.constants import Tasks, SCHEMA_TO_TASKS
import pandas as pd

_LARGE_CONFIG_NAMES = [
    'covost2_ind_eng_nusantara_sptext',
    'covost2_eng_ind_nusantara_sptext',
    'covost2_ind_eng_nusantara_t2t',
    'covost2_eng_ind_nusantara_t2t',
    'cc100_ind_source',
    'cc100_jav_source',
    'cc100_sun_source',
    'cc100_ind_nusantara_ssp',
    'cc100_jav_nusantara_ssp',
    'cc100_sun_nusantara_ssp',
    'indo4b_source', 'indo4b_nusantara_ssp',
    'indo4b_plus_source', 'indo4b_plus_nusantara_ssp',
    'kopi_cc_all-raw_source',
    'kopi_cc_all-dedup_source',
    'kopi_cc_all-neardup_source',
    'kopi_cc_all-neardup_clean_source',
    'kopi_cc_2021_10-dedup_source',
    'kopi_cc_2021_10-neardup_source',
    'kopi_cc_2021_10-neardup_clean_source',
    'kopi_cc_2021_17-raw_source',
    'kopi_cc_2021_17-dedup_source',
    'kopi_cc_2021_17-neardup_source',
    'kopi_cc_2021_17-neardup_clean_source',
    'kopi_cc_2021_21-raw_source',
    'kopi_cc_2021_21-dedup_source',
    'kopi_cc_2021_21-neardup_source',
    'kopi_cc_2021_21-neardup_clean_source',
    'kopi_cc_2021_25-raw_source',
    'kopi_cc_2021_25-dedup_source',
    'kopi_cc_2021_25-neardup_source',
    'kopi_cc_2021_25-neardup_clean_source',
    'kopi_cc_2021_31-raw_source',
    'kopi_cc_2021_31-dedup_source',
    'kopi_cc_2021_31-neardup_source',
    'kopi_cc_2021_31-neardup_clean_source',
    'kopi_cc_2021_39-raw_source',
    'kopi_cc_2021_39-dedup_source',
    'kopi_cc_2021_39-neardup_source',
    'kopi_cc_2021_39-neardup_clean_source',
    'kopi_cc_2021_43-raw_source',
    'kopi_cc_2021_43-dedup_source',
    'kopi_cc_2021_43-neardup_source',
    'kopi_cc_2021_43-neardup_clean_source',
    'kopi_cc_2021_49-dedup_source',
    'kopi_cc_2021_49-neardup_source',
    'kopi_cc_2021_49-neardup_clean_source',
    'kopi_cc_2022_05-raw_source',
    'kopi_cc_2022_05-dedup_source',
    'kopi_cc_2022_05-neardup_source',
    'kopi_cc_2022_05-neardup_clean_source',
    'kopi_cc_2022_21-raw_source',
    'kopi_cc_2022_21-dedup_source',
    'kopi_cc_2022_21-neardup_source',
    'kopi_cc_2022_21-neardup_clean_source',
    'kopi_cc_2022_27-raw_source',
    'kopi_cc_2022_27-dedup_source',
    'kopi_cc_2022_27-neardup_source',
    'kopi_cc_2022_27-neardup_clean_source',
    'kopi_cc_all-raw_nusantara_ssp',
    'kopi_cc_all-dedup_nusantara_ssp',
    'kopi_cc_all-neardup_nusantara_ssp',
    'kopi_cc_all-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_10-dedup_nusantara_ssp',
    'kopi_cc_2021_10-neardup_nusantara_ssp',
    'kopi_cc_2021_10-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_17-raw_nusantara_ssp',
    'kopi_cc_2021_17-dedup_nusantara_ssp',
    'kopi_cc_2021_17-neardup_nusantara_ssp',
    'kopi_cc_2021_17-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_21-raw_nusantara_ssp',
    'kopi_cc_2021_21-dedup_nusantara_ssp',
    'kopi_cc_2021_21-neardup_nusantara_ssp',
    'kopi_cc_2021_21-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_25-raw_nusantara_ssp',
    'kopi_cc_2021_25-dedup_nusantara_ssp',
    'kopi_cc_2021_25-neardup_nusantara_ssp',
    'kopi_cc_2021_25-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_31-raw_nusantara_ssp',
    'kopi_cc_2021_31-dedup_nusantara_ssp',
    'kopi_cc_2021_31-neardup_nusantara_ssp',
    'kopi_cc_2021_31-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_39-raw_nusantara_ssp',
    'kopi_cc_2021_39-dedup_nusantara_ssp',
    'kopi_cc_2021_39-neardup_nusantara_ssp',
    'kopi_cc_2021_39-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_43-raw_nusantara_ssp',
    'kopi_cc_2021_43-dedup_nusantara_ssp',
    'kopi_cc_2021_43-neardup_nusantara_ssp',
    'kopi_cc_2021_43-neardup_clean_nusantara_ssp',
    'kopi_cc_2021_49-dedup_nusantara_ssp',
    'kopi_cc_2021_49-neardup_nusantara_ssp',
    'kopi_cc_2021_49-neardup_clean_nusantara_ssp',
    'kopi_cc_2022_05-raw_nusantara_ssp',
    'kopi_cc_2022_05-dedup_nusantara_ssp',
    'kopi_cc_2022_05-neardup_nusantara_ssp',
    'kopi_cc_2022_05-neardup_clean_nusantara_ssp',
    'kopi_cc_2022_21-raw_nusantara_ssp',
    'kopi_cc_2022_21-dedup_nusantara_ssp',
    'kopi_cc_2022_21-neardup_nusantara_ssp',
    'kopi_cc_2022_21-neardup_clean_nusantara_ssp',
    'kopi_cc_2022_27-raw_nusantara_ssp',
    'kopi_cc_2022_27-dedup_nusantara_ssp',
    'kopi_cc_2022_27-neardup_nusantara_ssp',
    'kopi_cc_2022_27-neardup_clean_nusantara_ssp',
    'kopi_cc_news_2016_source',
    'kopi_cc_news_2017_source',
    'kopi_cc_news_2018_source',
    'kopi_cc_news_2019_source',
    'kopi_cc_news_2020_source',
    'kopi_cc_news_2021_source',
    'kopi_cc_news_2022_source',
    'kopi_cc_news_all_source',
    'kopi_cc_news_2016_nusantara_ssp',
    'kopi_cc_news_2017_nusantara_ssp',
    'kopi_cc_news_2018_nusantara_ssp',
    'kopi_cc_news_2019_nusantara_ssp',
    'kopi_cc_news_2020_nusantara_ssp',
    'kopi_cc_news_2021_nusantara_ssp',
    'kopi_cc_news_2022_nusantara_ssp',
    'kopi_cc_news_all_nusantara_ssp',
    'kopi_nllb_all-raw_source',
    'kopi_nllb_all-dedup_source',
    'kopi_nllb_all-neardup_source',
    'kopi_nllb_ace_Latn-raw_source',
    'kopi_nllb_ace_Latn-dedup_source',
    'kopi_nllb_ace_Latn-neardup_source',
    'kopi_nllb_ban_Latn-raw_source',
    'kopi_nllb_ban_Latn-dedup_source',
    'kopi_nllb_ban_Latn-neardup_source',
    'kopi_nllb_bjn_Latn-raw_source',
    'kopi_nllb_bjn_Latn-dedup_source',
    'kopi_nllb_bjn_Latn-neardup_source',
    'kopi_nllb_ind_Latn-raw_source',
    'kopi_nllb_ind_Latn-dedup_source',
    'kopi_nllb_ind_Latn-neardup_source',
    'kopi_nllb_jav_Latn-raw_source',
    'kopi_nllb_jav_Latn-dedup_source',
    'kopi_nllb_jav_Latn-neardup_source',
    'kopi_nllb_min_Latn-raw_source',
    'kopi_nllb_min_Latn-dedup_source',
    'kopi_nllb_min_Latn-neardup_source',
    'kopi_nllb_sun_Latn-raw_source',
    'kopi_nllb_sun_Latn-dedup_source',
    'kopi_nllb_sun_Latn-neardup_source',
    'kopi_nllb_all-raw_nusantara_ssp',
    'kopi_nllb_all-dedup_nusantara_ssp',
    'kopi_nllb_all-neardup_nusantara_ssp',
    'kopi_nllb_ace_Latn-raw_nusantara_ssp',
    'kopi_nllb_ace_Latn-dedup_nusantara_ssp',
    'kopi_nllb_ace_Latn-neardup_nusantara_ssp',
    'kopi_nllb_ban_Latn-raw_nusantara_ssp',
    'kopi_nllb_ban_Latn-dedup_nusantara_ssp',
    'kopi_nllb_ban_Latn-neardup_nusantara_ssp',
    'kopi_nllb_bjn_Latn-raw_nusantara_ssp',
    'kopi_nllb_bjn_Latn-dedup_nusantara_ssp',
    'kopi_nllb_bjn_Latn-neardup_nusantara_ssp',
    'kopi_nllb_ind_Latn-raw_nusantara_ssp',
    'kopi_nllb_ind_Latn-dedup_nusantara_ssp',
    'kopi_nllb_ind_Latn-neardup_nusantara_ssp',
    'kopi_nllb_jav_Latn-raw_nusantara_ssp',
    'kopi_nllb_jav_Latn-dedup_nusantara_ssp',
    'kopi_nllb_jav_Latn-neardup_nusantara_ssp',
    'kopi_nllb_min_Latn-raw_nusantara_ssp',
    'kopi_nllb_min_Latn-dedup_nusantara_ssp',
    'kopi_nllb_min_Latn-neardup_nusantara_ssp',
    'kopi_nllb_sun_Latn-raw_nusantara_ssp',
    'kopi_nllb_sun_Latn-dedup_nusantara_ssp',
    'kopi_nllb_sun_Latn-neardup_nusantara_ssp'
]

_RESOURCE_CONFIG_NAMES = [
    'inset_lexicon_nusantara_text',
    'kamus_alay_nusantara_t2t'
]

_CURRENTLY_BROKEN_NAMES = [

]

BENCHMARK_DICT = {
    'IndoNLU': [
        'emot_nusantara_text',
        'smsa_nusantara_text',
        'wrete_nusantara_pairs',
        'casa_nusantara_text_multi',
        'hoasa_nusantara_text_multi',
        'facqa_nusantara_qa',
        'indonlu_nergrit_nusantara_seq_label',
        'nerp_nusantara_seq_label',
        'posp_nusantara_seq_label',
        'term_a_nusantara_seq_label',
        'keps_nusantara_seq_label',
        'idn_tagged_corpus_csui_nusantara_seq_label'
    ],
    'IndoNLG': [
        # MT
        'bible_en_id_nusantara_t2t',
        'bible_su_id_nusantara_t2t',
        'bible_jv_id_nusantara_t2t',
        'ted_en_id_nusantara_t2t',
        'indo_general_mt_en_id_nusantara_t2t',
        'news_en_id_nusantara_t2t',        
        # Summarization
        'indosum_fold0_nusantara_t2t',
        'liputan6_canonical_nusantara_t2t',
        'liputan6_xtreme_nusantara_t2t',
        # Chit Chat
        'xpersona_id_nusantara_t2t',
        # QA
        'tydiqa_id_nusantara_qa',
    ],
    'IndoLEM': [
        'indolem_ntp_nusantara_pairs',
        'indolem_sentiment_nusantara_text',
        'indolem_ner_ugm_fold0_nusantara_seq_label',
        'indolem_ner_ugm_fold1_nusantara_seq_label',
        'indolem_ner_ugm_fold2_nusantara_seq_label',
        'indolem_ner_ugm_fold3_nusantara_seq_label',
        'indolem_ner_ugm_fold4_nusantara_seq_label',
        'indolem_ud_id_gsd_nusantara_kb',
        'indolem_ud_id_pud_nusantara_kb',
        'indolem_tweet_ordering_nusantara_seq_label',
        'indolem_nerui_fold0_nusantara_seq_label',
        'indolem_nerui_fold1_nusantara_seq_label',
        'indolem_nerui_fold2_nusantara_seq_label',
        'indolem_nerui_fold3_nusantara_seq_label',
        'indolem_nerui_fold4_nusantara_seq_label'
    ],
    'NusaX': [
        # Ind - XXX
        'nusax_mt_ind_ace_nusantara_t2t',
        'nusax_mt_ind_ban_nusantara_t2t',
        'nusax_mt_ind_bjn_nusantara_t2t',
        'nusax_mt_ind_bug_nusantara_t2t',
        'nusax_mt_ind_eng_nusantara_t2t',
        'nusax_mt_ind_jav_nusantara_t2t',
        'nusax_mt_ind_mad_nusantara_t2t',
        'nusax_mt_ind_min_nusantara_t2t',
        'nusax_mt_ind_nij_nusantara_t2t',
        'nusax_mt_ind_sun_nusantara_t2t',
        'nusax_mt_ind_bbc_nusantara_t2t',
        
        # XXX - Ind
        'nusax_mt_ace_ind_nusantara_t2t',
        'nusax_mt_ban_ind_nusantara_t2t',
        'nusax_mt_bjn_ind_nusantara_t2t',
        'nusax_mt_bug_ind_nusantara_t2t',
        'nusax_mt_eng_ind_nusantara_t2t',
        'nusax_mt_jav_ind_nusantara_t2t',
        'nusax_mt_mad_ind_nusantara_t2t',
        'nusax_mt_min_ind_nusantara_t2t',
        'nusax_mt_nij_ind_nusantara_t2t',
        'nusax_mt_sun_ind_nusantara_t2t',
        'nusax_mt_bbc_ind_nusantara_t2t',    
    ],    
    'NusaNLU': [
        'emot_nusantara_text',
        'emotcmt_nusantara_text',
        'emotion_id_opinion_nusantara_text',
        'id_abusive_nusantara_text',
        'id_google_play_review_nusantara_text',
        'id_google_play_review_posneg_nusantara_text',
        'id_hatespeech_nusantara_text',
        'imdb_jv_nusantara_text',
        'indolem_sentiment_nusantara_text',
        'jadi_ide_nusantara_text',
        'nusax_senti_ace_nusantara_text',
        'nusax_senti_ban_nusantara_text',
        'nusax_senti_bjn_nusantara_text',
        'nusax_senti_bug_nusantara_text',
        'nusax_senti_eng_nusantara_text',
        'nusax_senti_ind_nusantara_text',
        'nusax_senti_jav_nusantara_text',
        'nusax_senti_mad_nusantara_text',
        'nusax_senti_min_nusantara_text',
        'nusax_senti_nij_nusantara_text',
        'nusax_senti_sun_nusantara_text',
        'nusax_senti_bbc_nusantara_text',
        'sentiment_nathasa_review_nusantara_text',
        'smsa_nusantara_text',    
        'indolem_ntp_nusantara_pairs',
        'indonli_nusantara_pairs',
        'code_mixed_jv_id_jv_nusantara_text',
        'code_mixed_jv_id_id_nusantara_text',
        'id_am2ico_nusantara_pairs',
        'id_abusive_news_comment_nusantara_text',
        'id_hoax_news_nusantara_text',
        'id_hsd_nofaaulia_nusantara_text',
        'id_stance_nusantara_pairs',
        'indo_law_nusantara_text',
        'indotacos_nusantara_text',
        'karonese_sentiment_nusantara_text',
        'su_emot_nusantara_text',
        'wrete_nusantara_pairs',
        'id_short_answer_grading_nusantara_pairs'
    ],    
    'NusaNLG': [
        'bible_en_id_nusantara_t2t',
        'bible_jv_id_nusantara_t2t',
        'bible_su_id_nusantara_t2t',
        'id_panl_bppt_nusantara_t2t',
        'indo_general_mt_en_id_nusantara_t2t',
        'indo_religious_mt_en_id_nusantara_t2t',
        'minangnlp_mt_nusantara_t2t',
        'news_en_id_nusantara_t2t',
        'nusax_mt_ace_ind_nusantara_t2t',
        'nusax_mt_ban_ind_nusantara_t2t',
        'nusax_mt_bjn_ind_nusantara_t2t',
        'nusax_mt_bug_ind_nusantara_t2t',
        'nusax_mt_eng_ind_nusantara_t2t',
        'nusax_mt_ind_ace_nusantara_t2t',
        'nusax_mt_ind_ban_nusantara_t2t',
        'nusax_mt_ind_bjn_nusantara_t2t',
        'nusax_mt_ind_bug_nusantara_t2t',
        'nusax_mt_ind_eng_nusantara_t2t',
        'nusax_mt_ind_jav_nusantara_t2t',
        'nusax_mt_ind_mad_nusantara_t2t',
        'nusax_mt_ind_min_nusantara_t2t',
        'nusax_mt_ind_nij_nusantara_t2t',
        'nusax_mt_ind_sun_nusantara_t2t',
        'nusax_mt_ind_bbc_nusantara_t2t',
        'nusax_mt_jav_ind_nusantara_t2t',
        'nusax_mt_mad_ind_nusantara_t2t',
        'nusax_mt_min_ind_nusantara_t2t',
        'nusax_mt_nij_ind_nusantara_t2t',
        'nusax_mt_sun_ind_nusantara_t2t',
        'nusax_mt_bbc_ind_nusantara_t2t',
        'parallel_su_id_nusantara_t2t',
        'ted_en_id_nusantara_t2t',
        'ud_id_csui_nusantara_t2t',
        'korpus_nusantara_ind_jav_nusantara_t2t',
        'korpus_nusantara_ind_xdy_nusantara_t2t',
        'korpus_nusantara_ind_bug_nusantara_t2t',
        'korpus_nusantara_ind_sun_nusantara_t2t',
        'korpus_nusantara_ind_mad_nusantara_t2t',
        'korpus_nusantara_ind_bjn_nusantara_t2t',
        'korpus_nusantara_ind_bbc_nusantara_t2t',
        'korpus_nusantara_ind_khek_nusantara_t2t',
        'korpus_nusantara_ind_msa_nusantara_t2t',
        'korpus_nusantara_ind_min_nusantara_t2t',
        'korpus_nusantara_ind_tiociu_nusantara_t2t',
        'korpus_nusantara_jav_ind_nusantara_t2t',
        'korpus_nusantara_xdy_ind_nusantara_t2t',
        'korpus_nusantara_bug_ind_nusantara_t2t',
        'korpus_nusantara_sun_ind_nusantara_t2t',
        'korpus_nusantara_mad_ind_nusantara_t2t',
        'korpus_nusantara_bjn_ind_nusantara_t2t',
        'korpus_nusantara_bbc_ind_nusantara_t2t',
        'korpus_nusantara_khek_ind_nusantara_t2t',
        'korpus_nusantara_msa_ind_nusantara_t2t',
        'korpus_nusantara_min_ind_nusantara_t2t',
        'korpus_nusantara_tiociu_ind_nusantara_t2t',
        'indosum_fold0_nusantara_t2t',
        'liputan6_canonical_nusantara_t2t',
        'xl_sum_nusantara_t2t',
        'id_qqp_nusantara_t2t',
        'multilexnorm_nusantara_t2t',
        'paracotta_id_nusantara_t2t',
        'stif_indonesia_nusantara_t2t',        
        'xpersona_id_nusantara_t2t'
        'facqa_nusantara_qa',
        'idk_mrc_nusantara_qa',
        'tydiqa_id_nusantara_qa'
    ],
    'NusaASR': [
        # Ind
        'indspeech_digit_cdsr_nusantara_sptext',
        'indspeech_news_lvcsr_nusantara_sptext',
        'indspeech_teldialog_lvcsr_nusantara_sptext',
        'indspeech_teldialog_svcsr_nusantara_sptext',
        'librivox_indonesia_ind_nusantara_sptext',
        'titml_idn_nusantara_sptext'
        # Sun
        'indspeech_newstra_ethnicsr_nooverlap_sun_nusantara_sptext',
        'indspeech_news_ethnicsr_su_nooverlap_nusantara_sptext',
        'librivox_indonesia_sun_nusantara_sptext',
        'su_id_asr_nusantara_sptext',
        # Jav
        'indspeech_newstra_ethnicsr_nooverlap_jav_nusantara_sptext',
        'indspeech_news_ethnicsr_jv_nooverlap_nusantara_sptext',
        'librivox_indonesia_jav_nusantara_sptext',
        'jv_id_asr_nusantara_sptext',
        # Ban
        'indspeech_newstra_ethnicsr_nooverlap_ban_nusantara_sptext',
        'librivox_indonesia_ban_nusantara_sptext',
        # Btk
        'indspeech_newstra_ethnicsr_nooverlap_btk_nusantara_sptext',
        # Ace
        'librivox_indonesia_ace_nusantara_sptext',
        # Bug
        'librivox_indonesia_bug_nusantara_sptext',
        # Min
        'librivox_indonesia_min_nusantara_sptext',
    ],
#     'NusaTranslation': [
    
#     ],
#     'NusaParagraph': [
    
#     ],
#     'NusaWrites': [
    
#     ],
}

@dataclass
class NusantaraMetadata:
    """Metadata for one config of a dataset."""

    script: pathlib.Path
    dataset_name: str
    tasks: List[Tasks]
    languages: List[str]
    config: NusantaraConfig
    is_local: bool
    is_nusantara_schema: bool
    nusantara_schema_caps: Optional[str]
    is_large: bool
    is_resource: bool
    is_default: bool
    is_broken: bool
    nusantara_version: str
    source_version: str
    citation: str
    description: str
    homepage: str
    license: str

    _ds_module: datasets.load.DatasetModule = field(repr=False)
    _py_module: ModuleType = field(repr=False)
    _ds_cls: type = field(repr=False)

    def get_load_dataset_kwargs(
        self,
        **extra_load_dataset_kwargs,
    ):
        return {
            "path": self.script,
            "name": self.config.name,
            **extra_load_dataset_kwargs,
        }

    def load_dataset(
        self,
        **extra_load_dataset_kwargs,
    ):
        return datasets.load_dataset(
            path=self.script,
            name=self.config.name,
            **extra_load_dataset_kwargs,
        )

    def get_metadata(self, **extra_load_dataset_kwargs):
        if not self.is_nusantara_schema:
            raise ValueError("only supported for nusantara schemas")
        dsd = self.load_dataset(**extra_load_dataset_kwargs)
        split_metas = {}
        for split, ds in dsd.items():
            meta = SCHEMA_TO_METADATA_CLS[self.config.schema].from_dataset(ds)
            split_metas[split] = meta
        return split_metas


def default_is_keeper(metadata: NusantaraMetadata) -> bool:
    return not metadata.is_large and not metadata.is_resource and metadata.is_nusantara_schema

class NusantaraConfigHelper:
    """
    Handles creating and filtering NusantaraMetadata instances.
    """

    def __init__(
        self,
        helpers: Optional[Iterable[NusantaraMetadata]] = None,
        keep_broken: bool = False,
    ):

        path_to_here = pathlib.Path(__file__).parent.absolute()
        self.path_to_nusadatasets = (path_to_here / "nusa_datasets").resolve()
        self.dataloader_scripts = sorted(
            self.path_to_nusadatasets.glob(os.path.join("*", "*.py"))
        )
        self.dataloader_scripts = [
            el for el in self.dataloader_scripts if el.name != "__init__.py"
        ]

        # if helpers are passed in, just attach and go
        if helpers is not None:
            if keep_broken:
                self._helpers = helpers
            else:
                self._helpers = [helper for helper in helpers if not helper.is_broken]
            return

        # otherwise, create all helpers available in package
        helpers = []
        for dataloader_script in self.dataloader_scripts:
            dataset_name = dataloader_script.stem
            py_module = SourceFileLoader(
                dataset_name, dataloader_script.as_posix()
            ).load_module()
            ds_module = datasets.load.dataset_module_factory(
                dataloader_script.as_posix()
            )
            ds_cls = datasets.load.import_main_class(ds_module.module_path)

            for config in ds_cls.BUILDER_CONFIGS:

                is_nusantara_schema = config.schema.startswith("nusantara")
                if is_nusantara_schema:
                    nusantara_schema_caps = '_'.join(config.schema.split("_")[1:]).upper()
                    tasks = SCHEMA_TO_TASKS[nusantara_schema_caps] & set(
                        py_module._SUPPORTED_TASKS
                    )
                else:
                    tasks = py_module._SUPPORTED_TASKS
                    nusantara_schema_caps = None

                helpers.append(
                    NusantaraMetadata(
                        script=dataloader_script.as_posix(),
                        dataset_name=dataset_name,
                        tasks=tasks,
                        languages=py_module._LANGUAGES,
                        config=config,
                        is_local=py_module._LOCAL,
                        is_nusantara_schema=is_nusantara_schema,
                        nusantara_schema_caps=nusantara_schema_caps,
                        is_large=config.name in _LARGE_CONFIG_NAMES,
                        is_resource=config.name in _RESOURCE_CONFIG_NAMES,
                        is_default=config.name == ds_cls.DEFAULT_CONFIG_NAME,
                        is_broken=config.name in _CURRENTLY_BROKEN_NAMES,
                        nusantara_version=py_module._NUSANTARA_VERSION,
                        source_version=py_module._SOURCE_VERSION,
                        citation=py_module._CITATION,
                        description=py_module._DESCRIPTION,
                        homepage=py_module._HOMEPAGE,
                        license=py_module._LICENSE,
                        _ds_module=ds_module,
                        _py_module=py_module,
                        _ds_cls=ds_cls,
                    )
                )

        if keep_broken:
            self._helpers = helpers
        else:
            self._helpers = [helper for helper in helpers if not helper.is_broken]

    @property
    def available_dataset_names(self) -> List[str]:
        return sorted(list(set([helper.dataset_name for helper in self])))

    def for_dataset(self, dataset_name: str) -> "NusantaraConfigHelper":
        helpers = [helper for helper in self if helper.dataset_name == dataset_name]
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.dataset_name = {dataset_name}")
        return NusantaraConfigHelper(helpers=helpers)

    def for_config_name(self, config_name: str) -> "NusantaraMetadata":
        helpers = [helper for helper in self if helper.config.name == config_name]
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.config.name = {config_name}")
        if len(helpers) > 1:
            raise ValueError(
                f"multiple helpers with helper.config.name = {config_name}"
            )
        return helpers[0]

    def default_for_dataset(self, dataset_name: str) -> "NusantaraMetadata":
        helpers = [
            helper
            for helper in self
            if helper.is_default and helper.dataset_name == dataset_name
        ]
        assert len(helpers) == 1
        return helpers[0]

    def filtered(
        self, is_keeper: Callable[[NusantaraMetadata], bool]
    ) -> "NusantaraConfigHelper":
        """Return dataset config helpers that match is_keeper."""
        return NusantaraConfigHelper(
            helpers=[helper for helper in self if is_keeper(helper)]
        )

    def __repr__(self):
        return "\n\n".join([helper.__repr__() for helper in self])

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for helper in self._helpers:
            yield helper

    def __len__(self):
        return len(self._helpers)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return NusantaraConfigHelper(
                helpers=[self._helpers[ii] for ii in range(start, stop, step)]
            )
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return self._helpers[key]
        else:
            raise TypeError("Invalid argument type.")
    
    def list_datasets(self, with_config=False):
        name_to_schema = {}
        for helper in self:
            if helper.dataset_name not in name_to_schema:
                name_to_schema[helper.dataset_name] = []
            name_to_schema[helper.dataset_name].append(helper.config.name)
        if not with_config:
            return list(name_to_schema.keys())
        else:
            return name_to_schema
    
    def load_dataset(self, dataset_name, schema='nusantara'):
        try:        
            return [
                helper.load_dataset()
                for helper in self.filtered(
                    lambda x: (
                        (dataset_name == x.dataset_name) and 
                        (x.is_nusantara_schema if schema == 'nusantara' else not x.is_nusantara_schema)
                    )
                )][0]
        except:
            raise ValueError(f"Couldn't find dataset with name=`{dataset_name}` and schema=`{schema}`")

    def load_datasets(self, dataset_names, schema='nusantara'):
        return {
            helper.config.name: helper.load_dataset()
            for helper in self.filtered(
                lambda x: (
                    (x.dataset_name in dataset_names) and 
                    (x.is_nusantara_schema if schema == 'nusantara' else not x.is_nusantara_schema)
                )
            )
       }
        
    def list_benchmarks(self):
        return list(BENCHMARK_DICT.keys())

    def load_benchmark(self, benchmark_name):
        return {
            helper.config.name: helper.load_dataset()
            for helper in self.filtered(
                lambda x: (
                    x.config.name in BENCHMARK_DICT[benchmark_name]
                )
            )
        }

# Metadata Helper
@dataclass
class MetaDict:
    data: dict = None
    
class NusantaraMetadataHelper:
    """
    Handles creating and filtering NusantaraMetadata instances.
    """

    def __init__(
        self,
        meta_df: Optional[pd.DataFrame] = None,
        keep_broken: bool = False
    ):
        # Load Config Helper
        self._conhelps = NusantaraConfigHelper()
        
        # if meta_df are passed in, just attach and go
        if meta_df is not None:
            if keep_broken:
                self._meta_df = meta_df
            else:
                self._meta_df = meta_df[~meta_df.is_broken]
            return
        
        # Load Metadata
        self._meta_df = pd.read_csv('https://docs.google.com/spreadsheets/d/17o83IvWxmtGLYridZis0nEprHhsZIMeFtHGtXV35h6M/export?format=csv&gid=879729812', skiprows=1)
        self._meta_df = self._meta_df[self._meta_df['Implemented'] != 0].rename({
            'No.': 'id', 'Name': 'name', 'Subsets': 'subsets', 'Link': 'source_link', 'Description': 'description',
            'HF Link': 'hf_link', 'License': 'license', 'Year': 'year', 'Collection Style': 'collection_style',
            'Language': 'language', 'Dialect': 'dialect', 'Domain': 'domain', 'Form': 'modality', 'Tasks': 'tasks',
            'Volume': 'volume', 'Unit': 'unit', 'Ethical Risks': 'ethical_risk', 'Provider': 'provider',
            'Paper Title': 'paper_title', 'Paper Link': 'paper_link', 'Access': 'access', 'Derived From': 'derived_from', 
            'Test Split': 'is_splitted', 'Notes': 'notes', 'Dataloader': 'dataloader', 'Implemented': 'implemented'
        }, axis=1)
        self._meta_df['is_splitted'] = self._meta_df['is_splitted'].apply(lambda x: True if x =='Yes' else False)

        # Merge Metadata with Config
        name_to_meta_map = {}
        for cfg_meta in self._conhelps:
            # Assign metadata to meta dataframe
            self._meta_df.loc[self._meta_df.dataloader == cfg_meta.dataset_name, [
                'is_large', 'is_resource', 'is_default', 'is_broken',
                'is_local', 'citation', 'license', 'homepage', 'tasks'
            ]] = [
                cfg_meta.is_large, cfg_meta.is_resource, cfg_meta.is_default, cfg_meta.is_broken, 
                cfg_meta.is_local, cfg_meta.citation, cfg_meta.license, cfg_meta.homepage, '|'.join([task.value for task in cfg_meta.tasks])
            ]

            if cfg_meta.dataset_name not in name_to_meta_map:
                name_to_meta_map[cfg_meta.dataset_name] = {}
            if cfg_meta.config.schema not in name_to_meta_map[cfg_meta.dataset_name]:
                name_to_meta_map[cfg_meta.dataset_name][cfg_meta.config.schema] = []
            name_to_meta_map[cfg_meta.dataset_name][cfg_meta.config.schema].append(cfg_meta)
        
        self._meta_df = self._meta_df.fillna(False)
        for dset_name in name_to_meta_map.keys():
            self._meta_df.loc[self._meta_df.dataloader == dset_name, 'metadata'] = MetaDict(data=name_to_meta_map[dset_name])

        if not keep_broken:
            self._meta_df = self._meta_df[~self._meta_df.is_broken]
    
    def filtered(
        self, is_keeper: Callable[[], bool]
    ) -> "NusantaraConfigHelper":
        """Return dataset config helpers that match is_keeper."""
        meta_df = self._meta_df[self._meta_df.apply(is_keeper, axis=1, reduce=True)]
        return NusantaraMetadataHelper(meta_df=meta_df)

    def filter_and_load(
        self, is_keeper: Callable[[], bool]
    ) -> "Dict<str, Dataset>":
        """Return dataset that match is_keeper."""
        filtered_helper = self.filtered(is_keeper)
        for metas in filtered_helper._meta_df.metadata:
            if schema in metas.data:
                for meta in metas.data[schema]:
                    if len(meta.languages) > 1:
                        if lang in meta.config.name:
                            datasets[meta.config.name] = meta.load_dataset()
                    else:
                        datasets[meta.config.name] = meta.load_dataset()
                
    @property
    def available_dataset_names(self) -> List[str]:
        return sorted(self._meta_df.name)

#     def __repr__(self):
#         return self._meta_df.to_string()

#     def __str__(self):
#         return self.__repr__()

    def __iter__(self):
        for row in self._meta_df.iterrows():
            yield row

    def __len__(self):
        return len(self._meta_df)
    
###
# NusaCrowd Interface
###

def list_datasets(with_config=False):
    conhelps = NusantaraConfigHelper()
    return conhelps.list_datasets(with_config=with_config)

def load_dataset(dataset_name, schema='nusantara'):
    conhelps = NusantaraConfigHelper()
    return conhelps.load_dataset(dataset_name=dataset_name, schema=schema)

def load_datasets(dataset_names, schema='nusantara'):
    conhelps = NusantaraConfigHelper()
    return conhelps.load_datasets(dataset_names=dataset_names, schema=schema)

def list_benchmarks():
    conhelps = NusantaraConfigHelper()
    return conhelps.list_benchmarks()

def load_benchmark(benchmark_name):
    conhelps = NusantaraConfigHelper()
    return conhelps.load_benchmark(benchmark_name=benchmark_name)

if __name__ == "__main__":
    print(f'LIST DATASETS')
    dset_names = list_datasets()
    print(dset_names[:10])
    print()

    print(f'LOAD DATASET `{dset_names[1]}`')
    dset = load_dataset(dset_names[1])
    print(dset)
    print()
    
    print(f'LOAD DATASETS [{dset_names[1:4]}]')
    dsets = load_datasets(dset_names[1:4])
    print(dsets)
    print()
    
    print(f'LIST BENCHMARKS')
    benchmark_names = list_benchmarks()
    print(benchmark_names[:3])
    print()

    print(f'LOAD BENCHMARK `{benchmark_names[0]}`')
    benchmark_dsets = load_benchmark(benchmark_names[0])
    print(benchmark_dsets)
    print()
