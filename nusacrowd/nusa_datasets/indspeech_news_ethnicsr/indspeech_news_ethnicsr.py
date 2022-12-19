from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import json
import os

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks
from zipfile import ZipFile

_CITATION = """\
@inproceedings{sani-cocosda-2012,
    title = "Towards Language Preservation: Preliminary Collection and Vowel Analysis of {I}ndonesian Ethnic Speech Data",
    author = "Sani, Auliya and Sakti, Sakriani and Neubig, Graham and Toda, Tomoki and Mulyanto, Adi and Nakamura, Satoshi",
    booktitle = "Proc. Oriental COCOSDA",
    year = "2012",
    pages = "118--122"
    address = "Macau, China"
}
"""

_LOCAL = False
_LANGUAGES = ["sun", "jav"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indspeech_news_ethnicsr"

_DESCRIPTION = """
INDspeech_NEWS_EthnicSR is a collection of Indonesian ethnic speech corpora for Javanese and Sundanese for Indonesian ethnic speech recognition. It was developed in 2012 by the Nara Institute of Science and Technology (NAIST, Japan) in collaboration with the Bandung Institute of Technology (ITB, Indonesia) [Sani et al., 2012].
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_news_ethnicsr"

_LICENSE = "CC-BY-NC-SA 4.0"

_URLS = {
    _DATASETNAME: "https://github.com/s-sakti/data_indsp_news_ethnicsr/archive/refs/heads/main.zip",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndSpeechNewsEthnicSR(datasets.GeneratorBasedBuilder):
    """INDspeech_NEWS_EthnicSR is a collection of Indonesian ethnic speech corpora for Javanese and Sundanese for Indonesian ethnic speech recognition. It was developed in 2012 by the Nara Institute of Science and Technology (NAIST, Japan) in collaboration with the Bandung Institute of Technology (ITB, Indonesia) [Sani et al., 2012]."""


    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    
    BUILDER_CONFIGS = []
    
    for fold_id in ["overlap", "nooverlap"]:
        for fold_name in ['jv', "su"]:
            BUILDER_CONFIGS.extend(
                [NusantaraConfig(
                    name=f"indspeech_news_ethnicsr_{fold_name}_{fold_id}_source",
                    version=_SOURCE_VERSION,
                    description="indspeech_news_ethnicsr source schema",
                    schema="source",
                    subset_id=f"indspeech_news_ethnicsr_{fold_name}_{fold_id}"
                ),
                NusantaraConfig(
                    name=f"indspeech_news_ethnicsr_{fold_name}_{fold_id}_nusantara_sptext",
                    version=_SOURCE_VERSION,
                    description="indspeech_news_ethnicsr Nusantara schema",
                    schema="nusantara_sptext",
                    subset_id=f"indspeech_news_ethnicsr_{fold_name}_{fold_id}"
                ),]
            ) 
            
    DEFAULT_CONFIG_NAME = "indspeech_news_ethnicsr_jv_nooverlap_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
               {
                   "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
               }
            )

        elif self.config.schema == "nusantara_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )


    def _get_fold_name_id(self):
        subset_id = self.config.subset_id
        subset_id_list = subset_id.split('_')
        fold_name = subset_id_list[-2]
        fold_id = subset_id_list[-1]
        if fold_id == "overlap":
            fold_id = 1
        elif fold_id == "nooverlap":
            fold_id = 2
        return fold_name, fold_id
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        fold_name, fold_id = self._get_fold_name_id()
        if fold_name == 'su':
            fold_name1 = "Sunda"
            fold_name2 = 'Snd'
            
        else:
            fold_name1 = 'Jawa'
            fold_name2 = 'Jaw'
        
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
#         print("data_dir", data_dir)
        text_file = os.path.join(data_dir, f"data_indsp_news_ethnicsr-main/{fold_name1}/text/transcript.txt")
        wav_folder = os.path.join(data_dir, f"data_indsp_news_ethnicsr-main/{fold_name1}/speech/16kHz/")
        train_list = os.path.join(data_dir, f"data_indsp_news_ethnicsr-main/{fold_name1}/lst/dataset{fold_id}_train_news_{fold_name2}.lst")
        test_list = os.path.join(data_dir, f"data_indsp_news_ethnicsr-main/{fold_name1}/lst/dataset{fold_id}_test_news_{fold_name2}.lst")
        
        #unzip        
        for speaker_id in range(1, 11):
            speaker_id = "%03d" % (speaker_id)
            zip_file = os.path.join(wav_folder, f"{fold_name2}{speaker_id}.zip")
            out_folder = os.path.join(wav_folder, f"{fold_name2}{speaker_id}")
            if not os.path.exists(out_folder):
                with ZipFile(zip_file, 'r') as f:
                    f.extractall(out_folder)
                    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "wav_folder": wav_folder,
                    "text_path": text_file,
                    "split": "train",
                    "fold_name": fold_name,
                    "file_list": train_list,
                },
            ),
            
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "wav_folder": wav_folder,
                    "text_path": text_file,
                    "split": "test",
                    "fold_name": fold_name,
                    "file_list": test_list,
                },
            )
        ]
    

    def _generate_examples(self, wav_folder: Path, text_path: Path, split: str, fold_name: str, file_list: Path) -> Tuple[int, Dict]:
        if fold_name == 'su':
            fold_name2 = 'Snd'
        else:
            fold_name2 = 'Jaw'
            
        id2text = {}
        with open(text_path, "r", encoding='unicode_escape') as f:
            for text_idx, line in enumerate(f.readlines()):
                id2text.update({"%04d" % (text_idx + 1):line.strip()})
                
        
        wave_list = []
        with open(file_list) as f:
            for l in f.readlines():
                audio_id = l.strip()[:-4]
                speaker_id = audio_id.split('_')[0][-3:]
                text_id = audio_id.split('_')[-1]
                text = id2text[text_id]
                
                wav_path = os.path.join(wav_folder, audio_id.split('_')[0], l.strip())
                if not os.path.exists(wav_path):
                    print('no exisit wav_path', wav_path)
                assert os.path.exists(wav_path)
                
                if self.config.schema == "source":
                    ex = {
                        "id": audio_id,
                        "speaker_id": speaker_id,
                        "path": wav_path,
                        "audio": wav_path,
                        "text": text,
                    }
                    yield audio_id, ex

                elif self.config.schema == "nusantara_sptext":
                    ex = {
                        "id": audio_id,
                        "speaker_id": speaker_id,
                        "path": wav_path,
                        "audio": wav_path,
                        "text": text,
                        "metadata": {
                            "speaker_age": None,
                            "speaker_gender": audio_id.split("_")[1],
                        }
                    }
                    yield audio_id, ex
