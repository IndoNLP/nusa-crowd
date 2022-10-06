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
@inproceedings{sakti-tcast-2008,
    title = "Development of {I}ndonesian Large Vocabulary Continuous Speech Recognition System within {A-STAR} Project",
    author = "Sakti, Sakriani and Kelana, Eka and Riza, Hammam and Sakai, Shinsuke and Markov, Konstantin and Nakamura, Satoshi",
    booktitle = "Proc. IJCNLP Workshop on Technologies and Corpora for Asia-Pacific Speech Translation (TCAST)",
    year = "2008",
    pages = "19--24"
    address = "Hyderabad, India"
}


@inproceedings{sakti-icslp-2004,
    title = "Indonesian Speech Recognition for Hearing and Speaking Impaired People",
    author = "Sakti, Sakriani and Hutagaol, Paulus and Arman, Arry Akhmad and Nakamura, Satoshi",
    booktitle = "Proc. International Conference on Spoken Language Processing (INTERSPEECH - ICSLP)",
    year = "2004",
    pages = "1037--1040"
    address = "Jeju Island, Korea"
}

@article{sakti-s2st-csl-2013,
    title = "{A-STAR}: Toward Tranlating Asian Spoken Languages",
    author = "Sakti, Sakriani and Paul, Michael and Finch, Andrew and Sakai, Shinsuke and Thang, Tat Vu, and Kimura, Noriyuki 
    and Hori, Chiori and Sumita, Eiichiro and Nakamura, Satoshi and Park, Jun and Wutiwiwatchai, Chai and Xu, Bo and Riza, Hammam 
    and Arora, Karunesh and Luong, Chi Mai and Li, Haizhou",
    journal = "Special issue on Speech-to-Speech Translation, Computer Speech and Language Journal",
    volume = "27",
    number ="2",
    pages = "509--527",
    year = "2013",
    publisher = "Elsevier"
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_DATASETNAME = "indspeech_teldialog_lvcsr"

_DESCRIPTION = """
INDspeech_TELDIALOG_LVCSR is one of the first Indonesian speech datasets for large vocabulary continuous speech recognition (LVCSR) based on telephon application. R&D Division of PT Telekomunikasi Indonesia developed the data in 2005-2006, in collaboration with Advanced Telecommunication Research Institute International (ATR) Japan, as the continuation of the Asia-Pacific Telecommunity (APT) project [Sakti et al., 2004]. It has also been successfully used for developing Indonesian LVCSR in the Asian speech translation advanced research (A-STAR) project [Sakti et al., 2013].
"""

_HOMEPAGE = "https://github.com/s-sakti/data_indsp_teldialog_lvcsr"

_LICENSE = "CC-BY-NC-SA 4.0"


URL_TEMPLATE = {
    "lst": "https://raw.githubusercontent.com/s-sakti/data_indsp_teldialog_lvcsr/main/lst/",  # transcript.lst
    "speech": "https://github.com/s-sakti/data_indsp_teldialog_lvcsr/raw/main/speech/",  # Ind3/Ind304.zip~Ind400.zip
    "text": "https://github.com/s-sakti/data_indsp_teldialog_lvcsr/raw/main/text/",  # all_transcript.zip
}

_URLS = {
    "lst_spk_Ind": [URL_TEMPLATE["lst"] + "spk_Ind" + str(n) + ".lst" for n in range(0, 4)],
    "lst_spk_all": URL_TEMPLATE["lst"] + "spk_all.lst",
    "lst_spk_test": URL_TEMPLATE["lst"] + "spk_test.lst",
    "lst_spk_train": URL_TEMPLATE["lst"] + "spk_train.lst",
    "lst_transcript": URL_TEMPLATE["lst"] + "transcript.lst",
    "speech_Ind": [URL_TEMPLATE["speech"] + "Ind" + str(n) + "/Ind" + str(p).zfill(3) + ".zip" for n in range(0, 4) for p in range(n * 100 + 1, n * 100 + 101)],
    "transcript_all": URL_TEMPLATE["text"] + "all_transcript.zip",
    "transcript_spk": URL_TEMPLATE["text"] + "spk_transcript.zip",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_NUSANTARA_VERSION = "1.0.0"


class IndSpeechTelDialLVCSR(datasets.GeneratorBasedBuilder):
    """INDspeech_TELDIALOG_LVCSR is one of the first Indonesian speech datasets for large vocabulary continuous speech recognition (LVCSR) based on telephon application. R&D Division of PT Telekomunikasi Indonesia developed the data in 2005-2006, in collaboration with Advanced Telecommunication Research Institute International (ATR) Japan, as the continuation of the Asia-Pacific Telecommunity (APT) project [Sakti et al., 2004]. It has also been successfully used for developing Indonesian LVCSR in the Asian speech translation advanced research (A-STAR) project [Sakti et al., 2013]."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    NUSANTARA_VERSION = datasets.Version(_NUSANTARA_VERSION)
    
    BUILDER_CONFIGS = [
        NusantaraConfig(
            name=f"indspeech_teldialog_lvcsr_source",
            version=_SOURCE_VERSION,
            description="indspeech_teldialog_lvcsr source schema",
            schema="source",
            subset_id=f"indspeech_teldialog_lvcsr"
        ),
        NusantaraConfig(
            name=f"indspeech_teldialog_lvcsr_nusantara_sptext",
            version=_SOURCE_VERSION,
            description="indspeech_teldialog_lvcsr Nusantara schema",
            schema="nusantara_sptext",
            subset_id=f"indspeech_teldialog_lvcsr"
        ),]
            
    DEFAULT_CONFIG_NAME = "indspeech_teldialog_lvcsr_source"

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

    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        
        audio_files_dir = []
        for aud_url in _URLS["speech_Ind"]:
            onespeaker_folder = dl_manager.download_and_extract(aud_url)
            audio_files_dir.append(Path(os.path.join(onespeaker_folder, aud_url.split("/")[-1][:-4])))
            
        text_path = Path(dl_manager.download_and_extract(_URLS["lst_transcript"]))
        speak_list = Path(dl_manager.download_and_extract(_URLS["lst_spk_all"]))
        train_list = Path(dl_manager.download_and_extract(_URLS["lst_spk_train"]))
        test_list = Path(dl_manager.download_and_extract(_URLS["lst_spk_test"]))
        
        
          
        speaker_num2id = {}
        with open(speak_list) as f:
            for l in f.readlines():
                l = l.strip()
                speaker_num2id.update({l.split("_")[0]: l})
         
            
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "audio_files_dir": audio_files_dir,
                    "text_path": text_path,
                    "split": "train",
                    "file_list": train_list,
                    "speaker_num2id": speaker_num2id
                },
            ),
            
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_files_dir": audio_files_dir,
                    "text_path": text_path,
                    "split": "test",
                    "file_list": test_list,
                    "speaker_num2id": speaker_num2id
                },
            )
        ]
    

    def _generate_examples(self, audio_files_dir: List, text_path: Path, split: str, file_list: Path, speaker_num2id: Dict) -> Tuple[int, Dict]:
        speaker_nums = []
        with open(file_list) as f:
            for l in f.readlines():
                speaker_nums.append(l.strip())
                
                
        sentid = {}
        with open(text_path) as f:
            for i, l in enumerate(f.readlines()):
                sentid.update({"appl_"+"%04d" % i: l.strip()})
                
                
        for wav_one_speaker_folder in audio_files_dir: #XXXX/Ind0/Ind001
            if wav_one_speaker_folder.name in speaker_nums:
                speaker_num = wav_one_speaker_folder.name #Ind001
                speaker_id = speaker_num2id[speaker_num] #Ind001_F_B

                for wave_file in os.listdir(wav_one_speaker_folder):
                    audio_id = wave_file[:-4]
                    sentence_id = "appl_"+wave_file[:-4].split('_')[-1]
                    text = sentid[sentence_id]
                    wav_path = os.path.join(wav_one_speaker_folder, wave_file)

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
                                "speaker_gender": speaker_id.split("_")[1],
                            }
                        }
                        yield audio_id, ex
