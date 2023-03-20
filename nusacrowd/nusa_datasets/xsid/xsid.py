from pathlib import Path
from typing import List

import datasets

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{van-der-goot-etal-2020-cross,
      title={From Masked-Language Modeling to Translation: Non-{E}nglish Auxiliary Tasks Improve Zero-shot Spoken Language Understanding},
      author={van der Goot, Rob and Sharaf, Ibrahim and Imankulova, Aizhan and {\"U}st{\"u}n, Ahmet and Stepanovic, Marija and Ramponi, Alan and Khairunnisa, Siti Oryza and Komachi, Mamoru and Plank, Barbara},
    booktitle = "Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics"
}
"""
_DATASETNAME = "xsid"
_DESCRIPTION = """\
XSID is a new benchmark for cross-lingual (X) Slot and Intent Detection in 13 languages from 6 language families, including a very low-resource dialect.
"""
_HOMEPAGE = "https://bitbucket.org/robvanderg/xsid/src/master/"
_LANGUAGES = ["ind"]
_LICENSE = "CC-BY-SA 4.0"
_LOCAL = False
_URLS = {
    _DATASETNAME: "https://bitbucket.org/robvanderg/xsid/get/04ce1e6c8c28.zip",
}
_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION, Tasks.POS_TAGGING]
_SOURCE_VERSION = "0.3.0"
_NUSANTARA_VERSION = "1.0.0"

INTENT_LIST = [
    "AddToPlaylist",
    "BookRestaurant",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
    "alarm/cancel_alarm",
    "alarm/modify_alarm",
    "alarm/set_alarm",
    "alarm/show_alarms",
    "alarm/snooze_alarm",
    "alarm/time_left_on_alarm",
    "reminder/cancel_reminder",
    "reminder/set_reminder",
    "reminder/show_reminders",
    "weather/checkSunrise",
    "weather/checkSunset",
    "weather/find"
]

TAG_LIST = [
    "B-album",
    "B-artist",
    "B-best_rating",
    "B-condition_description",
    "B-condition_temperature",
    "B-cuisine",
    "B-datetime",
    "B-ecurring_datetime",
    "B-entity_name",
    "B-facility",
    "B-genre",
    "B-location",
    "B-movie_name",
    "B-movie_type",
    "B-music_item",
    "B-object_location_type",
    "B-object_name",
    "B-object_part_of_series_type",
    "B-object_select",
    "B-object_type",
    "B-party_size_description",
    "B-party_size_number",
    "B-playlist",
    "B-rating_unit",
    "B-rating_value",
    "B-recurring_datetime",
    "B-reference",
    "B-reminder/todo",
    "B-restaurant_name",
    "B-restaurant_type",
    "B-served_dish",
    "B-service",
    "B-sort",
    "B-track",
    "B-weather/attribute",
    "I-album",
    "I-artist",
    "I-best_rating",
    "I-condition_description",
    "I-condition_temperature",
    "I-cuisine",
    "I-datetime",
    "I-ecurring_datetime",
    "I-entity_name",
    "I-facility",
    "I-genre",
    "I-location",
    "I-movie_name",
    "I-movie_type",
    "I-music_item",
    "I-object_location_type",
    "I-object_name",
    "I-object_part_of_series_type",
    "I-object_select",
    "I-object_type",
    "I-party_size_description",
    "I-party_size_number",
    "I-playlist",
    "I-rating_unit",
    "I-rating_value",
    "I-recurring_datetime",
    "I-reference",
    "I-reminder/todo",
    "I-restaurant_name",
    "I-restaurant_type",
    "I-served_dish",
    "I-service",
    "I-sort",
    "I-track",
    "I-weather/attribute",
    "O",
    "Orecurring_datetime"
]

class XSID(datasets.GeneratorBasedBuilder):
    """xSID datasets contains datasets to detect the intent from the text"""

    BUILDER_CONFIGS = [
        NusantaraConfig(
            name="xsid_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="xSID source schema",
            schema="source",
            subset_id="xsid",
        ),
        NusantaraConfig(
            name="xsid_nusantara_text",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="xSID Nusantara intent classification schema",
            schema="nusantara_text",
            subset_id="xsid",
        ),
        NusantaraConfig(
            name="xsid_nusantara_seq_label",
            version=datasets.Version(_NUSANTARA_VERSION),
            description="xSID Nusantara pos tagging schema",
            schema="nusantara_seq_label",
            subset_id="xsid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "xsid_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text-en": datasets.Value("string"),
                    "intent": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "nusantara_text":
            features = schemas.text_features(label_names=INTENT_LIST)
        elif self.config.schema == "nusantara_seq_label":
            features = schemas.seq_label_features(label_names=TAG_LIST)
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
        base_path = Path(dl_manager.download_and_extract(urls)) / "robvanderg-xsid-04ce1e6c8c28" / "data" / "xSID-0.3"
        data_files = {
            "train": base_path / "id.projectedTrain.conll",
            "test": base_path / "id.test.conll",
            "validation": base_path / "id.valid.conll"
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        print('filepath', filepath)
        if self.config.name == "xsid_source":
            with open(filepath, "r") as file:
                data = file.read().strip("\n").split("\n\n")

            i = 0
            for sample in data:
                id = ""
                tokens = []
                for row_sample in sample.split("\n"):
                    s = row_sample.split(": ")
                    if s[0] == "# id":
                        id = s[1]
                    elif s[0] == "# text-en":
                        text_en = s[1]
                    elif s[0] == "# text":
                        text = s[1]
                    elif s[0] == "# intent":
                        intent = s[1]
                    else:
                        tokens.append(s[0])
                
                if id == "":
                    id = i
                    i = i + 1

                ex = {
                    "id": id,
                    "text": text,
                    "text-en": text_en,
                    "intent": intent,
                    "tokens": tokens
                }
                yield id, ex

        elif self.config.name == "xsid_nusantara_text":
            with open(filepath, "r") as file:
                data = file.read().strip("\n").split("\n\n")

            i = 0
            for sample in data:
                id = ""
                for row_sample in sample.split("\n"):
                    s = row_sample.split(": ")
                    if s[0] == "# id":
                        id = s[1]
                    elif s[0] == "# text":
                        text = s[1]
                    elif s[0] == "# intent":
                        intent = s[1]
                
                if id == "":
                    id = i
                    i = i + 1

                ex = {
                    "id": id,
                    "text": text,
                    "label": intent
                }
                yield id, ex

        elif self.config.name == "xsid_nusantara_seq_label":
            with open(filepath, "r") as file:
                data = file.read().strip("\n").split("\n\n")

            i = 0
            for sample in data:
                id = ""
                tokens = []
                labels = []
                for row_sample in sample.split("\n"):
                    s = row_sample.split(": ")
                    if s[0] == "# id":
                        id = s[1]
                    elif len(s) == 1:
                        tokens.append(s[0].split("\t")[1])
                        labels.append(s[0].split("\t")[3])
                
                if id == "":
                    id = i
                    i = i + 1

                ex = {
                    "id": id,
                    "tokens": tokens,
                    "labels": labels
                }
                yield id, ex

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
