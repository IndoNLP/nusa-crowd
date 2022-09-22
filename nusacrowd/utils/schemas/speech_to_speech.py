"""
SpeechSpeech Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "path_1": datasets.Value("string"),
        "audio_1": datasets.Audio(sampling_rate=16_000),
        "text_1": datasets.Value("string"),
        "metadata_1": {
            "name": datasets.Value("string"),
            "speaker_age": datasets.Value("int64"),
            "speaker_gender": datasets.Value("string"),
        },
        "path_2": datasets.Value("string"),
        "audio_2": datasets.Audio(sampling_rate=16_000),
        "text_2": datasets.Value("string"),
        "metadata_2": {
            "name": datasets.Value("string"),
            "speaker_age": datasets.Value("int64"),
            "speaker_gender": datasets.Value("string"),
        }
    }
)
