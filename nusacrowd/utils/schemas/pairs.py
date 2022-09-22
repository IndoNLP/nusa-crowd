"""
Text Pairs Schema
"""
import datasets


def features(label_names=["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "text_1": datasets.Value("string"),
            "text_2": datasets.Value("string"),
            "label": datasets.ClassLabel(names=label_names),
        }
    )


def features_with_continuous_label():
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "text_1": datasets.Value("string"),
            "text_2": datasets.Value("string"),
            "label": datasets.Value("float64"),
        }
    )
