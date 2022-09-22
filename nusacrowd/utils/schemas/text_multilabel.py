"""
General Text Classification Schema
"""
import datasets

def features(label_names = ["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
        }
    )

