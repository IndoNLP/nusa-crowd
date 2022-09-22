"""
General ImageText Classification Schema
"""
import datasets

def features(label_names = ["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "image_paths": datasets.Sequence(datasets.Value("string")),
            "texts": datasets.Value("string"),
            "metadata": {
                "context": datasets.Value("string"),
                "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
            }
        }
    )

