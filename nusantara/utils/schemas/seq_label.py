"""
Seq Labeling Schema

Several tasks boil down to transforming sequence of tokens into annother sequence of tokens with the same length, including:

- Named Entity Recognition
- Keyword Extraction
- POS Tagging
"""
import datasets

def features(label_names = ["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "tokens": [datasets.Value("string")],
            "labels": [datasets.ClassLabel(names=label_names)],
        }
    )
