"""
Self-supervised Pretraining (SSP) Classification Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "text": datasets.Value("string"),
    }
)