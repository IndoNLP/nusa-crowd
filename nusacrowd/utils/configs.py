from dataclasses import dataclass

import datasets


@dataclass
class NusantaraConfig(datasets.BuilderConfig):
    """BuilderConfig for Nusantara."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None
