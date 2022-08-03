"""
Utility for filtering and loading BigBio datasets.
"""
from collections import Counter
from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Dict

from dataclasses import dataclass
from dataclasses import field
import datasets
from datasets import load_dataset

from nusantara.utils.configs import NusantaraConfig
from nusantara.utils.constants import Tasks, SCHEMA_TO_TASKS

_LARGE_CONFIG_NAMES = [
    
]

_RESOURCE_CONFIG_NAMES = [

]

_CURRENTLY_BROKEN_NAMES = [

]

@dataclass
class NusantaraMetadata:
    """Metadata for one config of a dataset."""

    script: pathlib.Path
    dataset_name: str
    tasks: List[Tasks]
    languages: List[str]
    config: NusantaraConfig
    is_local: bool
    is_nusantara_schema: bool
    nusantara_schema_caps: Optional[str]
    is_large: bool
    is_resource: bool
    is_default: bool
    is_broken: bool
    nusantara_version: str
    source_version: str
    citation: str
    description: str
    homepage: str
    license: str

    _ds_module: datasets.load.DatasetModule = field(repr=False)
    _py_module: ModuleType = field(repr=False)
    _ds_cls: type = field(repr=False)

    def get_load_dataset_kwargs(
        self,
        **extra_load_dataset_kwargs,
    ):
        return {
            "path": self.script,
            "name": self.config.name,
            **extra_load_dataset_kwargs,
        }

    def load_dataset(
        self,
        **extra_load_dataset_kwargs,
    ):
        return load_dataset(
            path=self.script,
            name=self.config.name,
            **extra_load_dataset_kwargs,
        )

    def get_metadata(self, **extra_load_dataset_kwargs):
        if not self.is_nusantara_schema:
            raise ValueError("only supported for nusantara schemas")
        dsd = self.load_dataset(**extra_load_dataset_kwargs)
        split_metas = {}
        for split, ds in dsd.items():
            meta = SCHEMA_TO_METADATA_CLS[self.config.schema].from_dataset(ds)
            split_metas[split] = meta
        return split_metas


def default_is_keeper(metadata: NusantaraMetadata) -> bool:
    return not metadata.is_large and not metadata.is_resource and metadata.is_nusantara_schema

class NusantaraConfigHelper:
    """
    Handles creating and filtering BigBioDatasetConfigHelper instances.
    """

    def __init__(
        self,
        helpers: Optional[Iterable[NusantaraMetadata]] = None,
        keep_broken: bool = False,
    ):

        path_to_here = pathlib.Path(__file__).parent.absolute()
        self.path_to_biodatasets = (path_to_here / "nusa_datasets").resolve()
        self.dataloader_scripts = sorted(
            self.path_to_biodatasets.glob(os.path.join("*", "*.py"))
        )
        self.dataloader_scripts = [
            el for el in self.dataloader_scripts if el.name != "__init__.py"
        ]

        # if helpers are passed in, just attach and go
        if helpers is not None:
            if keep_broken:
                self._helpers = helpers
            else:
                self._helpers = [helper for helper in helpers if not helper.is_broken]
            return

        # otherwise, create all helpers available in package
        helpers = []
        for dataloader_script in self.dataloader_scripts:
            dataset_name = dataloader_script.stem
            py_module = SourceFileLoader(
                dataset_name, dataloader_script.as_posix()
            ).load_module()
            ds_module = datasets.load.dataset_module_factory(
                dataloader_script.as_posix()
            )
            ds_cls = datasets.load.import_main_class(ds_module.module_path)

            for config in ds_cls.BUILDER_CONFIGS:

                is_nusantara_schema = config.schema.startswith("nusantara")
                if is_nusantara_schema:
                    nusantara_schema_caps = config.schema.split("_")[1].upper()
                    if nusantara_schema_caps == 'SEQ':
                        nusantara_schema_caps = 'SEQ_LABEL'
                    tasks = SCHEMA_TO_TASKS[nusantara_schema_caps] & set(
                        py_module._SUPPORTED_TASKS
                    )
                else:
                    tasks = py_module._SUPPORTED_TASKS
                    nusantara_schema_caps = None

                helpers.append(
                    NusantaraMetadata(
                        script=dataloader_script.as_posix(),
                        dataset_name=dataset_name,
                        tasks=tasks,
                        languages=py_module._LANGUAGES,
                        config=config,
                        is_local=py_module._LOCAL,
                        is_nusantara_schema=is_nusantara_schema,
                        nusantara_schema_caps=nusantara_schema_caps,
                        is_large=config.name in _LARGE_CONFIG_NAMES,
                        is_resource=config.name in _RESOURCE_CONFIG_NAMES,
                        is_default=config.name == ds_cls.DEFAULT_CONFIG_NAME,
                        is_broken=config.name in _CURRENTLY_BROKEN_NAMES,
                        nusantara_version=py_module._NUSANTARA_VERSION,
                        source_version=py_module._SOURCE_VERSION,
                        citation=py_module._CITATION,
                        description=py_module._DESCRIPTION,
                        homepage=py_module._HOMEPAGE,
                        license=py_module._LICENSE,
                        _ds_module=ds_module,
                        _py_module=py_module,
                        _ds_cls=ds_cls,
                    )
                )

        if keep_broken:
            self._helpers = helpers
        else:
            self._helpers = [helper for helper in helpers if not helper.is_broken]

    @property
    def available_dataset_names(self) -> List[str]:
        return sorted(list(set([helper.dataset_name for helper in self])))

    def for_dataset(self, dataset_name: str) -> "NusantaraConfigHelper":
        helpers = [helper for helper in self if helper.dataset_name == dataset_name]
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.dataset_name = {dataset_name}")
        return NusantaraConfigHelper(helpers=helpers)

    def for_config_name(self, config_name: str) -> "NusantaraMetadata":
        helpers = [helper for helper in self if helper.config.name == config_name]
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.config.name = {config_name}")
        if len(helpers) > 1:
            raise ValueError(
                f"multiple helpers with helper.config.name = {config_name}"
            )
        return helpers[0]

    def default_for_dataset(self, dataset_name: str) -> "NusantaraMetadata":
        helpers = [
            helper
            for helper in self
            if helper.is_default and helper.dataset_name == dataset_name
        ]
        assert len(helpers) == 1
        return helpers[0]

    def filtered(
        self, is_keeper: Callable[[NusantaraMetadata], bool]
    ) -> "NusantaraConfigHelper":
        """Return dataset config helpers that match is_keeper."""
        return NusantaraConfigHelper(
            helpers=[helper for helper in self if is_keeper(helper)]
        )

    def __repr__(self):
        return "\n\n".join([helper.__repr__() for helper in self])

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for helper in self._helpers:
            yield helper

    def __len__(self):
        return len(self._helpers)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return NusantaraConfigHelper(
                helpers=[self._helpers[ii] for ii in range(start, stop, step)]
            )
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return self._helpers[key]
        else:
            raise TypeError("Invalid argument type.")