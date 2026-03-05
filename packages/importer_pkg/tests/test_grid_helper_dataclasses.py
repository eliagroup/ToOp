# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Type enforcement tests for dataclasses and eqx.Module classes."""

import dataclasses
import importlib
import inspect

import pytest
from beartype.typing import Iterable

MODULES: list[str] = [
    "toop_engine_importer.pypowsybl_import.powsybl_masks",
    "toop_engine_importer.ucte_toolset.ucte_io",
    "toop_engine_importer.pandapower_import.pp_masks",
]


def _iter_target_classes() -> Iterable[type]:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module_name:
                continue
            if dataclasses.is_dataclass(obj):
                yield obj


def _build_wrong_call_args(cls: type) -> tuple[list[object], dict[str, object]]:
    signature = inspect.signature(cls)
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.default is not param.empty:
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = "wrong"
        else:
            args.append("wrong")
    return args, kwargs


@pytest.mark.parametrize(
    "cls",
    list(_iter_target_classes()),
    ids=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
)
def test_dataclasses_reject_wrong_types(cls: type) -> None:
    args, kwargs = _build_wrong_call_args(cls)
    if not args and not kwargs:
        pytest.skip("No required parameters to validate")
    with pytest.raises(Exception):
        cls(*args, **kwargs)
