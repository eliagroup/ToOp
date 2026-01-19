# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains the data classes used in the powsybl preprocessing and postprocessing of the data.

File: data_classes.py
Author:  Benjamin Petrick
Created: 2024-09-04
"""

from typing import Any, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    UcteImportResult,
)

PreProcessingIds: TypeAlias = Literal[
    "relevant_subs",
    "line_for_nminus1",
    "trafo_for_nminus1",
    "tie_line_for_nminus1",
    "dangling_line_for_nminus1",
    "generator_for_nminus1",
    "load_for_nminus1",
    "switch_for_nminus1",
    "line_disconnectable",
    "white_list",
    "black_list",
]


class PreProcessingStatistics(BaseModel):
    """Contains all the statistics of the postprocessing."""

    id_lists: dict[PreProcessingIds, Any] = Field(default_factory=dict)
    """ Contains the ids of the N-1 analysis, border line currents and CB lists.
    keys: relevant_subs, line_for_nminus1, trafo_for_nminus1,
          tie_line_for_nminus1, dangling_line_for_nminus1, generator_for_nminus1,
          load_for_nminus1, switches_for_nminus1
    white_list, black_list"""

    import_result: UcteImportResult
    """ Statistics and results from an import process."""

    border_current: dict[str, Any] = Field(default_factory=dict)
    """ Contains the statistics of the current limit for the lines that leave the tso area. """

    network_changes: dict[str, Any] = Field(default_factory=dict)
    """ Contains the statistics of the changes made to the network.
    keys: black_list, white_list, low_impedance_lines, branches_across_switch
    """

    import_parameter: Optional[Union[UcteImporterParameters, CgmesImporterParameters]] = None
    """ Contains the statistics of the post processing. """


class PowsyblSecurityAnalysisParam(BaseModel):
    """Contains all the parameter for a Security Analysis with pypowsybl."""

    single_element_contingencies_ids: dict[str, list[str]]
    """ The ids of the single element contingencies for the different element types.

    The keys are the element types and the values are the ids of the elements.
    keys example: "dangling", "generator", "line", "switch", "tie", "transformer", "load", "custom"
    """

    current_limit_factor: float
    """ The factor to reduce the current limit on the lines.

    This factor needs to be applied before the security analysis in for current limit
    and after in the violation dataframe."""

    monitored_branches: list
    """ The branches that are monitored during the security analysis."""

    monitored_buses: list
    """ The buses that are monitored during the security analysis."""

    ac_run: bool = True
    """ Define load flow type.

        True: run AC N-1 Analysis.
        False: run DC N-1 Analysis."""


class PostProcessingStatistics(BaseModel):
    """Contains all the statistics of the postprocessing.

    TODO define
    """
