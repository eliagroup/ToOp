"""The N-1 definition holds monitored and outaged elements for a grid.

This information is not present in the grid models and hence needs to be stored separately to run an N-1 computation. The
order of the outages should be the same as in the jax code, where it's hardcoded to the following:
- branch outages
- multi outage
- non-relevant injection outages
- relevant injection outages
"""

from pathlib import Path

from beartype.typing import Literal, Optional, Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel, Field
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs, save_pydantic_model_fs

# The type of the ids used in the N-1 definition. This changes how the elements are identified in the grid.
# - unique_pandapower:
#     The ids are unique across all elements in the pandapower net by
#     using the pandapower index and table separated by %%. e.g. "0%%line".
#     Only supported for pandapower grids.
# - powsybl:
#     The ids are the global string ids used in powsybl. These are unique across all elements in the powsybl net.
#     Only supported for powsybl grids.
# - cgmes:
#     The GUIDs used in cgmes. Currently only supported for powsybl grids.
#     TODO: Support pandapower grids with cgmes ids.
# - ucte:
#     The UCTE ids of the elements. Currently only supported for powsybl grids.
POWSYBL_SUPPORTED_ID_TYPES = Literal["powsybl", "cgmes", "ucte"]
PANDAPOWER_SUPPORTED_ID_TYPES = Literal["unique_pandapower", "cgmes"]
ELEMENT_ID_TYPES = Literal[PANDAPOWER_SUPPORTED_ID_TYPES, POWSYBL_SUPPORTED_ID_TYPES]


class GridElement(BaseModel):
    """A grid element is identified by its id(powsybl) or its id and type (pandapower)"""

    id: str
    """The id of the element. For powsybl grids this is the global string id, for pandapower this is the integer index into
    the dataframe
    """

    name: str = ""
    """The name of the element. This is optional, but can be used to provide a more human-readable name for the element."""

    type: Optional[str]
    """For pandapower, we need to further specify a type which corresponds to the table pandapower stores the information in.
    Valid tables are 'line', 'trafo', 'ext_grid', 'gen', 'load', 'shunt', ...
    For powsybl, this is not strictly needed to identify the element however it makes it easier. In that case, type will be
    something like TIE_LINE, LINE, TWO_WINDING_TRANSFORMER, GENERATOR, etc."""

    kind: Literal["branch", "bus", "injection", "switch"]
    """The kind of the element. Usually these are handled differently in the grid modelling software, so it
    can make assembling an N-1 analysis easier if it is known if the element is a branch, bus or injection.
    This could be inferred from the type, however for conveniece it is stored separately.

    For the bus type there is some potential confusion in powsybl. In pandapower, this always refers to the net.bus df.
    In powsybl in a bus/branch model, there are no busbar sections in powsybl, i.e. net.get_node_breaker_topology does not
    deliver busbar sections. Meaning, the "bus" type refers to the net.get_bus_breaker_topology buses if it's a bus/breaker
    topology bus. If it's a node/breaker topology, then "bus" refers to the busbar section.
    """


class Contingency(BaseModel):
    """A single N-1 case"""

    elements: list[GridElement]
    """The grid elements that are to be outaged under this contingency. Usually, this will be exactly one element
    however exceptional contingencies and multi-outages might include more than one element."""

    id: str
    """The id of the contingency. This is used to identify the contingency in the results. It should be unique
    across all contingencies in the N-1 definition."""

    name: str = ""
    """The name of the contingency. This is optional, but can be used to provide a more human-readable name.
    This will show up in the Loadflowresult-tables as column contingency_name."""

    def is_multi_outage(self) -> bool:
        """Check if the contingency is a multi-outage.

        A multi-outage is defined as a contingency that has more than one element in it.
        """
        return len(self.elements) > 1

    def is_basecase(self) -> bool:
        """Check if the contingency is the N-0 base case.

        A base case is defined as a contingency that has no elements in it.
        """
        return len(self.elements) == 0

    def is_single_outage(self) -> bool:
        """Check if the contingency is a normal single-element outage.

        A single outage is defined as a contingency that has exactly one element in it.
        """
        return len(self.elements) == 1


class LoadflowParameters(BaseModel):
    """Loadflow parameters for the N-1 computation."""

    distributed_slack: bool = False
    """Whether to distribute the slack across all injections in the grid. Only relevant for powsybl grids."""

    contingency_propagation: bool = False
    """Whether to enable powsybl's contingency propagation in the N-1 analysis.

    Powsybl:
    https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/security/parameters.html
    Security Analysis will determine by topological search the switches with type circuit breakers
    (i.e. capable of opening fault currents) that must be opened to isolate the fault. Depending on the network structure,
    this could lead to more equipments to be simulated as tripped, because disconnectors and load break switches
    (i.e., not capable of opening fault currents) are not considered.

    Pandapower:
    Currently not supported in pandapower.
    """


class Nminus1Definition(BaseModel):
    """An N-1 definition holds monitored and outaged elements for a grid.

    For powsybl, ids are unique across types (i.e. a branch and an injection can not have the same id), however in
    pandapower, ids are not unique and we have to store the type alongside with them.
    """

    monitored_elements: list[GridElement]
    """A list of monitored elements that should be observed during the N-1 computation."""

    contingencies: list[Contingency]
    """A list of contingencies that should be computed during the N-1 computation."""

    loadflow_parameters: LoadflowParameters = Field(default_factory=LoadflowParameters)
    """Loadflow parameters for the N-1 computation."""

    id_type: Optional[ELEMENT_ID_TYPES] = None
    """The type of the ids used in the N-1 definition. This is used to determine how to interpret the ids in the
    monitored elements and contingencies. See ELEMENT_ID_TYPES for more information. If none,
    pandapower will try to use the globally unique ids, and powsybl will use the global string ids."""

    @property
    def base_case(self) -> Optional[Contingency]:
        """Get the base case contingency, which is the contingency with no elements in it."""
        for contingency in self.contingencies:
            if contingency.is_basecase():
                return contingency
        return None

    def __getitem__(self, key: str | int | slice) -> "Nminus1Definition":
        """Get a subset of the nminus1definition based on the contingencies.

        If a string is given, the contingency id must be in the contingencies list.
        If an integer or slice is given, the case id will be indexed by the integer or slice.
        """
        if isinstance(key, str):
            contingency_ids = [contingency.id for contingency in self.contingencies]
            if key not in contingency_ids:
                raise KeyError(f"Contingency id {key} not in contingencies.")
            index = contingency_ids.index(key)
            index = slice(index, index + 1)
        elif isinstance(key, int):
            index = slice(key, key + 1)
        elif isinstance(key, slice):
            index = key
        else:
            raise TypeError("Key must be a string, int or slice.")

        # pylint: disable=unsubscriptable-object
        return Nminus1Definition(
            monitored_elements=self.monitored_elements,
            contingencies=self.contingencies[index],
            loadflow_parameters=self.loadflow_parameters,
        )


def load_nminus1_definition_fs(
    filesystem: AbstractFileSystem,
    file_path: Union[str, Path],
) -> Nminus1Definition:
    """Load an N-1 definition from a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to load the N-1 definition.
    file_path : Union[str, Path]
        The path to the file containing the N-1 definition in json format.

    Returns
    -------
    Nminus1Definition
        The loaded N-1 definition.
    """
    return load_pydantic_model_fs(
        filesystem=filesystem,
        file_path=file_path,
        model_class=Nminus1Definition,
    )


def load_nminus1_definition(filename: Path) -> Nminus1Definition:
    """Load an N-1 definition from a json file

    Parameters
    ----------
    filename : Path
        The path to the json file containing the N-1 definition.

    Returns
    -------
    Nminus1Definition
        The loaded N-1 definition.
    """
    return load_nminus1_definition_fs(
        filesystem=LocalFileSystem(),
        file_path=filename,
    )


def save_nminus1_definition(filename: Path, nminus1_definition: Nminus1Definition) -> None:
    """Save an N-1 definition to a json file

    Parameters
    ----------
    filename : Path
        The path to the json file to save the N-1 definition to.
    nminus1_definition : Nminus1Definition
        The N-1 definition to save.
    """
    save_pydantic_model_fs(filesystem=LocalFileSystem(), file_path=filename, pydantic_model=nminus1_definition)
