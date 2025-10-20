"""Holds functions that help with all sorts of pandapower conversions, that are not specific to a single task."""

from numbers import Integral

import numpy as np
import pandapower as pp
import pandapower.toolbox
import pandas as pd
from beartype.typing import Iterable, Literal, Optional, Sequence
from jaxtyping import Bool, Float, Integer
from pandapower.pypower.idx_brch import SHIFT
from pandapower.toolbox import get_connected_buses


def get_phaseshift_key(trafo_type: str) -> str:
    """Get the key of the shift_degree in the trafo

    Parameters
    ----------
    trafo_type : str
        The type of trafo

    Returns
    -------
    str
        The key of the phaseshift attribute in the trafo(3w)dataframe
    """
    lookup_table = {
        "trafo": "shift_degree",
        "trafo3w_lv": "shift_lv_degree",
        "trafo3w_mv": "shift_mv_degree",
    }
    assert trafo_type in lookup_table
    return lookup_table[trafo_type]


def get_phaseshift_mask(
    net: pandapower.pandapowerNet,
) -> tuple[Bool[np.ndarray, " n_branches"], Bool[np.ndarray, " n_branches"], list[Float[np.ndarray, " n_tap_positions"]]]:
    """Return the mask over phaseshifters in the ppci and the minimum and maximum setpoints for these PSTs

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to extract the data from

    Returns
    -------
    pst_mask : Bool[np.ndarray, " n_branches"]
        The mask over phaseshifters in the ppci, true if it is a trafo with a phase shift in the ppci
    controllable_mask : Bool[np.ndarray, " n_branches"]
        The mask over controllable phase shifters in the ppci. True if a trafo has a tap that can
        change the phase shift
        Note that this won't catch trafos with a fixed shift_degree, only those which have a tap
        changer that can change the angle shift.
    shift_taps : list[Float[np.ndarray, " n_tap_positions"]]
        A list of arrays with the tap positions for each controllable phase shifter. The outer list has length
        sum(controllable_mask), the inner arrays have length equal to the number of taps of each PST (varying)
    """
    # Everything that has a shift angle in the PPCI must be regarded as a phase shifter
    ppci = pp.converter.to_ppc(net, init="flat", calculate_voltage_angles=True)
    has_shift_angle = np.array(ppci["branch"][:, SHIFT] != 0)

    # Additionally there are trafos that have a tap_changer_type, but might be set to neutral
    # Hence, these need to be regarded separately. Trafos with a tap will also get a max and min
    # setpoint
    if not len(net.trafo):
        return has_shift_angle, np.zeros_like(has_shift_angle), []

    controllable: Bool[np.ndarray, " n_trafos"] = (
        (net.trafo.vn_hv_kv == net.trafo.vn_lv_kv)
        & (net.trafo.tap_changer_type is not None)
        & (net.trafo.tap_step_degree != 0)
        & ~net.trafo.tap_min.isna()
        & ~net.trafo.tap_max.isna()
        & ~net.trafo.tap_step_degree.isna()
    ).values
    tap_min = np.array(net.trafo.tap_min)[controllable]
    tap_max = np.array(net.trafo.tap_max)[controllable]
    tap_step = np.array(net.trafo.tap_step_degree)[controllable]

    shift_taps = [np.arange(t_min, t_max + 1) * t_step for (t_min, t_max, t_step) in zip(tap_min, tap_max, tap_step)]

    ppci_start, ppci_end = net._pd2ppc_lookups["branch"]["trafo"]
    pst_indices = np.arange(ppci_start, ppci_end)[controllable]

    controllable_global: Bool[np.ndarray, " n_branches"] = np.zeros_like(has_shift_angle)
    controllable_global[pst_indices] = True
    has_shift_angle[pst_indices] = True

    return has_shift_angle, controllable_global, shift_taps


def get_bus_key(branch_name: str, branch_from_end: bool) -> str:
    """Get the key of the bus in the branch name

    Parameters
    ----------
    branch_name : str
        The name of the branch
    branch_from_end : bool
        Whether to get the from or to bus, True for from, False for to

    Returns
    -------
    str
        The key of the bus in the branch name
    """
    lookup_table = {
        "line": {True: "from_bus", False: "to_bus"},
        "trafo": {True: "hv_bus", False: "lv_bus"},
        "trafo3w_lv": {True: "lv_bus", False: "lv_bus"},
        "trafo3w_mv": {True: "mv_bus", False: "mv_bus"},
        "trafo3w_hv": {True: "hv_bus", False: "hv_bus"},
    }

    assert branch_name in lookup_table
    assert branch_from_end in lookup_table[branch_name]

    return lookup_table[branch_name][branch_from_end]


def get_bus_key_injection(injection_type: str) -> str:
    """Get the key of the bus in the injection type

    Parameters
    ----------
    injection_type : str
        The type of the injection

    Returns
    -------
    str
        The key of the bus in the injection type
    """
    lookup_table = {
        "gen": "bus",
        "sgen": "bus",
        "load": "bus",
        "shunt": "bus",
        "dcline_from": "from_bus",
        "dcline_to": "to_bus",
        # "ward_load": "bus",
        # "ward_shunt": "bus",
        # "xward_load": "bus",
        # "xward_gen": "bus",
        # "xward_shunt": "bus",
    }
    assert injection_type in lookup_table
    return lookup_table[injection_type]


def get_element_table(element_type: str, res_table: bool = False) -> str:
    """Get the element table for the injection or branch type or bus

    Parameters
    ----------
    element_type : str
        The type description of the branch/injection/bus
    res_table : bool
        Whether to get the res table. Defaults to False

    Returns
    -------
    str
        The element table for the branch/injection/bus type
    """
    lookup_table = {
        "bus": "bus",
        "switch": "switch",
        # branches
        "line": "line",
        "trafo": "trafo",
        "trafo3w": "trafo3w",
        "trafo3w_lv": "trafo3w",
        "trafo3w_mv": "trafo3w",
        "trafo3w_hv": "trafo3w",
        # injections
        "gen": "gen",
        "sgen": "sgen",
        "load": "load",
        "shunt": "shunt",
        "dcline_from": "dcline",
        "dcline_to": "dcline",
        "ward_load": "ward",
        "ward_shunt": "ward",
        "xward_load": "xward",
        "xward_gen": "xward",
        "xward_shunt": "xward",
    }
    assert element_type in lookup_table

    key = lookup_table[element_type]

    if res_table:
        key = "res_" + key

    return key


power_key_lookup_table = {
    "def_active": {
        "gen": "p_mw",
        "sgen": "p_mw",
        "load": "p_mw",
        "shunt": "p_mw",
        "dcline_from": "p_mw",
        "dcline_to": "dcline",
        "ward_load": "ps_mw",
        "ward_shunt": "pz_mw",
        "xward_load": "ps_mw",
        "xward_shunt": "pz_mw",
    },
    "def_reactive": {
        # "gen": "no_column",
        "sgen": "q_mvar",
        "load": "q_mvar",
        "shunt": "q_mvar",
        # "dcline_from": "no_column",
        # "dcline_to": "no_column",
        "ward_load": "qs_mvar",
        "ward_shunt": "qz_mvar",
        "xward_load": "qs_mvar",
        "xward_shunt": "qz_mvar",
    },
    "res_active": {
        "gen": "p_mw",
        "sgen": "p_mw",
        "load": "p_mw",
        "shunt": "p_mw",
        "dcline_from": "p_from_mw",
        "dcline_to": "p_to_mw",
        "ward_load": "p_mw",
        # "ward_shunt": "no_column",
        "xward_load": "p_mw",
        # "xward_shunt": "no_column",
    },
    "res_reactive": {
        "gen": "q_mvar",
        "sgen": "q_mvar",
        "load": "q_mvar",
        "shunt": "q_mvar",
        "dcline_from": "q_from_mvar",
        "dcline_to": "q_to_mvar",
        "ward_load": "q_mvar",
        # "ward_shunt": "no_column",
        "xward_load": "q_mvar",
        # "xward_shunt": "no_column",
    },
}


def get_power_key(element_type: str, res_table: bool = False, reactive: bool = False) -> str:
    """
    Get the power key for the injection type in the res/non-res tables

    Parameters
    ----------
    element_type : str
        The type description of the injection
    res_table : bool
        Whether to get the key for the res table. Defaults to False
    reactive : bool
        Whether to get the reactive power instead of the active power. Defaults to False

    Returns
    -------
    str
        The power key for the injection type in pandapowers
    """
    table_key = ("res" if res_table else "def") + ("_reactive" if reactive else "_active")

    return power_key_lookup_table[table_key][element_type]


def get_pandapower_loadflow_results_in_ppc(
    net: pp.pandapowerNet,
) -> Float[np.ndarray, " n_branches_ppc"]:
    """Get the loadflow result for all pandapower branches that are included in ppc (= even if not in service)

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network with loadflows already computed

    Returns
    -------
    Float[np.ndarray, " n_branches_ppc"]
    An array of all loadflows of branches that are in ppci in the correct order
    """
    branch_loads_pandapower = np.concatenate(
        [
            net.res_line.p_from_mw.values,
            net.res_trafo.p_hv_mw.values,
            net.res_trafo3w.p_hv_mw.values,
            net.res_trafo3w.p_mv_mw.values,
            net.res_trafo3w.p_lv_mw.values,
            net.res_impedance.p_from_mw.values,
            np.zeros(
                len(net.xward), dtype=float
            ),  # The p_mw from res_xward is stored on aux bus level not on the aux branch
        ]
    )
    return branch_loads_pandapower


type_mapper = {
    "line": "res_line",
    "trafo": "res_trafo",
    "trafo3w_lv": "res_trafo3w",
    "trafo3w_mv": "res_trafo3w",
    "trafo3w_hv": "res_trafo3w",
    "impedance": "res_impedance",
    "xward": "res_xward",
}

sign_mapper = {
    "line": -1,
    "trafo": -1,
    "trafo3w_lv": 1,
    "trafo3w_mv": 1,
    "trafo3w_hv": -1,
    "impedance": -1,
    "xward": -1,
}


column_mapper = {
    "active_from": {
        "line": "p_from_mw",
        "trafo": "p_hv_mw",
        "trafo3w_lv": "p_lv_mw",
        "trafo3w_mv": "p_mv_mw",
        "trafo3w_hv": "p_hv_mw",
        "impedance": "p_from_mw",
        "xward": "p_mw",
    },
    "reactive_from": {
        "line": "q_from_mvar",
        "trafo": "q_hv_mvar",
        "trafo3w_lv": "q_lv_mvar",
        "trafo3w_mv": "q_mv_mvar",
        "trafo3w_hv": "q_hv_mvar",
        "impedance": "q_from_mvar",
        "xward": "q_mvar",
    },
    "current_from": {
        "line": "i_from_ka",
        "trafo": "i_hv_ka",
        "trafo3w_lv": "i_lv_ka",
        "trafo3w_mv": "i_mv_ka",
        "trafo3w_hv": "i_hv_ka",
        "impedance": "i_from_ka",
        "xward": "no_column",
    },
    "active_to": {
        "line": "p_to_mw",
        "trafo": "p_lv_mw",
        "trafo3w_lv": "p_lv_mw",
        "trafo3w_mv": "p_mv_mw",
        "trafo3w_hv": "p_hv_mw",
        "impedance": "p_to_mw",
        "xward": "p_mw",
    },
    "reactive_to": {
        "line": "q_to_mvar",
        "trafo": "q_lv_mvar",
        "trafo3w_lv": "q_lv_mvar",
        "trafo3w_mv": "q_mv_mvar",
        "trafo3w_hv": "q_hv_mvar",
        "impedance": "q_to_mvar",
        "xward": "q_mvar",
    },
    "current_to": {
        "line": "i_to_ka",
        "trafo": "i_lv_ka",
        "trafo3w_lv": "i_lv_ka",
        "trafo3w_mv": "i_mv_ka",
        "trafo3w_hv": "i_hv_ka",
        "impedance": "i_to_ka",
        "xward": "no_column",
    },
}


def get_pandapower_loadflow_results_injection(
    net: pp.pandapowerNet,
    types: Sequence[str],
    ids: Sequence[Integral],
    reactive: bool = False,
) -> Float[np.ndarray, " n_injections"]:
    """Use a list of injection types and ids to get the loadflow results

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network with loadflows already computed
    types : Sequence[str]
        The types of the injections, length n_injections
    ids : Sequence[int]
        The ids of the injections, length n_injections
    reactive : bool
        Whether to get the reactive power instead of the active power. Defaults to False

    Returns
    -------
    Float[np.ndarray, " n_injections"]
        The loadflow results of the injections in the correct order
    """

    def get_from_net(elem_type: str, elem_id: Integral) -> float:
        """Get data for a single injection

        Parameters
        ----------
        elem_type : str
            The type of the injection
        elem_id : int
            The id of the injection

        Returns
        -------
        float
            The power of the injection
        """
        table = get_element_table(elem_type, res_table=False)
        res_table = get_element_table(elem_type, res_table=True)
        try:
            power_key = get_power_key(elem_type, res_table=True, reactive=reactive)
        except KeyError:
            return 0.0

        sign = pandapower.toolbox.signing_system_value(table)
        return net[res_table].loc[elem_id, power_key] * sign

    injection_power = np.array([get_from_net(t, i) for t, i in zip(types, ids)])
    return injection_power


def get_pandapower_branch_loadflow_results_sequence(
    net: pp.pandapowerNet,
    types: Sequence[str],
    ids: Sequence[Integral],
    measurement: Literal["active", "reactive", "current"],
    from_end: bool = True,
    fill_na: Optional[float] = None,
    adjust_signs: bool = True,
) -> Float[np.ndarray, " n_branches"]:
    """Use a list of branch types and ids to get the loadflow results

    This can also applied in case the network is slightly different than the net in the backend,
    e.g. to use a non-preprocessed net

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network with loadflows already computed
    types : Sequence[str]
        The types of the branches, length n_branches
    ids : Sequence[int]
        The ids of the branches, length n_branches
    measurement : Literal["active", "reactive", "current"]
        The type of the measurement to get. Can be "active", "reactive" or "current"
    from_end : bool
        Whether to get the results from the end of the branch. Defaults to True
    fill_na: Optional[float]
        Whether to replace NaN values with a constant
    adjust_signs: bool
        Whether to adjust the signs of the results according to the type of branch. Defaults to True

    Returns
    -------
    Float[np.ndarray, " n_branches"]
        The loadflow results of the branches in the correct order
    """
    assert len(types) == len(ids)

    mapper_key = measurement + ("_from" if from_end else "_to")

    mapped_types = (type_mapper[t] for t in types)
    mapped_columns = (column_mapper[mapper_key][t] for t in types)
    mapped_signs = (sign_mapper[t] for t in types) if adjust_signs else (1 for _ in types)

    # It might be more performant to group this in the future, i.e. to get all results at once
    branch_loads = np.array([net[t][c].loc[i] * s for t, c, i, s in zip(mapped_types, mapped_columns, ids, mapped_signs)])

    # Currents are saved in kA, we want A
    if measurement == "current":
        branch_loads *= 1000

    if fill_na is not None:
        branch_loads = np.nan_to_num(branch_loads, nan=fill_na)

    return branch_loads


def get_pandapower_bus_loadflow_results_sequence(
    net: pp.pandapowerNet,
    monitored_bus_ids: Sequence[Integral],
    voltage_magnitude: bool = False,
) -> Float[np.ndarray, " n_buses"]:
    """Use a list of bus ids to get the loadflow results

    This can also applied in case the network is slightly different than the net in the backend,
    e.g. to use a non-preprocessed net

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network with loadflows already computed
    monitored_bus_ids : Sequence[int]
        The ids of the buses, length n_buses
    voltage_magnitude : bool
        Whether to get the voltage magnitude instead of the angle. Defaults to False

    Returns
    -------
    Float[np.ndarray, " n_buses"]
        The loadflow results of the buses in the correct order
    """
    if voltage_magnitude:
        res = net.res_bus.vm_pu.loc[monitored_bus_ids].values
    else:
        res = net.res_bus.va_degree.loc[monitored_bus_ids].values
    return res


isnan_column_mapper = {
    "line": "i_from_ka",
    "trafo": "i_hv_ka",
    "trafo3w_lv": "i_lv_ka",
    "trafo3w_mv": "i_mv_ka",
    "trafo3w_hv": "i_hv_ka",
    "impedance": "i_from_ka",
}


def check_for_splits(
    net: pp.pandapowerNet,
    monitored_branch_types: Sequence[str],
    monitored_branch_ids: Sequence[Integral],
) -> bool:
    """Check for splits by checking for NaN values

    Unfortunately, p_xx values are zero on split in pandapower

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network with loadflows already computed
    monitored_branch_types : Sequence[str]
        The types of the monitored branches, length N_branches
    monitored_branch_ids : Sequence[int]
        The ids of the monitored branches, length N_branches

    Returns
    -------
    bool
        Whether a split is present, i.e. any monitored branch has a NaN value
    """
    assert len(monitored_branch_types) == len(monitored_branch_ids)

    mapped_types = (type_mapper[t] for t in monitored_branch_types)
    mapped_columns = (isnan_column_mapper[t] for t in monitored_branch_types)

    isnan = any(np.isnan(net[t][c].loc[i]) for t, c, i in zip(mapped_types, mapped_columns, monitored_branch_ids))
    return isnan


def get_dc_bus_voltage(net: pp.pandapowerNet) -> pd.Series:
    """Get the bus voltage of all buses in the network under DC conditions

    This is usually bus.vn_kv, except a generator is connected in which case the voltage is the
    generator voltage

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network

    Returns
    -------
    pd.Series
        The bus voltage of all buses in the network, indexed by bus id
    """
    buses = pd.merge(
        left=net.bus,
        right=net.gen,
        left_index=True,
        right_on="bus",
        how="left",
        suffixes=("", "_gen"),
    )
    buses.index = net.bus.index
    buses["vn_kv"] = buses["vm_pu"].fillna(1) * buses["vn_kv"]
    return buses["vn_kv"]


def get_shunt_real_power(
    bus_voltage: Float[np.ndarray, " n_shunts"],
    shunt_power: Float[np.ndarray, " n_shunts"],
    shunt_voltage: Optional[Float[np.ndarray, " n_shunts"]] = None,
    shunt_step: Optional[Integer[np.ndarray, " n_shunts"]] = None,
) -> Float[np.ndarray, " n_shunts"]:
    """Get the real power of all shunts in the network

    Parameters
    ----------
    bus_voltage: Float[np.ndarray, " n_shunts"]
        The voltage level of the buses the shunts are connected to
    shunt_power: Float[np.ndarray, " n_shunts"]
        The power of the shunts
    shunt_voltage: Optional[Float[np.ndarray, " n_shunts"]]
        The voltage level of the shunts. Defaults to bus_voltage
    shunt_step: Optional[Int[np.ndarray, " n_shunts"]]
        The step of the shunts. Defaults to np.ones(shunt_power.shape[0], dtype=int)

    Returns
    -------
    Float[np.ndarray, " n_shunts"]
        The real power of all shunts in the network
    """
    if shunt_voltage is None:
        shunt_voltage = bus_voltage
    if shunt_step is None:
        shunt_step = np.ones(shunt_power.shape[0], dtype=int)
    shunt_vratio = (bus_voltage / shunt_voltage) ** 2
    shunt_p = shunt_power * shunt_step * shunt_vratio
    return shunt_p


def get_remotely_connected_buses(
    net: pp.pandapowerNet,
    buses: Iterable[Integral],
    consider: tuple[str, ...] = ("l", "s", "t", "t3", "i"),
    respect_switches: bool = True,
    respect_in_service: bool = False,
    max_num_iterations: int = 100,
) -> set[int]:
    """Get the remotely connected buses, calls pp.toolbox.get_connected_buses until no more buses are found

    This is useful for finding grid islands or stations and uses the same function interface as get_connected_buses

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network
    buses : Sequence[int]
        The buses to get the remotely connected buses for
    consider : Optional[Sequence[str]]
        The types of elements to consider. Defaults to all. Use ("s",) to only consider switches, which will limit
        the search to a station. According to the pandapower docs:
        - l: lines
        - s: switches
        - t: trafos
        - t3: trafo3ws
        - i: impedances
    respect_switches : bool
        Whether to respect switches. If true, will not hop over open switches. Defaults to True
    respect_in_service : bool
        Whether to respect the in_service flag of the elements. Will not hop over out-of-service elements. Defaults to False
    max_num_iterations : int
        After how many iterations to stop searching for new buses. Defaults to 100

    Returns
    -------
    set[int]
        The remotely connected buses, indexed by bus id. In contrast to the pandapower function, this will include the set
        of buses passed in as a parameter except for those that are not in the network.

    Raises
    ------
    RuntimeError
        If the maximum number of iterations is reached without finding all connected buses
    """
    working_set = set(buses).intersection(net.bus.index)
    for i in range(max_num_iterations):
        new_set: set[Integral] = get_connected_buses(
            net=net,
            buses=working_set,
            consider=consider,
            respect_switches=respect_switches,
            respect_in_service=respect_in_service,
        )
        if new_set in working_set or len(new_set) == 0:
            break
        working_set.update(new_set)

        if i >= max_num_iterations - 1:
            raise RuntimeError(
                f"Max number of iterations ({max_num_iterations}) reached while searching for remotely connected buses."
            )
    return {int(node_id) for node_id in working_set}
