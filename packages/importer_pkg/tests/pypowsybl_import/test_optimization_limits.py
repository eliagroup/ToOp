# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
import polars as pl
import pypowsybl
import pytest
from pypowsybl.network.impl.network import Network
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import get_p_max
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import.artificial_limits import (
    ARTIFICIAL_DC_LIMITS,
    add_default_permanent_limits,
    create_artificial_limits,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_worst_case_branch_results_polars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import DoubleLimits
from toop_engine_interfaces.nminus1_definition import Contingency, MonitoredElement, Nminus1Definition


def test_create_artificial_limits(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    limits = network.get_operational_limits().reset_index()
    limits = limits[limits.element_type == "LINE"]
    permanent = limits[limits.name == "permanent_limit"].groupby(["element_id", "side"]).value.max()
    n_1 = limits[limits.name == "dacf_permanent_n_1"].groupby(["element_id", "side"]).value.max()
    n_1 = n_1.reindex(permanent.index).fillna(permanent)
    line_ids = permanent.unstack()["ONE"].dropna().index[:3]
    optimized_id, non_worsening_id, both_id = line_ids
    flow_fraction = {optimized_id: 0.5, non_worsening_id: 0.3, both_id: 0.5}
    n_1_factor = 1.2

    # The lines of the complex grid only have limits on side one
    rows = [(line_id, "ONE") for line_id in line_ids]
    worst_case = pl.DataFrame(
        {
            "element": [line_id for line_id, _ in rows],
            "side": [1 for _ in rows],
            "n0": [flow_fraction[line_id] * permanent[line_id, side] for line_id, side in rows],
            "n1": [n_1_factor * flow_fraction[line_id] * n_1[line_id, side] for line_id, side in rows],
        }
    )
    nminus1_definition = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=optimized_id, kind="branch", type="LINE", optimized=True, non_worsening=False),
            MonitoredElement(id=non_worsening_id, kind="branch", type="LINE", optimized=False, non_worsening=True),
            MonitoredElement(id=both_id, kind="branch", type="LINE", optimized=True, non_worsening=True),
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    create_artificial_limits(network, nminus1_definition, worst_case, DoubleLimits(lower=0.9), allow_missing_permanent=True)

    new_limits = network.get_operational_limits().reset_index()
    for case, physical, factor in (("n_0", permanent, 1.0), ("n_1", n_1, n_1_factor)):
        case_limits = new_limits[new_limits.name == f"artificial_limit_{case}"].set_index(["element_id", "side"]).value
        assert set(case_limits.index.get_level_values("element_id")) == set(line_ids)
        for (line_id, side), value in case_limits.items():
            # Optimized-only branches below the lower limit get the lower limit, the others their flow
            # but non-worsening branches at least the minimum non-worsening limit
            fraction = 0.9 if line_id == optimized_id else flow_fraction[line_id] * factor
            if line_id != optimized_id:
                fraction = max(
                    fraction * (1 + DoubleLimits().non_worsening_margin),
                    DoubleLimits().minimum_lower_physical_non_worsening_limit,
                )
            assert value == pytest.approx(fraction * physical[line_id, side])


def test_artificial_limits_equal_permanent_limits_without_double_limits(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    branches = network.get_branches(attributes=["type", "i1"])
    # Branches without a loadflow are out of service or outside the main component, the DC solver ignores them
    branches_with_flow = branches.index[branches["i1"].notna()]
    nminus1_definition = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=branch_id, kind="branch", type=branch_type, optimized=True, non_worsening=False)
            for branch_id, branch_type in branches["type"].items()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )
    security_analysis_results = get_ac_loadflow_results(
        net=network, n_minus_1_definition=nminus1_definition, lf_params=CGMES_DISTRIBUTED_SLACK
    )
    worst_case_currents = extract_worst_case_branch_results_polars(security_analysis_results, nminus1_definition, timestep=0)

    create_artificial_limits(
        network,
        nminus1_definition,
        worst_case_currents,
        DoubleLimits(lower=1.0, upper=1.0),
        allow_missing_permanent=True,
    )

    limits = network.get_operational_limits().reset_index()
    for_branches = limits[limits.element_type.isin(["LINE", "TWO_WINDINGS_TRANSFORMER"])]
    for_branches = for_branches[for_branches.element_id.isin(branches_with_flow)]
    permanent = for_branches[for_branches.name == "permanent_limit"].groupby(["element_id", "side"]).value.max()
    n_1 = for_branches[for_branches.name == "dacf_permanent_n_1"].groupby(["element_id", "side"]).value.max()
    n_1 = n_1.reindex(permanent.index).fillna(permanent)
    for case, physical in (("n_0", permanent), ("n_1", n_1)):
        optimization = for_branches[for_branches.name == f"artificial_limit_{case}"]
        optimization = optimization.groupby(["element_id", "side"]).value.max()
        pd.testing.assert_series_equal(optimization, physical, check_names=False)

    # Every branch with a permanent limit gets an optimization limit, so the DC solver does not fall back to fillna
    p_max = get_p_max(network, fillna=-1.0)
    branches_with_limits = limits[limits.name == "permanent_limit"].element_id.unique()
    branches_with_limits = [branch_id for branch_id in branches_with_limits if branch_id in branches_with_flow]
    assert (p_max.loc[branches_with_limits] > 0).all().all()


def test_ac_and_dc_artificial_limits_coexist_after_reload(complex_grid_network: Network, tmp_path) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    line_id = network.get_lines(attributes=["i1"]).dropna().index[0]
    nminus1_definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=line_id, kind="branch", type="LINE", optimized=True, non_worsening=False)],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )
    worst_case = pl.DataFrame({"element": [line_id], "side": [1], "n0": [1.0], "n1": [1.0]})

    create_artificial_limits(network, nminus1_definition, worst_case, DoubleLimits(), allow_missing_permanent=True)
    create_artificial_limits(
        network, nminus1_definition, worst_case, DoubleLimits(), limits=ARTIFICIAL_DC_LIMITS, allow_missing_permanent=True
    )
    network.save(tmp_path / "grid.xiidm", format="XIIDM")
    reloaded = pypowsybl.network.load(str(tmp_path / "grid.xiidm"))

    names = set(reloaded.get_operational_limits().reset_index().query("element_id == @line_id")["name"])
    expected = {f"{family}_{case}" for family in ("artificial_limit", "artificial_dc_limit") for case in ("n_0", "n_1")}
    assert expected <= names


def test_artificial_limits_skip_occupied_durations(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    all_limits = network.get_operational_limits().reset_index()
    with_permanent = all_limits[(all_limits.name == "permanent_limit") & (all_limits.side == "ONE")].element_id
    line_id = next(line for line in network.get_lines(attributes=["i1"]).dropna().index if line in set(with_permanent))
    permanent = all_limits[
        (all_limits.element_id == line_id) & (all_limits.side == "ONE") & (all_limits.name == "permanent_limit")
    ]
    foreign = permanent.assign(name="foreign", value=1234.0, acceptable_duration=100)
    network.create_operational_limits(
        pd.concat([all_limits[all_limits.element_type != "TIE_LINE"], foreign])
        .drop(columns=["element_type"])
        .set_index("element_id")
    )
    nminus1_definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=line_id, kind="branch", type="LINE", optimized=True, non_worsening=False)],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )
    worst_case = pl.DataFrame({"element": [line_id], "side": [1], "n0": [1.0], "n1": [1.0]})

    create_artificial_limits(network, nminus1_definition, worst_case, DoubleLimits(), allow_missing_permanent=True)

    limits = network.get_operational_limits().reset_index().query("element_id == @line_id and side == 'ONE'")
    assert limits.loc[limits.name == "foreign", "value"].iloc[0] == 1234.0
    artificial = limits[limits.name.str.startswith("artificial_limit")]
    assert set(artificial.name) == {"artificial_limit_n_0", "artificial_limit_n_1"}
    assert not (artificial.acceptable_duration == 100).any()


def _line_with_limit_on_side_one_only(network: Network) -> str:
    limits = network.get_operational_limits().reset_index()
    permanent = limits[limits.name == "permanent_limit"]
    sides = permanent.groupby("element_id").side.agg(set)
    lines = network.get_lines(attributes=["i1"]).dropna().index
    return next(line_id for line_id in lines if sides.get(line_id) == {"ONE"})


def test_add_default_permanent_limits_adds_permanent_limit_on_missing_side(complex_grid_network: Network) -> None:
    network = complex_grid_network
    line_id = _line_with_limit_on_side_one_only(network)

    add_default_permanent_limits(network, [line_id], default_permanent=4321.0)

    limits = network.get_operational_limits().reset_index().query("element_id == @line_id and name == 'permanent_limit'")
    assert limits.set_index("side").value["TWO"] == 4321.0
    assert limits.set_index("side").value["ONE"] != 4321.0


def test_create_artificial_limits_raises_when_monitored_branch_has_no_permanent_limit(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    line_id = _line_with_limit_on_side_one_only(network)
    nminus1_definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=line_id, kind="branch", type="LINE", optimized=True, non_worsening=False)],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )
    worst_case = pl.DataFrame({"element": [line_id], "side": [1], "n0": [1.0], "n1": [1.0]})

    with pytest.raises(ValueError, match=line_id):
        create_artificial_limits(network, nminus1_definition, worst_case, DoubleLimits())


def test_create_artificial_limits_sets_both_sides_after_default_permanent_limits(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    line_id = _line_with_limit_on_side_one_only(network)
    nminus1_definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=line_id, kind="branch", type="LINE", optimized=True, non_worsening=False)],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )
    worst_case = pl.DataFrame({"element": [line_id] * 2, "side": [1, 2], "n0": [1.0] * 2, "n1": [1.0] * 2})
    add_default_permanent_limits(network, [line_id], default_permanent=4321.0)

    create_artificial_limits(network, nminus1_definition, worst_case, DoubleLimits())

    limits = network.get_operational_limits().reset_index().query("element_id == @line_id and side == 'TWO'")
    artificial = limits[limits.name.str.startswith("artificial_limit")].set_index("name").value
    assert artificial.to_dict() == {"artificial_limit_n_0": 4321.0, "artificial_limit_n_1": 4321.0}
