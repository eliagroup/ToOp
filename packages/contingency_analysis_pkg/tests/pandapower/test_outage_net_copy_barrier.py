# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The write barrier around :func:`copy_net_for_outage`.

``MUTABLE_COLUMNS_BY_TABLE`` is the specification of what one outage may write.
:func:`freeze_net_columns` makes every other column of the source net read-only once, and
each outage copy inherits that for the columns it shares - so a write into one raises instead of
corrupting the base case and every later outage. These tests cover that barrier, including the
case where the map has gone stale, which is the failure mode it exists for.
"""

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower import outage_net_copy as onc
from toop_engine_contingency_analysis.pandapower import run_contingency_analysis_pandapower
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    ParallelConfig,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
)


def create_net() -> pp.pandapowerNet:
    """Small four-bus net with a coupler switch: enough to exercise the outage copy path."""
    built = pp.create_empty_network(sn_mva=100.0)
    buses = [
        pp.create_bus(built, vn_kv=110.0, name=f"b{i}", origin_id=f"b{i}_id", GeographicalRegion_id="1") for i in range(4)
    ]
    pp.create_switch(built, bus=buses[0], element=buses[1], et="b", closed=True, type="CB", name="sw01", origin_id="sw01_id")
    line_params = dict(length_km=10.0, r_ohm_per_km=0.05, x_ohm_per_km=0.25, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_line_from_parameters(built, from_bus=buses[1], to_bus=buses[2], name="L1", origin_id="L1_id", **line_params)
    pp.create_line_from_parameters(built, from_bus=buses[1], to_bus=buses[3], name="L2", origin_id="L2_id", **line_params)
    pp.create_gen(built, bus=buses[0], p_mw=0.0, vm_pu=1.0, slack=True, name="slack", origin_id="slack_id")
    pp.create_load(built, bus=buses[2], p_mw=20.0, q_mvar=5.0, name="load", origin_id="load_id")
    pp.create_sgen(built, bus=buses[3], p_mw=8.0, q_mvar=0.0, name="sgen", origin_id="sgen_id")

    for table in ("line", "bus", "switch"):
        built[table]["global_id"] = built[table].index.map(lambda idx, t=table: get_globally_unique_id(idx, t))
    return built


def _build_nminus1_definition(net: pp.pandapowerNet) -> Nminus1Definition:
    monitored = (
        [MonitoredElement(id=r.global_id, type="line", kind="branch", name=r.name) for r in net.line.itertuples()]
        + [MonitoredElement(id=r.global_id, type="bus", kind="bus", name=r.name) for r in net.bus.itertuples()]
        + [MonitoredElement(id=r.global_id, type="switch", kind="switch", name=r.name) for r in net.switch.itertuples()]
    )
    contingencies = [Contingency(id="BASECASE", name="BASECASE", elements=[])] + [
        Contingency(id=r.origin_id, name=r.name, elements=[GridElement(id=r.global_id, type="line", kind="branch")])
        for r in net.line.itertuples()
    ]
    return Nminus1Definition(monitored_elements=monitored, contingencies=contingencies)


@pytest.fixture
def net() -> pp.pandapowerNet:
    return create_net()


def _run(net: pp.pandapowerNet, *, freeze: bool) -> LoadflowResults:
    return run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=_build_nminus1_definition(net),
        job_id="test",
        timestep=0,
        cfg=ContingencyAnalysisConfig(
            method="ac",
            min_island_size=2,
            parallel=ParallelConfig(n_processes=1, batch_size=None),
            runpp_kwargs={"lightsim2grid": False, "enforce_q_lims": True},
            freeze_net_columns=freeze,
        ),
    )


def test_the_real_pipeline_writes_nothing_undeclared(net: pp.pandapowerNet) -> None:
    """With the barrier armed a full N-1 must complete: no outage writes an undeclared column."""
    results = _run(net, freeze=True)

    assert not results.branch_results.empty


def test_barrier_changes_no_results(net: pp.pandapowerNet) -> None:
    """The barrier only marks storage read-only, so results must be identical either way."""
    without = _run(net, freeze=False).branch_results
    with_barrier = _run(create_net(), freeze=True).branch_results

    pd.testing.assert_frame_equal(without, with_barrier)


def test_barrier_catches_a_stale_map(net: pp.pandapowerNet, monkeypatch: pytest.MonkeyPatch) -> None:
    """The canary: drop a column pandapower provably writes, and the barrier must fail the run.

    Without this, a barrier that has quietly stopped working (a pandas release reshaping the
    block manager, say) would report a clean run forever. Every contingency here outages a
    line, which clears ``line.in_service``; removing that column from the map leaves it shared,
    so the write lands in the base-case net and the barrier must stop it.
    """
    stale = dict(onc.MUTABLE_COLUMNS_BY_TABLE)
    stale["line"] = ()
    monkeypatch.setattr(onc, "MUTABLE_COLUMNS_BY_TABLE", stale)

    with pytest.raises(ValueError, match="read-only"):
        _run(net, freeze=True)


def test_barrier_refuses_to_pass_vacuously(monkeypatch: pytest.MonkeyPatch) -> None:
    """A barrier that freezes nothing proves nothing, and must say so rather than succeed."""
    monkeypatch.setattr(onc, "_freeze_table_columns", lambda _frame, _mutable: (0, [], []))

    with pytest.raises(RuntimeError, match="froze nothing"):
        onc.freeze_net_columns(create_net())


def test_barrier_reports_shared_columns_it_cannot_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """A *shared* column with no freezable storage (e.g. Arrow-backed) must fail, not be skipped.

    Only sharing makes it dangerous: a write into a deep-copied table cannot reach this net.
    """
    monkeypatch.setattr(onc, "_freeze_values", lambda _values: False)

    with pytest.raises(RuntimeError, match="cannot be made read-only"):
        onc.freeze_net_columns(create_net())


def test_copies_inherit_the_barrier_from_the_frozen_source() -> None:
    """Freezing the source once is enough: the copy shares those columns and so their flag.

    This is what lets the barrier cost nothing per outage. The declared-mutable columns are
    reassigned from a deep copy, which yields writable storage, so the outage's own writes work.
    """
    net = create_net()
    onc.freeze_net_columns(net)
    copied = onc.copy_net_for_outage(net)

    with pytest.raises(ValueError, match="read-only"):
        copied["line"].loc[0, "max_i_ka"] = 99.0
    copied["line"].loc[0, "in_service"] = False  # declared mutable: still writable
    assert bool(net["line"].loc[0, "in_service"]) is True, "the write must not reach the source"


@pytest.mark.parametrize(
    "dtype",
    ["float64", "bool", "object", "Int64", "boolean", "string[python]"],
)
def test_every_dtype_in_use_can_be_frozen(dtype: str) -> None:
    """Freezing must reach the real storage for every dtype family the nets carry.

    Numpy blocks are frozen through ``flags``; an ExtensionArray has none, so the numpy arrays
    it wraps are frozen instead. Both must make an in-place write raise.
    """
    values = pd.array([1, 0, 1], dtype=dtype) if dtype != "object" else np.array(["a", "b", "c"], dtype=object)
    frame = pd.DataFrame({"immutable": pd.Series(values)})

    frozen, unfreezable, skipped = onc._freeze_table_columns(frame, ())

    assert frozen == 1, f"{dtype} was not frozen"
    assert not unfreezable
    assert not skipped
    with pytest.raises(ValueError, match="read-only"):
        frame.loc[0, "immutable"] = values[1]


def test_declared_mutable_columns_stay_writable() -> None:
    """The barrier must not block the writes the map explicitly allows."""
    frame = pd.DataFrame({"in_service": [True, True], "name": ["a", "b"]})

    frozen, unfreezable, skipped = onc._freeze_table_columns(frame, ("in_service",))

    assert frozen == 1  # "name" only
    assert not unfreezable
    assert not skipped
    frame.loc[0, "in_service"] = False  # declared mutable: must still be writable
    with pytest.raises(ValueError, match="read-only"):
        frame.loc[0, "name"] = "changed"


def test_columns_left_by_a_block_split_are_still_frozen() -> None:
    """Detaching one column splits its block; the survivors must still be frozen.

    Those survivors come back as a view into the original buffer - a different block object over
    the same memory - so a write into one reaches the source net. Freezing has to cover them.
    """
    source = pd.DataFrame({"mutable": [1.0, 2.0], "shared": [3.0, 4.0], "also_shared": [5.0, 6.0]})
    copied = source.copy(deep=False)
    copied["mutable"] = source["mutable"].copy(deep=True)  # splits the float64 block

    assert np.shares_memory(source["shared"].values, copied["shared"].values)

    frozen, unfreezable, skipped = onc._freeze_table_columns(copied, ("mutable",))

    assert frozen >= 1
    assert not unfreezable
    assert not skipped
    with pytest.raises(ValueError, match="read-only"):
        copied.loc[0, "shared"] = 99.0
    copied.loc[0, "mutable"] = 99.0  # detached and declared mutable: must still be writable


def test_deep_copied_tables_are_outside_the_barrier() -> None:
    """A deep copy of a frozen array is writable again, so unlisted tables are not covered.

    They cannot corrupt the source either, for the same reason - but it does mean the barrier
    says nothing about a table pandapower adds that nobody has classified.
    """
    net = create_net()
    net["controller"] = pd.DataFrame({"in_service": [True], "order": [0.0]})
    onc.freeze_net_columns(net)
    copied = onc.copy_net_for_outage(net)

    copied["controller"].loc[0, "order"] = 1.0  # no raise: deep-copied, hence writable
    assert float(net["controller"].loc[0, "order"]) == 0.0, "the write must not reach the source"


def test_row_mutating_tables_are_deep_copied_and_independent() -> None:
    """``gen``/``sgen`` gain and lose rows per outage, so the copy must own its own storage.

    ``assign_slack_per_island`` promotes an ``sgen`` to a ``gen`` when it picks an island's
    slack, dropping one row and creating another. A shallow copy shares the index and could not
    represent that.
    """
    net = create_net()
    copied = onc.copy_net_for_outage(net)

    for table in onc.ROW_MUTATING_TABLES:
        assert not copied[table].empty, f"the fixture must exercise {table}"
        assert not np.shares_memory(net[table]["p_mw"].values, copied[table]["p_mw"].values)

    # Adding and dropping rows on the copy must leave the source untouched.
    before = len(net["sgen"])
    copied["sgen"] = copied["sgen"].drop(copied["sgen"].index[0])
    pp.create_gen(copied, bus=0, p_mw=1.0, vm_pu=1.0, name="promoted", origin_id="promoted_id")
    assert len(net["sgen"]) == before
    assert len(net["gen"]) == 1


def test_row_mutating_tables_cannot_be_listed_as_column_wise() -> None:
    """Listing one of them would silently reintroduce a hazard the write barrier cannot see.

    Inserting a row reallocates rather than writing in place, so no read-only flag fires. The
    module refuses the combination at import time rather than trusting a comment to hold.
    """
    assert not onc.ROW_MUTATING_TABLES & onc.MUTABLE_COLUMNS_BY_TABLE.keys()


def test_row_mutating_tables_are_left_writable() -> None:
    """The barrier skips ``gen``/``sgen``: it would buy nothing and could bite a real write.

    They are deep-copied, so the outage copy has writable storage whatever we do here, while
    ``assign_slack_per_island`` writes ``gen.slack`` in place on this net once per run.
    """
    net = create_net()
    onc.freeze_net_columns(net)

    for table in onc.ROW_MUTATING_TABLES:
        net[table].loc[net[table].index[0], "in_service"] = False  # must not raise


def test_a_frozen_net_can_be_run_again() -> None:
    """Freezing mutates the caller's net, and multi-timestep callers reuse the same object.

    Every table is frozen, ``res_*`` included, so this only works because ``runpp`` replaces
    those frames rather than writing into them. If that ever changes, this test catches it
    before a second timestep does.
    """
    net = create_net()
    pp.runpp(net, lightsim2grid=False)
    onc.freeze_net_columns(net)

    pp.runpp(net, lightsim2grid=False)  # a later run must still be able to solve it

    assert not net.res_bus.empty
