# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Policy for dropping loadflow result rows that carry no decision value.

The branch and node result tables dominate the size of a stored loadflow result: they hold one row per
(timestep, contingency, element[, side]) combination, whether or not the value says anything. A CO/CB pair at 12% loading
and a bus sitting at nominal voltage are computed, transported and stored exactly like a violation is.

This module defines *what* counts as uninteresting. Translating a policy into a predicate, and applying it, belongs to the
contingency analysis package (``toop_engine_contingency_analysis.result_filter``); this one only carries the settings, so
that a stored result can record the policy it was produced under without depending on a backend.

Semantics:

- Filters are a **disjunction of keep-conditions**: a row survives if any active filter wants it.
- A threshold of ``None`` **contributes no keeps**. It does not mean "keep everything" - that is what an entirely inert
  filter means, and the model validators reject the combinations where the difference would be surprising.
- **Unknown is not small.** Where the tested column is null or NaN - a branch with no rated current, a node with no
  voltage limits, or any row of a contingency that did not converge - the row is always kept. We cannot assess it, so we
  do not discard it.
- The jump filter is an **additive exemption** on top of the base threshold, never a filter of its own.
- The same thresholds apply to N-0 and N-1; ``retain_basecase`` is the only N-0-specific control.

Branch loading jumps - the branch-side counterpart of :attr:`NodeLoadflowResultFilter.vm_basecase_deviation_above` - are
deliberately absent: unlike the node voltage jump, they need a basecase-relative column that ``BranchResultSchema`` does
not have yet. They follow in a later PR, for which the design is already settled:

- ``loading`` is built from a current *magnitude* in both backends, so it cannot express a flow reversal on its own. Sign
  it by the flow direction instead - ``p`` is signed and sits in the same frame, per side, in both backends - with a
  plain ``copysign`` and no deadband.
- Persist the delta against N-0 as a ``loading_basecase_deviation`` column on ``BranchResultSchema``, mirroring
  ``vm_basecase_deviation`` on the node schema, so a reader can see why a row survived.
- Source the base-case value the way the node side already does: pandapower snapshots it once into ``ResultConstants``
  (see ``basecase_vm``), powsybl self-joins the base-case rows it already has.
- Do not sign ``loading`` itself. ``count_critical_branches`` tests ``loading > 1.0`` and ``compute_max_load`` takes
  ``loading.max()``; both would quietly stop seeing reversed overloads.
"""

from pydantic import BaseModel, Field, model_validator


class BranchLoadflowResultFilter(BaseModel):
    """All filtering settings concerning branch results.

    The effective filtering reads in such a way that if any filter wants to keep an element in, it will be kept in and not
    filtered out.

    Some None combinations are not valid and disallowed by model validator:
    - loading_above = None, retain_basecase = False -- This would never remove N-0 values even though the instruction is to
    do so, as there is no filter that can ever apply to basecase results.
    """

    loading_above: float | None = Field(default=None, ge=0.0)
    """Keep branch rows with |loading| >= this (1.0 == 100% of rated current). None disables, i.e. all branch results will be
    in the file. If an element does not have a rated current defined, it will never be filtered out."""

    retain_basecase: bool = True
    """Whether the basecase results should be retained always. If not, loading_above will apply to both basecase and
    contingency case similarly.

    .. warning::
       Disabling this drops N-0 rows like any other. Anything that reads the base case as a reference then has nothing to
       read: ``compute_metrics`` asserts that every N-0 metric is computable and will fail on a quiet base case.
    """

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "BranchLoadflowResultFilter":
        """Reject filter combinations whose effect would not match their reading.

        Returns
        -------
        BranchLoadflowResultFilter
            The validated filter.

        Raises
        ------
        ValueError
            If ``retain_basecase`` is disabled while no threshold can ever apply to basecase rows.
        """
        if self.loading_above is None and not self.retain_basecase:
            raise ValueError(
                "retain_basecase=False requires loading_above to be set: with no threshold, nothing can ever filter "
                "basecase rows, so lifting their exemption would have no effect."
            )
        return self

    def is_active(self) -> bool:
        """Whether this filter can ever drop a row.

        Returns
        -------
        bool
            True if at least one threshold is set.
        """
        return self.loading_above is not None


class NodeLoadflowResultFilter(BaseModel):
    """All filtering settings concerning node results.

    The effective filtering reads in such a way that if any filter wants to keep an element in (i.e. it has a node loading
    below threshold but the voltage jump is above threshold) they will be kept in and not filtered out.

    Some None combinations are not valid and disallowed by model validator:
    - vm_loading_above = None, retain_basecase = False -- This would never remove N-0 values even though the instruction is
    to do so, as there is no filter that can ever apply to basecase results (vm_basecase_deviation_above will never apply to
    basecase).
    - vm_loading_above = None, vm_basecase_deviation_above != None -- This would mean all nodes with a low voltage jump will
    be filtered out, even if they went from 1.3pu to 1.301pu. - That seems like a flawed business decision.
    """

    vm_loading_above: float | None = Field(default=None, ge=0.0)
    """Keep all node results in that have a vm_loading above defined threshold. If |vm_loading| >= this, it will stay in.
    If vm_loading is not defined, it will also stay in.

    Unit: a fraction of the distance to the voltage band edge, where 1.0 sits exactly on v_max (or on v_min). Note that
    vm_loading is *signed* - it is negative for undervoltage - so the threshold is compared against its absolute value and
    thereby applies symmetrically in the low-voltage and high-voltage violation directions."""

    retain_basecase: bool = True
    """Whether to retain all basecase results by default. If not, vm_loading_above will also apply to the basecase.

    .. warning::
       Disabling this drops N-0 rows like any other. Anything that reads the base case as a reference then has nothing to
       read: ``compute_metrics`` asserts that every N-0 metric is computable and will fail on a quiet base case.
    """

    vm_basecase_deviation_above: float | None = Field(default=None, ge=0.0)
    """Keep elements where a voltage jump from N-0 to N-1 is larger than this threshold, even if their vm_loading is below
    threshold and they should be dropped. If None, deactivates this filter.

    Unit: percent, matching ``NodeResultSchema.vm_basecase_deviation`` - unlike vm_loading_above, which is a fraction.
    Setting this requires the results to carry that column"""

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "NodeLoadflowResultFilter":
        """Reject filter combinations whose effect would not match their reading.

        Returns
        -------
        NodeLoadflowResultFilter
            The validated filter.

        Raises
        ------
        ValueError
            If the jump threshold is set without a base threshold, or if ``retain_basecase`` is disabled while no threshold
            can ever apply to basecase rows.
        """
        if self.vm_loading_above is None:
            if self.vm_basecase_deviation_above is not None:
                raise ValueError(
                    "vm_basecase_deviation_above is an exemption on top of vm_loading_above, not a filter of its own: "
                    "setting it alone would drop every node that is stable, however far outside its voltage band it sits."
                )
            if not self.retain_basecase:
                raise ValueError(
                    "retain_basecase=False requires vm_loading_above to be set: with no threshold, nothing can ever "
                    "filter basecase rows, so lifting their exemption would have no effect."
                )
        return self

    def is_active(self) -> bool:
        """Whether this filter can ever drop a row.

        Returns
        -------
        bool
            True if at least one threshold is set.
        """
        return self.vm_loading_above is not None


class LoadflowResultFilter(BaseModel):
    """Policy for dropping loadflow result rows that carry no decision value.

    The default is inert: it keeps every row, so that adding this to a call site changes nothing until a threshold is set.

    Only the branch and node tables are filtered. The remaining tables are at most O(n_contingencies) and are left alone;
    switch results have no natural severity column to threshold on.
    """

    branch_filters: BranchLoadflowResultFilter = Field(default_factory=BranchLoadflowResultFilter)
    """Filter policies that decide branch result retention"""

    node_filters: NodeLoadflowResultFilter = Field(default_factory=NodeLoadflowResultFilter)
    """Filter policies that decide node result retention"""

    def is_active(self) -> bool:
        """Whether this policy can ever drop a row.

        Returns
        -------
        bool
            True if either sub-filter is active.
        """
        return self.branch_filters.is_active() or self.node_filters.is_active()
