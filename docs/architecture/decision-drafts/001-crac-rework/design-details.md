# CRAC rework ToOp - design details

This document belongs to ADR-001 in [README.md](README.md). The ADR takes the decision and sets the high level picture.
This document holds the terminology, examples and the details that are to be considered when taking the approach.

## Terminology

Note that this terminology is preliminary, we plan to rename them in a second refactor later.

- **Optimized element**: A grid element that, if it exceeds a limit, should be healed to be below the
  limit after the optimization
- **Non-Worsening:** (do-not-make-worse) A grid element that should not be healed if it is
  above its limit, but should also not be pushed even further above its limit. Something that is neither monitored nor
  optimized will not appear in the loadflow results.
- **CRAC:** ***C***ontingency list, ***R***emedial ***A***ctions and additional ***C***onstraints - basically all the
  configs that go into ToOp
- **Double limit:** Every monitored and optimized element can have a double limit. In the case that
  the element was below limit before the optimization, a buffer zone below the actual limit is
  introduced to keep it safely away from that limit post-optimization.
- **Contingency cases:** The N-1 outage cases to be computed as part of the security
  assessment.
- **Artificial limit:** A limit that is determined based on the initial loadflow and either the double
  limit and monitored rules.
- **Generative CRAC API** An API that implements rule based, heuristic or any other logic to generate specific CRAC API
  files. E.g. switch everything in this grid area
- **Specific CRAC API** An API which describes exactly the actions available, including specific switching sequences
  and an ID for each entry. e.g. A single topological action including all operated switches and their target
  open/close state.

## Examples

- Element is optimized, not monitored (and double limit band from 90%)
  - \>100% → artificial limit is 100%
  - \>90%, <100 → double band logic sets artificial limit to exactly the loading
  - <90% → artificial limit 90%
- Element is monitored, not optimized
  - → artificial limit is exactly the loading
- Element is monitored and optimized
  - \>100% → artificial limit is 100%
  - <100% → artificial limit is exactly the loading
- Element is neither
  - → will not appear in optimization or results.

## Rules for limits

DSO Trafos: Their initial loadflow loading (N-1) is set as an artificial limit for the optimization.
There is a penalty when the flow is increased above that, but no bonus for reducing it.

Lightly loaded elements: Elements which are, pre-optimization, below their operational limit (e.g.
<100%) shall receive a penalty once they move upwards in a defined double limit band (e.g.
90%).

Different limits for Redispatch vs Topology: It might be fundamentally thinkable to assign
different limits to the topology part vs redispatch. However, this would add another layer of
complexity and we currently assume that all limits that hold for ToOp will also hold for ORAO.

TATL/PATL: This proposal is not formulated for TATL yet, everything only holds for PATL.

## Displaying results for alternative limits

When displaying results for alternative limits, we should find a UI concept. It is not intuitive to
just adjust the limit in the display and claim a monitored item sits at 100%, while it actually is at
125%, even though for our optimizer this is the case. One option would be a toggle (i.e. show
"natural" load vs artificial), however in the artificial load scenario we would see a lot of elements
around 100% (everything that is monitored or in the double limit band). We could also add a
toggle for "optimized" only, which would make these odd trafos disappear that we have
currently. Then, we should still answer how we display the double band logic.

To be able to convey these thoughts, we would require the option of adding multiple types of
limits. Currently, loading is a property that is hard-coded into the loadflow results. This will not
scale with multiple artificial limits. I.e. we should generate the loading value on-the-fly and
potentially drop it from the loadflow result interface.
