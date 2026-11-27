---
title: "ADR-006: Moving behind the powsybl RA API"
status: Draft
date: 2026-11-26
decision-makers: []
consulted: []
informed: []
depends_on: ["001"]
---

There is a multitude of remedial action optimizers being developed in our product line, in MCCS and in the open source community. Most notably, the internal VoltControl project and OpenRAO are RAOs that do different things, but not that different. Every RAO needs somewhat similar input information. Observed elements, contingency definitions, objective functions, constraints, remedial action freedom/potentials are all shared between the RAOs. Every RAO builds its own format, tailored to the capabilities of the specific RAO. As part of the Remedial Action API effort in [powsybl](https://github.com/powsybl/powsybl-core/issues/4223), it is planned to develop towards a shared API that represents the superset of all RAO capabilities in a unified way, such that shared elements can be used by every RAO. 

Without this API, glue code between the optimizers is necessary to build a joint workflow. Currently, ToOp and OpenRAO are glued together by TopRAO. Contingency definitions are translated, sensible defaults are found for parameter that OpenRAO needs but ToOp doesn't set and the topologies are carried over. With a unified RA API, the TopRAO layer would disappear and ToOp could take most of what it needs through a unified RA API.

It is a question however, which features may become part of a shared RA API. There are some no-brainers:

- Contingency cases
- Observed elements (including hold/cap/... semantics)
- Objective function
- Constraints

Some which need refinement:

- Remedial actions (how to deal with action enumeration?)

Some which should not be part of the shared API:

- Batchsize
- GA setup/repertoire
- etc


