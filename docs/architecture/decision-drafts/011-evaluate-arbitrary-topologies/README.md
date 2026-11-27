---
title: "ADR-011: Evaluate arbitrary topologies"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: ["002"]

---

The current communication between AC and DC is action-index based. This makes sense as long as all topologies that the AC validator ever has to validate come from a shared action set. However, it seems that custom, human-generated topologies should be evaluated as well in the future. This only becomes relevant after implementation of postgres (002). The flow is roughly that ToOp presents a topology to the operator, he/she opens that topology but adjusts a few switches and then wants to feed the topology back into ToOp as a custom topology. After feeding it back, metrics and contingency analysis results shall be generated the same way as for all other topologies.

This breaks the assumption of action-set only. Hence, a few reworks are required:
- In the communication of topologies, the SwitchUpdateSchema is needed instead of action indices
- To retain information about action indices, the SwitchUpdateSchema should get annotations which switches came from which action index (if they are not custom)
- SwitchUpdateSchema should also group them into reassignment+coupler opening or disconnection based switching actions
- SwitchUpdateSchema needs to support PSTs (ongoing work by Nico)
- We need to rethink the hashing, probably hash the SwitchUpdateSchema

This will also increase the memory footprint of our communications, but makes it easier for the frontend to display topology information as the action set lookup that currently happens in the listener is not needed anymore.
