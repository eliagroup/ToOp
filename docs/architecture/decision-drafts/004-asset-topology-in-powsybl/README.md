---
title: "ADR-004: Moving asset topologies into powsybl"
status: Draft
date: 2026-11-26
decision-makers: []
consulted: []
informed: []
depends_on: []
---

Currently, the asset topology logic is one of the heavier logics in ToOp, at the core of the action enumeration. It provides a semantic understanding of asset bays, station flexibility and the role of switches by looking at the structure of the station and comparing it against hard-coded versions.

Currently, this logic is maintained inside ToOp as part of the Importing procedure. However, it seems sensible to offer the semantic classification of assets in a bay as a general service, potentially to users which do not even do topology optimization. As a central powsybl service, we would

- Gain performance: The asset topology classification is mostly a graph walk, which will likely run more performantly in java than in python
- Gain adoption: More people using the asset topology classification would mean that potential changes such as integrating the nordic station layout would be easier to integrate

At the disadvantage, this would mean that the importing can no longer work on pandapower. As the ToOp project is moving further towards powsybl, this might be a tradeoff worth doing, but certainly requires a rethink.