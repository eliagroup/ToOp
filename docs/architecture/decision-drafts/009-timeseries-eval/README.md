---
title: "ADR-009: Time-series based evaluation"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: ["002", "011"]
---


The business requires time-series based optimization and evaluation. We split this into two proposals as evaluation is significantly easier to implement. The flow for time-series based evaluation is an AC based, human triggered evaluation of a topology across timesteps. Human-triggered evaluation is only possible with the postgres rework (002) and arbitrary topology evaluation (011).

There are two modes to this evaluation:
- **walk**: Start at a defined "core" timestep and move left + right until the contingency results become significantly worse
- **full**: Evaluate a given list of timesteps exhaustively

Fundamentally, this means we will have a strategy/topology separation. The strategy concept is defined as a topology over time. We already have remnants of this here and there, but with this ADR, we will have to fully commit to it and only consider strategies from now on, even if it contains only one timestep. Meaning, we will have to go through all storages/communications and use strategies everywhere. Furthermore, the AC worker might want to parallelize across timesteps. We will have to use metric aggregations as defined in the optimizer to sum/max metrics across timesteps. And what will be difficult, we will need a policy for how to deal with missing or changing elements. It could happen that the substation switching state or the elements in the grid change between timesteps. That might result in a topology not applying at all because a switch is missing or returning far worse results because the switches not in the SwitchUpdateSchema have changed and thus the resulting assignment is different. We will require some sort of concept, in the worst case we will have to refactor our topology storage again to not only store SwitchUpdates for switches that have been touched but for all switches in a station, such that these errors can be detected.