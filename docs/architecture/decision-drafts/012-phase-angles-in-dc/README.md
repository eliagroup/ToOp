---
title: "ADR-012: Phase angles in DC"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: []
---

Currently we do not check phase angles in DC, even though the DC linearization in principle does have phase angles. It is even possible to include them directly in the PTDF, likely not requiring much adjustments to lodf/bsdf/etc. We just haven't done it yet, hence the AC validator has the job of filtering out aggressive topologies based on phase angle.

A sensible rework would be to extend the current PTDF formulation to include a phase angle computation for split stations and disconnected lines. The phase angle between the two sides of the opened coupler or the two sides of the disconnected branch would be the dimension to look out for. Other phase angle differences in the grid will likely be influenced as well, but it might be sufficient in the DC computation to only check the directly opened switches, as we expect the highest impact directly in the opened switch in most cases.
