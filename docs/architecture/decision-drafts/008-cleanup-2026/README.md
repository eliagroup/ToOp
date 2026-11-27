---
title: "ADR-008: End of year cleanup 2026"
status: Decided / In Implementation
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: []
---

Currently, we carry some unsupported features around that are not needed in a hot path or have been superseeded. We want to drop them, which makes maintenance tasks such as ty integration much easier.

## Contingency lib

- ac loadflow service

## Importer

- UCTE Importer support (will be replaced by bus/breaker import)
- dacf white liste/black liste, power factory import (to be decided if they move to a separate helper module or are dropped)
- pandapower support (completely? or retain some helper functions to build ptdf/etc? - some academics use the pandapower based import, we might just drop the action set enumeration)
- Some functions: 
    - get_branches_including_limits_and_dangling_lines()
    - get_loadflow_based_line_limits()
    - get_loadflow_based_trafo_limits()
    - get_all_border_line_limits()
    - get_all_dso_trafo_limits()
    - create_new_border_limits()


## DC Solver

- The jax topology_looper (superseeded by bruteforce optimizer)
- To be discussed: Skeleton busbar outage logic? MODF with islanding not always better
- Benchmarks (move to toop_engine_benchmarks, split out what is needed for jupyter to helper module)
- config.py
- batching.py
- Dependency to importer (we have an import cycle)
- AbstractLoadflowRunner -> create a functional API that wraps the contingency lib shape but with action indices (discuss if we need this)

## Optimizer:

- Unused metrics underload_energy / transport
- Move interfaces to interfaces package
- QDax dependency
- benchmarks (move to toop-engine-benchmarks)
- CLI interfaces
- RealizedTopology

## Others:

- Moving test grids into their own package + adding the internal test grids
