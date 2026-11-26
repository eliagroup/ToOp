---
title: "ADR-007: Using the violations dataframe"
status: Decided
date: 2026-11-16
decision-makers: []
consulted: []
informed: []
---

The current ToOp contingency library uses the .bus_results / .branch_result dataframes with subsequent polars based data filtering. These dataframes are a major bottleneck to performance, and even though a [draft pypowsybl PR](https://github.com/powsybl/pypowsybl/pull/1312) might massively accelerate them, they are still very slow. The intended way is to add limits and then use the limit violations dataframe in powsybl's security analysis API, so the security analysis can decide in-the-loop which results to store and which to discard. This reduces the amount of storage required in the loop and increases performance beyond what is possible with .bus_results/.branch_results even in optimized form.

Hence, we decide to rework towards violations dataframes. This means we will likely have to implement in powsybl:

- Limits on switches/terminals in the security analysis (similar to switchflow)
- Voltage jump limits

However, this approach seems cleaner. Until the missing features are implemented in powsybl, we will retain the legacy .bus_results/.branch_results path.

