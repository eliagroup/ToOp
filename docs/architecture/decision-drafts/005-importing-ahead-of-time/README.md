---
title: "ADR-005: Importing ahead of time"
status: Draft
date: 2026-11-26
decision-makers: []
consulted: []
informed: []
depends_on: ["001"]
---

Operationally, ToOp runs in the Day-Ahead process as soon as market closes and injection forecasts are available. From the moment the data is available, there is intense time pressure on ToOp to deliver results. So there is an incentive to move as much computation ahead of time as possible. The structure of the grid may already be available ahead of time. Resultingly, tasks that only require the topology of the grid and not any injections could happen before the critical time. Coincidentally, the asset topology logic and action enumeration are major performance bottlenecks in the importing. Hence, it makes sense to split up the importing into a injection-dependent and an injection-independent part.

Fundamentally, there is a risk introduced by computing things ahead-of-time: If the operational data received does include topology changes, even though it is expected not to include any, ToOp must at least raise. Even better would be to adjust the action set such that the topology change influences the action set. Then, it might be possible to run the ahead-of-time enumeration once on the fully connected grid with no maintenances and only rerun when new assets are built. It might be significantly easier though to just check but require the same topology at least on the switching area.
