---
title: "ADR-010: Time-series based optimization"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: ["009"]
---

Time-series based optimization means including timesteps into the DC solver. Here, we have a tricky choice to do. Fundamentally, timesteps can be included in 3 ways:

- Only through injection time-series with constant PTDF -> this requires the structure of the grid to not change, so it is effectively only useable for stochastic optimization or grid planning use-cases, where no connections/disconnections happen between the timestep and the underlying topology stays constant. For a full implementation, this route is not sufficient.
- Low-rank based: The PTDF changes due to structural changes on the grid are represented as diffs towards a common PTDF. Each timestep thus needs to first apply the low-rank updates before any loadflows can be evaluated. The compute_bsdf_lodf_static_flows would then get another responsibility. This approach can be batched, so that not all PTDFs of all timesteps need to be in VRAM. However, generating the timestep PTDFs at the beginning of each batch computation is costly, might be the main dominating factor.
- PTDF copies: For every timestep, a new PTDF is computed. Ideally, they are padded towards a common shape so that they can be stored in one tensor. This will however eat quite some memory, especially if we expect a 15 minute resolution in future day-ahead planning.

A first experiment should weigh the low-rank against the ptdf copy approach on reasonable grid sizes + hardware. My instinct points towards the copies approach. However, this also depends on the optimization strategy:

- Core-timesteps then walk: This resembles the thinking of the operators. First identify the timesteps that are at the peak of the congestion, usually there are only 2 or 3 structurally different congestions on every given day. Then optimize these identified timesteps in single-timestep mode. After the optimization, walk the topologies to the left+right timewise until the congestion situation worsenes significantly. This process returns a set of topologies and their validity, which is fed into a merging decision that sets the concrete boundary of the topologies. The switchover point does not need to be optimal, as realtime operators will adjust that switchover on-the-fly anyway, so a rough idea is sufficient. This approach also opens another sub-decision: It might be sensible to run the GA on the core timesteps for a while before starting the walk+merge functionality, as the first strategies found are usually pretty bad. This might mean that DC doesn't produce any results and the AC service is idling.
- All-timesteps jointly: Create a strategy repertoire that holds topology information for all timesteps and validate each strategy on all timesteps. This joins the two problems of finding the best topology and finding the duration of the topology into one problem, so it is likely a bit slower to converge but might produce better results.

Also this decision might require an experiment. The result might also influence the decision on the representation format.
