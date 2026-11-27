---
title: "ADR-013: Sparse PTDF"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: ["014"]
---

ToOp is currently optimized for grids up to ca 10k buses, after that the quadratic memory requirement of a dense inverse slows down the approach so much that it is no longer sensible. All operations that update the dense inverse (as in compute_bsdf_lodf_static_flows) are extremely costly and very low batch sizes are required to even run the problem, even on strong hardware.

However, there is some interest in a europe-scale optimization within the ToOp approach. A fundamental proposal how to tackle these scales is to replicate the same DC/AC structure, but let the DC solver retain sparse LU decompositions instead of dense inverses. As sparse support on GPU is tricky, this would likely mean that the sparse DC solver will run on CPU. Potentially, some of the jax code can be reused as jax also runs on CPU. If we are very sneaky, we might be able to write replacements for only the low-level routines (bsdf/lodf/etc) that touch the PTDF matrix and which are specialized to sparse, and leave all the rest in a shared codebase. The question is then how performant constructions such as the busbar outage skeleton logic would be, I could imagine that a cpu native implementation might outperform it, but it could be possible.

Even better, we would first implement 014 and harmonize our loadflow solver APIs such that replacing our GPU DC solver for the CPU-native powsybl DC solver is easily possible.
