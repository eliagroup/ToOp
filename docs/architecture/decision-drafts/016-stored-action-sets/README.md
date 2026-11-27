---
title: "ADR-016: Stored action sets"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: []
---

Currently, ToOp relies on always generating its own action set through enumeration in the importer pipeline. However, some use cases require passing in a pre-computed action set. If this is done, the action set should merely be translated into the optimized DC format during the importing (and checked for feasibility, maybe some actions need to be dropped). The importing stage should thus define a data format (maybe from 006-powsybl-ra-api?) for topology actions and be able to read it. If such a file is specified, the enumeration may be skipped. Pre-configured actions could also be used to define a subset of trusted actions (see 015), so the enumeration might add additional actions.
