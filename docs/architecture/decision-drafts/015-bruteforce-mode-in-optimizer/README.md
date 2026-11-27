---
title: "ADR-015: Bruteforce mode in optimizer"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: ["001", "016"]
---

Currently, there is a draft bruteforce optimizer implementation next to the main dc optimizer. However, it would be required to join them into one to represent the following use case:

- First, ToOp exhaustively bruteforces all single actions stored in a "trusted actions" action set that is specified by operators and populates the initial repertoire with them
- Then, ToOp goes into normal operation mode and randomly explores the search space for the remainning runtime.

This requires unifying the bruteforce and ga optimizers into one shared format. Furthermore, 016 and 001 are required to specify which actions are trusted and which are not.
