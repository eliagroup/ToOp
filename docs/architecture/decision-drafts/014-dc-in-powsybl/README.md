---
title: "ADR-014: DC solver behind powsybl loadflow interface"
status: Draft
date: 2026-11-27
decision-makers: []
consulted: []
informed: []
depends_on: []
---

Our DC solver is basically a very specialized solver that can only perform topology actions that have been pre-recorded. A question would be if it is easily possible to include the powsybl DC loadflow API in between the optimizer and loadflow solver, such that the optimizer could also work with another solver that implements the powsybl contract.

There are some indications that this is a tricky endeavour and might not work well:

- The powsybl API is a java API, while we do all development in python
- The powsybl API is cpu-native, while our optimizer-solver boundary currently does not leave the GPU.
- Our solver relies on pre-computed information per action, which is not foreseen in the powsybl API

The current tendency is to reject the ADR.
