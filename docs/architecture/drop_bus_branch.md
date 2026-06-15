# Drop Bus-branch support

In the early days of ToOp, development happened on bus-branch grids (such as UCTE files). However, by now the support for bus/branch grids is mostly gone. Asset topologies require node/breaker information and also the export functionalities are built on node/breaker grids.

Note that dropping bus/branch support will not remove the ability to optimize bus/branch grids alltogether. It only requires a preprocessing routine that amends the bus/branch grid using common assumptions to a node/breaker grid, e.g. by introducing asset bays manually.

## Why we didn't do it

A lot of tests build on bus/branch grids (ieee grids). It would be wise to introduce a general routine that performs bespoke node/breaker amendment for all test grids, so generally all tests run on node/breaker grids. Even though the jax-based tests might not need it everywhere, for comparing against the postprocess runners this would make sense. 

## How to deal with pandapower

Pandapower tests are fully bus/branch because we didn't implement the asset topologies properly for node/breaker. We might end up in a situation where it would be much easier to either drop bus/branch AND pandapower support or to keep both in. If such a situation arises, it shall first be checked whether pulling in the asset topology support for pandapower might be feasible. If we end up in a stalemate situation, redecide as dropping pandapower completely is not intended at the moment.
