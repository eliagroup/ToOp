# Switching distance

This document briefly describes the idea behind how to constrain switching distance in the solver context.

## Physical vs Electrical distance

First, we define physical and electrical switching distance. The physical switching describes how each physical switch is set. Meaning, a branch can be connected to multiple busbars and all constraints for busbar failures need to be upheld. The electrical switching defines the layout in terms of which branch is connected to which electrical busbar. Busbars which are connected by a closed coupler are merged in that, so a branch can only be on one busbar at a time. In our case, we also add a restriction of having at most two separated electrical busbars, i.e. to switch at most a two-way split, even if there are more busbars in the substation. The electrical switching is suited for optimization, as the complexities of a physical switching can be omitted and symmetries can be effectively avoided. However, this brings some complication for computing the physical switching distance to a topology. The operators eventually care about the physical switching distance, so we need to proxy this distance as good as possible.

Furthermore, the starting point of the switching distance is not necessarily known at preprocess time. For the first optimized timestep, the unsplit grid from the gridmodel can be used, but for later switchings, the result of the optimization run of previous timesteps shall be used. Furthermore, there is an interest in storing topologies long-term, which is why we introduce the concept of an *Asset Topology*. This is a topology defined on assets of the grid with their ids attached, so it can be read for later timesteps.

## Computing switching distance

As we don't know the physical layout but only the electrical layout, we have to proxy the switching distance. For this, we use an array of possible translations from the physical pre-existing layout to an electrical, binary representation. The switching distance is then defined as the minimal hamming distance to all translations. Especially cases where a branch is connected to multiple busbars can't be clearly translated, which is why we have a two way translation from physical layout to the translation set: First, we disconnect branches that are connected to multiple busbars in such a way that the busbar with the most other branches connected to it is retained. Then we enumerate all possible binary assignments of physical busbars to electrical busbars through opening of any coupler. This set is then passed to the solver and a hamming distance is computed.

Examples:

A physical assignment could look like this:

bus 1    1   0   0   0   1   0
bus 2    0   0   1   0   0   0
bus 3    0   0   0   0   0   1
bus 4    0   1   0   1   0   0
        br1 br2 br3 br4 br5 br6

couplerstate:
bus1 - bus3: closed
bus2 - bus4: closed

The electrical assignment could look like this:

bus a   1   0   0   0   1   1
bus b   0   1   1   1   0   0


The problem is now, how do we find out the distance between these assignments. In principle, we would need to walk through all combinations of opening/closing a coupler and enumerating all possible assignments. However, as we are only interested in switchings that result in two separate clusters, we can use a graph algorithm involving edge cuts to enumerate all possible assignments. This is done in the `switching_distance` module.
