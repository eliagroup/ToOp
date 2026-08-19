# Wall of shame

Known shortcuts that might lead to issues later on / bugs we did not have time to fix yet.

## Busbar outage articulation nodes
Busbar outages that would split a node are currently excluded from the busbar outages. This could lead to 

## Islanding branches
Islanding branches are currently excluded from the n-1 analysis, because the lodf can not deal with it

## AC runs on reduced outages
Since the dc solver cant compute all outages (see islanding) some are excluded from the N-1 that is also used in AC.

## AC Topologies that improve convergence rate, but make overload worse are rejected in AC
If we have a topology that improves the grid situation in a way, that an outage at least converges but shows even more overloads, this topology is excluded although the situation might have been improved. We just did not know about the overloads in AC before.

# powsybl NO_CALCULATION is treated as not successful
in the AC contingency analysis NO_CALCULATION-status is treated as not successful, although this could also mean branch-was-already-disconnected or other things. This has to be evaluated.
