# N Minus 2

To catch exquisite failure cases, we implement an N-2 calculation routine. The main function for this is [`n_2_analysis`][toop_engine_dc_solver.jax.nminus2_outage.n_2_analysis]. Computing an N-2 analysis is only part of the process; sensible evaluation requires splitting the computation into an *unsplit analysis* during preprocessing and a *split analysis* under the applied topology during the solving process.

A N-2 analysis consists in our case of
- a list of first-level branch failures, which are all branches at split relevant substations. We will refer to them as L1 cases
- a list of second-level branch failures, either all N-1 branch cases from the normal N-1 analysis or a subset of these. We will refer to these as L2 cases

In general, we do not consider injection or multi-outages in the N-2 cases (yet) and restrict ourselves to computing only
the N-2 problems of branches directly at the split relevant substations.

## Unsplit Analysis

As we don't want to heal N-2 problems, but rather prevent causing them, we need to compare the N-2 problems under the topology with the N-2 problems present in the split grid. Hence, we need to establish a baseline N-2 status of all relevant N-2 cases. Hence, we go over all existing branches at all relevant substations and compute the N-2 overload energy. We store this in a registry so that the split analysis can lookup the reference values. The time to run an unsplit analysis was around 10 seconds on the 50Hz UCTE grid, additionally increasing the preprocessing time.

During this analysis we can furthermore cut down the number of N-2 cases computed by selecting our L2 cases such that only those with the most drastic change towards the N-1 case are included. This is not implemented yet

## Split Analysis

During the solving process, a N-2 analysis is computed with the set of L1 cases being all branches ending directly at a split substation. The overload energy for each L1 case is summed and compared with the baseline case. If it is worse, then the diff is summed up as a penalty. Hence, making it better under N-2 does not yield a reward, only making it worse. There are two additional edge cases that can happen. In case some L2 cases that previously converged don't converge anymore, there will be a fixed penalty *more_splits_penalty* for each non-converged case added to the total penalty. If a L1 case does not converge anymore, this is penalized by multiplying the *more_splits_penalty* by the number of L2 cases and adding that to the penalty. However, such a case should also be caught within the normal N-1 analysis.
