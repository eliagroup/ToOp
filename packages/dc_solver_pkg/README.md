# DC Solver

This is the documentation for the accelerated DC loadflow solver. This serves the purpose of computing a large number of similar DC loadflows in an accelerated fashion. Currently the solver supports the following batch dimensions, i.e. the workload must not change in anything other than these dimensions:

- Branch topology (assignment of branches to busbar A or B)
- Injection topology (assignment of injections to busbar A or B)
- Branch outages

Under the hood, it is using PTDF/(G)LODF/BSDF approaches to achieve this.

If your workflow suits these requirements like it is the case for topology optimization, this solver can help you out.

## Getting started

The main entrypoint to the solver is the [`run_solver`][toop_engine_dc_solver.jax.run_solver] function. This takes four main parameters:

- A set of [`TopoVectBranchComputations`][toop_engine_dc_solver.jax.TopoVectBranchComputations] which stores for each branch end on which busbar they are to be
- An optional set of line disconnections for each topology
- An optional set of [`InjectionComputations`][toop_engine_dc_solver.jax.InjectionComputations]. If not specifies, it bruteforces all possible injections.
- A [`StaticInformation`][toop_engine_dc_solver.jax.types.StaticInformation] dataclass holding all relevant information about the grid and some hyperparameters, obtained through the preprocessing.

By default, it will return sparsified N-0 and N-1 results for every topology passed in and the injection computation from the set of possible injection computations that was chosen.

## Concept overview

There are various concepts that are required to grasp in order to make sense of this solver:

- **Branch/injection topology**: The solver distinguishes between branch topologies in the form of [`TopoVectBranchComputations`][toop_engine_dc_solver.jax.TopoVectBranchComputations] and injection topologies in the form of [`InjectionComputations`][toop_engine_dc_solver.jax.types.InjectionComputations]. The solver will compute the DC loadflow for every combination of branch and injection topology.
- **Injection bruteforcing**: Describes the idea of computing multiple injection topologies for every branch topology. If you know you want to compute multiple injections for each branch set, this can speed up your computation by 2 OOM.
- **Symmetric mode**: This is the opposite of injection bruteforcing, means you have exactly one injection candidate for every branch topology candidate.
- **Static/dynamic information**: There is a distinction between static information, which is the information that is constant for every computation, and dynamic information, which is the information that changes for every computation. The static information is stored in a [`StaticInformation`][toop_engine_dc_solver.jax.types.StaticInformation] dataclass, and changes to this will trigger a recompile of the solver. Hence, effective search can only be done on the dynamic information which is passed as parameters to the solver.
- **AggregateMetricsFn/AggregateOutputFn**: The solver does not assume how to break down the output of the computation. Usually, it is not desired to store the full N-0 and N-1 flows, hence the default behaviour is to only store the worst results. You can overwrite this behaviour by passing custom functions that implement the [`AggregateMetricProtocol`][toop_engine_dc_solver.jax.types.AggregateMetricProtocol] and [`AggregateOutputProtocol`][toop_engine_dc_solver.jax.AggregateOutputProtocol] respectively.

## Preprocessing

The solver can not directly work with grid data in common grid formats (ucte, cgmes, etc). Instead, it needs to load the relevant information for loadflow computations from a backend. Currently, the supported backends are [pandapower](https://pandapower.readthedocs.io/) and [powsybl](https://powsybl.org). During a preprocessing step, the solver will compute the PTDF matrix and other relevant information for the solver. The aim of the preprocessing is to obtain a [`StaticInformation`][toop_engine_dc_solver.jax.types.StaticInformation] dataclass with relevant information for the loadflow solving and a [`NetworkData`][toop_engine_dc_solver.preprocess.network_data.NetworkData] dataclass with additional information useful for postprocessing.

Read more on the [`preprocessing page`](https://eliagroup.github.io/ToOp/dc_solver/preprocessing/).

## Injection bruteforcing

The solver offers an option to perform a bruteforce search for injection candidates in-place. The reason for this is that changing an injection is extremely cheap, it boils down to copying another nodal injection vector. Hence, no BSDF recomputations need to be done if all injection combinations are to be tried anyways. The solver will return the best injection combination based on the chosen metric if the bruteforce-mode is desired. To enable bruteforcing, pass `None` to the injection argument of [`run_solver`][toop_engine_dc_solver.jax.run_solver]. If you don't need injection bruteforcing, you will have to pass the injection combinations that you want to compute. There is a special case, if you have exactly one injection combination per branch topology, then the solver can use [`symmetric mode`][toop_engine_dc_solver.jax.compute_batch.compute_symmetric_batch]. [`run_solver`][toop_engine_dc_solver.jax.topology_looper.run_solver] will automatically choose symmetric mode if it detects this.

If you are computing single batches, you can also use the normal [`compute_batch`][toop_engine_dc_solver.jax.compute_batch.compute_batch] or [`compute_symmetric_batch`][toop_engine_dc_solver.jax.compute_batch.compute_symmetric_batch] mode directly.

## Metric-first mode

If multiple injection combinations are to be computed for each branch topology, then the question is how to aggregate the data. It is infeasible to store the full N-1 matrix for every injection combination due to the large volume of data, especially when bruteforcing, hence the solver does not offer this option. Fundamentally, there are two different ways how to approach this:

1. Aggregate -> select best ([see `compute_batch`][toop_engine_dc_solver.jax.compute_batch.compute_batch])
2. TODO: Select best -> aggregate ([see `compute_batch`][toop_engine_dc_solver.jax.compute_batch.compute_batch])

Depending on the number of injection candidates, either can be better. If you have very few injection candidates (<5 per branch topology, but >1 - in that case use [`symmetric mode`][toop_engine_dc_solver.jax.compute_batch.compute_symmetric_batch]), then the first option is likely better. 
If you have a lot of injection candidates, the second option is better. This is due to the fact that the aggregation is expensive, but deferring the aggregation as in 2. requires recomputing the N-1 matrix for the winning combination. 
The `metrics_first_solver` argument for [`run_solver`][toop_engine_dc_solver.jax.topology_looper.run_solver] lets you choose the better option.
It is recommended to try both to decide on the faster version for your use case.

## Buffer-size

In non-symmetric mode, it is not known how many injection candidates will be computed for each batch of branch topologies. As jax requires static shaping, a buffer needs to be allocated that is large enough for all injection candidates to fit. By default, you can just set this to None to let the run_solver method choose a good buffer size. This might however cause recompilations if you plan to call run_solver multiple times. In that case, it is advisable to manually set a buffer size such that the maximum number of injection candidates for any batch of branch topologies never exceeds ```buffer_size_injection * batch_size_injection```. Note that with large ```batch_size_bsdf```, this worst-case number can grow rather large. You can also get the theoretical upper bound through [`upper_bound_buffer_size_injections`][toop_engine_dc_solver.jax.upper_bound_buffer_size_injections]. Using a more aggresive lower bound can save memory, but might cause wrong results as the solver will fail silently. You can use [`get_buffer_utilization`][toop_engine_dc_solver.jax.get_buffer_utilization] to check if the buffer size is sufficient.

## Cross-coupler flows

By default, cross coupler flows are not computed because it adds a little bit of overhead to the computation. However, the solver supports computing the flow across the coupler before the coupler is opened. For this you need to set ```static_information.solver_config.cross_coupler_flow = True``` and ```static_information.dynamic_information.unsplit_flow = get_unsplit_flow(static_information.dynamic_information.ptdf, static_information.dynamic_information.nodal_injections)```. In this case, the cross-coupler input to the metrics and output functions will be an array of (n_splits, n_timesteps) with the flows, and internally the way the N-0 flows are computed changes. While the default variation computes ptdf @ nodal_injections in the loop with the final ptdf after all splits applied, the cross-coupler variation computes the flows by adjusting the unsplit flows for every split.

## Limited number of splits

The two most costly parts of the code are the BSDF computation and the LODF computation for disconnections. Both computations take a time proportional to the size of the inputs, e.g. the BSDF module will perform a BSDF computation for every substation regardless if only a few substations are actually split. Accordingly, the disconnections module will perform a LODF computation for every slot in the disconnections array even if all entries are masked with invalid branch indices. Hence, it is desireable to keep these arrays as small as possible. To reduce the number of substations in a topology, you can set a fixed number of splits through [`apply_limit_n_subs`][toop_engine_dc_solver.jax.apply_limit_n_subs].

## Scientific Background

A publication about the solver can be found [here](https://arxiv.org/abs/2501.17529).
The solver works with the <!-- markdown-link-check-disable -->[BSDF formulation](https://www.techrxiv.org/doi/full/10.36227/techrxiv.22298950.v1)<!-- markdown-link-check-enable --> to project bus splits and busbar assignments directly to the PTDF matrix. Furthermore, it uses the [MODF formulation](https://arxiv.org/abs/1606.07276) to do the same with branch outages. Hence, during the preprocessing, the [`PTDF`][toop_engine_dc_solver.jax.types.StaticInformation] matrix needs to be created. As the PTDF is not enough in the presence of phase shifters, the preprocessing scripts concatenates the [PSDF](https://doi.org/10.1109/TPAS.1985.319195) to the PTDF matrix to compute the loadflow results with one single matrix multiplication.
