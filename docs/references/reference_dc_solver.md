# DC Solver

## DC Solver Example
::: toop_engine_dc_solver.example_classes
::: toop_engine_dc_solver.example_grids


## DC Solver Preprocess
::: toop_engine_dc_solver.preprocess.preprocess
::: toop_engine_dc_solver.preprocess.pandapower.pandapower_backend
    handler: python  # Specify the handler for Python code
    options:
        heading_level: 2
::: toop_engine_dc_solver.preprocess.powsybl.powsybl_backend
    handler: python  # Specify the handler for Python code
    options:
        heading_level: 2
::: toop_engine_dc_solver.preprocess.load_grid
::: toop_engine_dc_solver.preprocess.action_set
::: toop_engine_dc_solver.preprocess.convert_to_jax
::: toop_engine_dc_solver.preprocess.harmonize
::: toop_engine_dc_solver.preprocess.network_data
::: toop_engine_dc_solver.preprocess.preprocess_bb_outage
::: toop_engine_dc_solver.preprocess.preprocess_station_realisations
::: toop_engine_dc_solver.preprocess.preprocess_switching

## DC Solver Postprocess
::: toop_engine_dc_solver.postprocess
::: toop_engine_dc_solver.postprocess.abstract_runner
::: toop_engine_dc_solver.postprocess.apply_asset_topo_pandapower
::: toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl
::: toop_engine_dc_solver.postprocess.postprocess_pandapower
::: toop_engine_dc_solver.postprocess.postprocess_powsybl
::: toop_engine_dc_solver.postprocess.realize_assignment
::: toop_engine_dc_solver.postprocess.validate_loadflow_results
::: toop_engine_dc_solver.postprocess.write_aux_data

## DC Solver Jax
::: toop_engine_dc_solver.jax.aggregate_results
::: toop_engine_dc_solver.jax.batching
::: toop_engine_dc_solver.jax.branch_action_set
::: toop_engine_dc_solver.jax.bsdf
::: toop_engine_dc_solver.jax.busbar_outage
::: toop_engine_dc_solver.jax.compute_batch
::: toop_engine_dc_solver.jax.config
::: toop_engine_dc_solver.jax.contingency_analysis
::: toop_engine_dc_solver.jax.cross_coupler_flow
::: toop_engine_dc_solver.jax.disconnections
::: toop_engine_dc_solver.jax.injections
::: toop_engine_dc_solver.jax.inputs
::: toop_engine_dc_solver.jax.inspector
::: toop_engine_dc_solver.jax.lodf
::: toop_engine_dc_solver.jax.multi_outages
::: toop_engine_dc_solver.jax.nminus2_outage
::: toop_engine_dc_solver.jax.result_storage
::: toop_engine_dc_solver.jax.topology_computations
::: toop_engine_dc_solver.jax.topology_looper
::: toop_engine_dc_solver.jax.unrolled_linalg
::: toop_engine_dc_solver.jax.utils
::: toop_engine_dc_solver.jax.types

## DC Solver Export
::: toop_engine_dc_solver.export.asset_topology_to_dgs
::: toop_engine_dc_solver.export.dgs_v7_definitions
