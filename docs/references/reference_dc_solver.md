# DC Solver

## DC Solver Example
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.example_classes
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.example_grids


## DC Solver Preprocess
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.pandapower.pandapower_backend
    handler: python  # Specify the handler for Python code
    options:
        heading_level: 2
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.powsybl.powsybl_backend
    handler: python  # Specify the handler for Python code
    options:
        heading_level: 2
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.load_grid
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.action_set
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.harmonize
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.network_data
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess_bb_outage
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess_station_realisations
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess_switching

## DC Solver Postprocess
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.abstract_runner
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.apply_asset_topo_pandapower
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.postprocess_pandapower
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.postprocess_powsybl
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.realize_assignment
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.validate_loadflow_results
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.postprocess.write_aux_data

## DC Solver Jax
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.aggregate_results
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.batching
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.branch_action_set
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.bsdf
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.busbar_outage
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.compute_batch
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.config
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.contingency_analysis
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.cross_coupler_flow
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.disconnections
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.injections
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.inputs
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.inspector
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.lodf
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.multi_outages
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.nminus2_outage
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.result_storage
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.topology_computations
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.topology_looper
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.unrolled_linalg
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.utils
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.types

## DC Solver Export
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.export.asset_topology_to_dgs
::: packages.dc_solver_pkg.src.toop_engine_dc_solver.export.dgs_v7_definitions
