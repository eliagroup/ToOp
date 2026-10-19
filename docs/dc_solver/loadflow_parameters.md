# Loadflow Parameters
The loadflow parameters influence the convergence and the results of the loadflow engine.
Detailed information about the possible settings can be found here: [Powsybl Docs](https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/stable/loadflow/parameters.html) or [Openloadflow Docs](https://powsybl.readthedocs.io/_/downloads/powsybl-open-loadflow/en/latest/pdf/).

During the import, a path to predefined parameters can be set on the import parameters. Otherwise we try two set of parameters, that can be found here:
**['Default'][toop_engine_grid_helpers.powsybl.loadflow_parameters.POWSYBL_LOADFLOW_PARAM_PF]**
**['Backup'][toop_engine_grid_helpers.powsybl.loadflow_parameters.CGMES_DISTRIBUTED_SLACK]**

Both use a distributed slack, since this leads to smaller mismatches between AC and DC Loadflow.

3 different voltage initializations are used, to find one set of parameters that is converging. The voltage initialization should have no effect on the results.

The default parameters switch off shunt compensator voltage control (`shunt_compensator_voltage_control_on=False`). OpenLoadFlow does not support regulating shunt compensators in contingencies, so the security analysis would fail on grids that have them.

The converging set of parameters is then stored in the processed grid folder and used for further pre- and postprocessing.

The importer checks that the base case loadflow converges twice: after loading the grid, and again after the preprocessing changes to the network. If a loadflow parameters file is given, only these parameters are used. If the base case does not converge at either check, the import raises a [`BaseCaseNotConvergedError`][toop_engine_importer.pypowsybl_import.preprocessing.BaseCaseNotConvergedError] before the loadflow parameters, masks, N-1 definition and importer statistics are written, and the [importer worker](../importer/worker/worker.md) sends an `ErrorResult`.

This also holds if `fail_on_non_convergence` is set to False (mostly for debugging). The flag only stops the parameter search from raising: it falls back to the backup parameters, and the import still stops if the base case does not converge with them.
