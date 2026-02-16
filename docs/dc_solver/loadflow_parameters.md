# Loadflow Parameters
The loadflow parameters influence the convergence and the results of the loadflow engine.
Detailed information about the possible settings can be found here: [Powsybl Docs](https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/stable/loadflow/parameters.html) or [Openloadflow Docs](https://powsybl.readthedocs.io/_/downloads/powsybl-open-loadflow/en/latest/pdf/).

During the import, we try two set of parameters, that can be found here:
**['Default'][toop_engine_grid_helpers.powsybl.loadflow_parameters.POWSYBL_LOADFLOW_PARAM_PF]**
**['Backup'][toop_engine_grid_helpers.powsybl.loadflow_parameters.DISTRIBUTED_SLACK]**

Both use a distributed slack, since this leads to smaller mismatches between AC and DC Loadflow.

3 different voltage initializations are used, to find one set of parameters that is converging. The voltage initialization should have no effect on the results.

The converging set of parameters is than stored in the processed grid folder and used for further pre- and postprocessing. 

If no converging parameters are found, the import stops early, unless the fail_on_non_convergence parameter is set to False (mostly for debugging)
