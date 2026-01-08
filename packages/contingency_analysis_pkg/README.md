# Contingency Analysis

The Contingency Analysis package provides the 
implementation to the [`Load-flow-Service`][packages.interfaces_pkg.src.toop_engine_interfaces.messages.lf_service] interface. This enables a backend independent Contingency definition. Additional it offers either Polars or Pandas DataFrames as a return value.

## Usage:

- Run the N-1 analysis with the function [`get_ac_loadflow_results`][packages.contingency_analysis_pkg.src.toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service.get_ac_loadflow_results]; depending on the network you provide, it will execute the contingency definition in either `pandapower` or `pypowsybl`.
- Compute metrics (polars only) for the results using [`compute_metrics`][packages.contingency_analysis_pkg.src.toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics]
- Direct entrypoint for PyPowSyBl: [`run_contingency_analysis_powsybl`][packages.contingency_analysis_pkg.src.toop_engine_contingency_analysis.pypowsybl.contingency_analysis_powsybl.run_contingency_analysis_powsybl]
- Direct entrypoint for Pandapower: [`run_contingency_analysis_pandapower`][packages.contingency_analysis_pkg.src.toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower.run_contingency_analysis_pandapower]
- Contingency definition is created during the import process: [`convert_file`][packages.importer_pkg.src.toop_engine_importer.pypowsybl_import.preprocessing.convert_file]
- Manually: Create a converter from you Contingency list to the [`Nminus1Definition`][packages.interfaces_pkg.src.toop_engine_interfaces.nminus1_definition.Nminus1Definition]
