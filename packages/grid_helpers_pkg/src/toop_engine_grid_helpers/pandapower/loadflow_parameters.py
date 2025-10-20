"""A Collection of Pandapower parameters used in the Loadflow Solver."""

PANDAPOWER_LOADFLOW_PARAM_RUNPP = {
    "distributed_slack": False,
    "voltage_depend_loads": False,
    "trafo_model": "t",
    "numba": True,
    "max_iteration": 20,
}

# TODO: these parameter need to be adjusted and should not be used until they are properly set up.

PANDAPOWER_LOADFLOW_PARAM_PPCI = {
    "calculate_voltage_angles": True,
    "trafo_model": "t",
    "check_connectivity": True,
    "mode": "pf",
    "switch_rx_ratio": 2,
    "enforce_q_lims": False,
    "voltage_depend_loads": False,
    "consider_line_temperature": False,
    "tdpf": False,
    "tdpf_update_r_theta": True,
    "tdpf_delay_s": None,
    "distributed_slack": False,
    "delta": 0,
    "trafo3w_losses": "hv",
    "init_results": True,
    "p_lim_default": 1000000000.0,
    "q_lim_default": 1000000000.0,
    "neglect_open_switch_branches": False,
    "tolerance_mva": 1e-08,
    "trafo_loading": "current",
    "numba": True,
    "algorithm": "nr",  # Normal Newton-Rhapson
    "max_iteration": 25,
    "v_debug": False,
    "only_v_results": False,
    "use_umfpack": True,
    "permc_spec": None,
    "lightsim2grid": False,
    "recycle": False,
}

PANDAPOWER_LOADFLOW_PARAM_PPCI_AC = {
    **PANDAPOWER_LOADFLOW_PARAM_PPCI,
    "ac": True,
    "init_vm_pu": "dc",
    "init_va_degree": "dc",
}

PANDAPOWER_LOADFLOW_PARAM_PPCI_DC = {
    **PANDAPOWER_LOADFLOW_PARAM_PPCI,
    "ac": False,
    "init_vm_pu": "none",  # key needed for pandapower not to crash
    "init_va_degree": "none",  # key needed for pandapower not to crash
}
