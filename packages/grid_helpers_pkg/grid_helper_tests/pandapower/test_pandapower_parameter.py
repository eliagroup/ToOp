import pandapower as pp
import pandapower.networks as pn
from toop_engine_grid_helpers.pandapower.loadflow_parameters import (
    PANDAPOWER_LOADFLOW_PARAM_PPCI_AC,
    PANDAPOWER_LOADFLOW_PARAM_PPCI_DC,
    PANDAPOWER_LOADFLOW_PARAM_RUNPP,
)


def test_pandapower_parameter():
    # Create a simple test network
    net = pn.simple_mv_open_ring_net()
    pp.runpp(net, **PANDAPOWER_LOADFLOW_PARAM_RUNPP)
    assert net["converged"], "Pandapower runpp did not converge successfully"

    ppc = net._ppc
    ppci = pp.pd2ppc._ppc2ppci(ppc, net)
    pp.pf.run_newton_raphson_pf._run_newton_raphson_pf(ppci, options=PANDAPOWER_LOADFLOW_PARAM_PPCI_DC)
    assert ppci["success"], "Pandapower DC load flow did not converge successfully"

    pp.pf.run_newton_raphson_pf._run_newton_raphson_pf(ppci, options=PANDAPOWER_LOADFLOW_PARAM_PPCI_AC)
    assert ppci["success"], "Pandapower load flow did not converge successfully"
