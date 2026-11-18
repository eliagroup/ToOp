# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Balance a powsybl grid by writing the distributed slack of a loadflow into the injection targets."""

import pypowsybl
import structlog
from beartype.typing import Literal
from pypowsybl.loadflow import ComponentStatus
from pypowsybl.network import Network

logger = structlog.get_logger(__name__)


def balance_grid(net: Network, method: Literal["ac", "dc"], lf_params: pypowsybl.loadflow.Parameters) -> bool:
    """Balance the grid in place: distribute the slack with a loadflow and write it into the injection targets.

    The active power of generators and batteries after the loadflow becomes their target_p, in AC the reactive power
    their target_q as well. A loadflow on the balanced grid then gives the same flows with (almost) no slack mismatch,
    also without a distributed slack. This is what a security analysis needs that runs its contingencies without a
    distributed slack, like the DC one, but has to keep the N-0 flows of the distributed slack.

    AC and DC balance differently: an AC balance covers the losses as well, a DC balance only the imbalance of
    generation and load. Balance with the method of the analysis that follows, after every topology change.

    Parameters
    ----------
    net : Network
        The powsybl network to balance. It is modified in place, its loadflow results are overwritten.
    method : Literal["ac", "dc"]
        The loadflow that distributes the slack.
    lf_params : pypowsybl.loadflow.Parameters
        The loadflow parameters. Without a distributed slack there is nothing to distribute, the grid is left alone.

    Returns
    -------
    bool
        Whether the grid was balanced. False if the parameters have no distributed slack, or if the loadflow of the
        main component did not converge, in which case the targets are left unchanged and a warning is logged, so
        the analysis that follows reports the failure.
    """
    if not lf_params.distributed_slack:
        return False
    run = pypowsybl.loadflow.run_ac if method == "ac" else pypowsybl.loadflow.run_dc
    component_results = run(net, lf_params)
    main = [result for result in component_results if result.connected_component_num == 0]
    if not main or main[0].status != ComponentStatus.CONVERGED:
        logger.warning(
            "The loadflow did not converge, the grid is not balanced",
            method=method,
            status=main[0].status_text if main else "no main component",
        )
        return False

    # Injections without a result, like disconnected ones or those of an island without a slack, keep their targets
    target_columns = ["target_p", "target_q"] if method == "ac" else ["target_p"]
    generators = net.get_generators(attributes=["p", "q", *target_columns])
    generators["target_p"] = (-generators["p"]).fillna(generators["target_p"])
    batteries = net.get_batteries(attributes=["p", "q", *target_columns])
    batteries["target_p"] = (-batteries["p"]).fillna(batteries["target_p"])
    if method == "ac":
        generators["target_q"] = (-generators["q"]).fillna(generators["target_q"])
        batteries["target_q"] = (-batteries["q"]).fillna(batteries["target_q"])
    net.update_generators(generators[target_columns])
    net.update_batteries(batteries[target_columns])
    return True
