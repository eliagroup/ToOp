# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A collection of Powsybl parameters used in the Loadflow Solver.

The parameter objects of pypowsybl are mutable. Every set is therefore built by a function that returns a new instance
on every call, so a caller can modify what it gets without changing the parameters of anyone else.
"""

from copy import deepcopy

import pypowsybl
from beartype.typing import Optional
from pypowsybl.loadflow import BalanceType, ComponentMode, Parameters, VoltageInitMode
from pypowsybl.security import IncreasedViolationsParameters, ModifiedMonitoredElementsParameters

#: The OpenLoadFlow provider parameter of the security analysis that propagates a contingency to the breakers isolating
#: it. The security analysis runners use it as they are given. Switch it off only for an N-1 definition whose
#: contingencies are already propagated, see propagate_busbar_contingencies of the contingency analysis package.
CONTINGENCY_PROPAGATION = "contingencyPropagation"


def get_openloadflow_provider_parameters() -> dict[str, str]:
    """Get a new instance of the OpenLoadFlow provider parameters of get_powsybl_loadflow_parameters.

    See https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/

    Returns
    -------
    dict[str, str]
        The OpenLoadFlow provider parameters.
    """
    return {
        "voltageInitModeOverride": "VOLTAGE_MAGNITUDE",
        "reactiveLimitsMaxPpvSwitch": "0",
        "svcVoltageMonitoring": "false",
        "maxActivePowerMismatch": "0.0001",
        "maxReactivePowerMismatch": "0.00001",
        "maxNewtonRaphsonIterations": "25",
        "newtonRaphsonStoppingCriteriaType": "PER_EQUATION_TYPE_CRITERIA",
        "generatorReactivePowerRemoteControl": "true",
        "plausibleActivePowerLimit": "300",
        "useActiveLimits": "false",  # unclear whether it is only used for slack
        "minPlausibleTargetVoltage": "0.893",
        "maxPlausibleTargetVoltage": "1.105",
        "referenceBusSelectionMode": "GENERATOR_REFERENCE_PRIORITY",
    }


def get_powsybl_loadflow_parameters() -> Parameters:
    """Get a new instance of the loadflow parameters the importer tries first to find a converging loadflow.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        Distributed slack proportional to the maximum active power, previous values as voltage initialization and the
        OpenLoadFlow provider parameters of get_openloadflow_provider_parameters.
    """
    return Parameters(
        balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
        component_mode=ComponentMode.MAIN_CONNECTED,
        countries_to_balance=None,
        dc_power_factor=1.0,
        dc_use_transformer_ratio=True,
        distributed_slack=True,
        phase_shifter_regulation_on=False,
        provider_parameters=get_openloadflow_provider_parameters(),
        read_slack_bus=True,
        # https://github.com/powsybl/powsybl-open-loadflow/blob/6ce13927fefb38a2cb2092d3cf2fe89151959ecc/src/main/java/com/powsybl/openloadflow/network/impl/PropagatedContingency.java#L220
        # shunt_compensator_voltage_control_on=False random errors, due to
        #     if (creationParameters.isShuntCompensatorVoltageControlOn() && shunt.isRegulating()) {
        #     throw new UnsupportedOperationException("Shunt compensator '" + shunt.getId()
        #             + "' with voltage control on: not supported yet");
        # }
        shunt_compensator_voltage_control_on=False,
        transformer_voltage_control_on=True,
        use_reactive_limits=True,
        twt_split_shunt_admittance=True,
        voltage_init_mode=VoltageInitMode.PREVIOUS_VALUES,
        write_slack_bus=True,
    )


def get_cgmes_distributed_slack_parameters() -> Parameters:
    """Get a new instance of the distributed slack loadflow parameters for CGMES grids.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        Distributed slack proportional to the active power, the reference bus chosen by generator priority.
    """
    return Parameters(
        distributed_slack=True,
        balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P,
        voltage_init_mode=VoltageInitMode.DC_VALUES,
        provider_parameters={
            "slackDistributionFailureBehavior": "LEAVE_ON_SLACK_BUS",
            "referenceBusSelectionMode": "GENERATOR_REFERENCE_PRIORITY",
        },
        dc_use_transformer_ratio=True,
        write_slack_bus=True,
        read_slack_bus=True,
    )


def get_ucte_distributed_slack_parameters() -> Parameters:
    """Get a new instance of the distributed slack loadflow parameters for UCTE grids.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        Distributed slack proportional to the active power, the first slack as reference bus.
    """
    return Parameters(
        distributed_slack=True,
        balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P,
        voltage_init_mode=VoltageInitMode.DC_VALUES,
        provider_parameters={
            "slackDistributionFailureBehavior": "LEAVE_ON_SLACK_BUS",
            "referenceBusSelectionMode": "FIRST_SLACK",
        },
        dc_use_transformer_ratio=True,
        write_slack_bus=True,
        read_slack_bus=True,
    )


def get_single_slack_parameters() -> Parameters:
    """Get a new instance of the single slack loadflow parameters.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        The loadflow parameters without a distributed slack.
    """
    return Parameters(distributed_slack=False, dc_use_transformer_ratio=True)


def get_uniform_single_slack_parameters() -> Parameters:
    """Get a new instance of the single slack loadflow parameters with uniform voltage initialization.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        The loadflow parameters without a distributed slack, without voltage control and with uniform values as
        voltage initialization.
    """
    return Parameters(
        balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P,
        component_mode=ComponentMode.ALL_CONNECTED,
        countries_to_balance=None,
        dc_power_factor=1.0,
        dc_use_transformer_ratio=False,
        distributed_slack=False,
        phase_shifter_regulation_on=False,
        provider_parameters=get_openloadflow_provider_parameters(),
        read_slack_bus=True,
        shunt_compensator_voltage_control_on=False,
        transformer_voltage_control_on=False,
        use_reactive_limits=False,
        twt_split_shunt_admittance=False,
        voltage_init_mode=VoltageInitMode.UNIFORM_VALUES,
        write_slack_bus=False,
    )


def get_increased_violations_parameters() -> IncreasedViolationsParameters:
    """Get a new instance of the increased violations parameters of the security analysis.

    A post-contingency violation that already exists in the N-0 base case is only reported if it increased by more than
    the threshold, i.e. if post > pre * (1 + threshold). The flow threshold of -1 reports every post-contingency flow
    violation, also those that are equal to or lower than in N-0, so the N-1 metrics from the limit violation table
    match those of the monitored elements. Note that a negative value is outside of the intended range: powsybl
    documents the threshold as positive, but does not validate it.

    Returns
    -------
    IncreasedViolationsParameters
        The increased violations parameters.
    """
    return IncreasedViolationsParameters(
        flow_proportional_threshold=-1.0,
        low_voltage_absolute_threshold=1e9,
        low_voltage_proportional_threshold=0.01,
        high_voltage_absolute_threshold=1e9,
        high_voltage_proportional_threshold=0.01,
    )


def get_modified_monitored_elements_parameters() -> ModifiedMonitoredElementsParameters:
    """Get a new instance of the modified monitored elements parameters of the security analysis.

    Only element results that changed compared to the N-0 base case are returned, which reduces the monitored element
    results significantly. Not part of get_default_security_analysis_parameters: the monitored element runner needs the
    complete N-1 results, and the limit violation runner only monitors the base case, where the thresholds have no
    effect.

    Returns
    -------
    ModifiedMonitoredElementsParameters
        The modified monitored elements parameters.
    """
    return ModifiedMonitoredElementsParameters(
        power_modification_threshold=1.0,
        voltage_modification_proportional_threshold=0.01,
        voltage_modification_absolute_threshold=1.0,
    )


def get_default_security_analysis_parameters(
    load_flow_parameters: Optional[Parameters] = None,
) -> pypowsybl.security.Parameters:
    """Get a new instance of the default security analysis parameters.

    A new instance is returned on every call, so callers can modify it, e.g. set provider parameters, without
    affecting other callers. The given loadflow parameters are copied, not shared.

    Parameters
    ----------
    load_flow_parameters : Optional[Parameters], optional
        The loadflow parameters of the security analysis, by default get_cgmes_distributed_slack_parameters.

    Returns
    -------
    pypowsybl.security.Parameters
        The security analysis parameters with the increased violations thresholds and the contingency propagation of
        powsybl switched on, see CONTINGENCY_PROPAGATION. They suit both contingency runners, which run the
        parameters as they are given.
    """
    return pypowsybl.security.Parameters(
        load_flow_parameters=(
            get_cgmes_distributed_slack_parameters() if load_flow_parameters is None else deepcopy(load_flow_parameters)
        ),
        increased_violations_parameters=get_increased_violations_parameters(),
        provider_parameters={CONTINGENCY_PROPAGATION: "true"},
    )
