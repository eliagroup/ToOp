# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A collection of Powsybl parameters used in the Loadflow Solver."""

import pypowsybl
from pypowsybl.loadflow import BalanceType, ComponentMode, Parameters, VoltageInitMode
from pypowsybl.security import IncreasedViolationsParameters, ModifiedMonitoredElementsParameters

# see https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/
OPENLOADFLOW_PARAM_PF = {
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

POWSYBL_LOADFLOW_PARAM_PF = Parameters(
    balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,  # BalanceType
    component_mode=ComponentMode.MAIN_CONNECTED,
    countries_to_balance=None,  # Sequence[str]
    dc_power_factor=1.0,
    dc_use_transformer_ratio=True,
    distributed_slack=True,
    phase_shifter_regulation_on=False,
    provider_parameters=OPENLOADFLOW_PARAM_PF,
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
    voltage_init_mode=VoltageInitMode.PREVIOUS_VALUES,  # VoltageInitMode
    write_slack_bus=True,
)

# for network single line diagram generation
SDL_PARAM = pypowsybl.network.SldParameters(
    use_name=True, component_library="Convergence", nodes_infos=True, display_current_feeder_info=True
)

# for network area diagram svg generation
NAD_PARAM = pypowsybl.network.NadParameters(edge_info_along_edge=True, substation_description_displayed=True)
CGMES_DISTRIBUTED_SLACK = Parameters(
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
UCTE_DISTRIBUTED_SLACK = Parameters(
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
SINGLE_SLACK = Parameters(distributed_slack=False, dc_use_transformer_ratio=True)
UNIFORM_SINGLE_SLACK = Parameters(
    balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P,
    component_mode=ComponentMode.ALL_CONNECTED,
    countries_to_balance=None,
    dc_power_factor=1.0,
    dc_use_transformer_ratio=False,
    distributed_slack=False,
    phase_shifter_regulation_on=False,
    provider_parameters=OPENLOADFLOW_PARAM_PF,
    read_slack_bus=True,
    shunt_compensator_voltage_control_on=False,
    transformer_voltage_control_on=False,
    use_reactive_limits=False,
    twt_split_shunt_admittance=False,
    voltage_init_mode=VoltageInitMode.UNIFORM_VALUES,
    write_slack_bus=False,
)

# Increased violation parameters: a post-contingency violation that already exists in the N-0 base case is only
# reported if it increased by more than the threshold, i.e. if post > pre * (1 + threshold).
# The flow threshold of -1 reports every post-contingency flow violation, also those that are equal to or lower than in
# N-0, so the N-1 metrics from the limit violation table match those of the monitored elements. Note that a negative
# value is outside of the intended range: powsybl documents the threshold as positive, but does not validate it.
VIOLATION_THRESHOLD = IncreasedViolationsParameters(
    flow_proportional_threshold=-1.0,
    low_voltage_absolute_threshold=1e9,
    low_voltage_proportional_threshold=0.01,
    high_voltage_absolute_threshold=1e9,
    high_voltage_proportional_threshold=0.01,
)

# Define a standard set for the modified monitored elements parameters, which can be used in the contingency analysis.
# reduces the monitored element results significantly, only element results are returned where there are changes
# compared to the N0 base case
# Not part of get_default_security_analysis_parameters: the monitored element runner needs the complete N-1 results,
# and the limit violation runner only monitors the base case, where the thresholds have no effect.
MODIFIED_MONITORED_ELEMENTS_PARAMETERS = ModifiedMonitoredElementsParameters(
    power_modification_threshold=1.0,
    voltage_modification_proportional_threshold=0.01,
    voltage_modification_absolute_threshold=1.0,
)


def get_default_security_analysis_parameters(
    load_flow_parameters: Parameters = CGMES_DISTRIBUTED_SLACK,
) -> pypowsybl.security.Parameters:
    """Get a new instance of the default security analysis parameters.

    A new instance is returned on every call, so callers can modify it, e.g. set provider parameters, without
    affecting other callers.

    Parameters
    ----------
    load_flow_parameters : Parameters, optional
        The loadflow parameters of the security analysis, by default CGMES_DISTRIBUTED_SLACK.

    Returns
    -------
    pypowsybl.security.Parameters
        The security analysis parameters with the increased violations thresholds. They suit both contingency
        runners: the contingency propagation is set by the runner, see run_contingency_analysis_powsybl and
        run_contingency_analysis_powsybl_violation_table.
    """
    return pypowsybl.security.Parameters(
        load_flow_parameters=load_flow_parameters,
        increased_violations_parameters=VIOLATION_THRESHOLD,
    )
