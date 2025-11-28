"""A collection of Powsybl parameters used in the powsybl internal loadflow solver."""

import pypowsybl
from pypowsybl.loadflow import BalanceType, ConnectedComponentMode, Parameters, VoltageInitMode

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
}

POWSYBL_LOADFLOW_PARAM_PF = Parameters(
    balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,  # BalanceType
    connected_component_mode=ConnectedComponentMode.MAIN,
    countries_to_balance=None,  # Sequence[str]
    dc_power_factor=1.0,
    dc_use_transformer_ratio=True,
    distributed_slack=True,
    phase_shifter_regulation_on=False,
    provider_parameters=OPENLOADFLOW_PARAM_PF,
    read_slack_bus=True,
    shunt_compensator_voltage_control_on=False,
    transformer_voltage_control_on=False,
    use_reactive_limits=True,
    twt_split_shunt_admittance=False,
    voltage_init_mode=VoltageInitMode.PREVIOUS_VALUES,  # VoltageInitMode
    write_slack_bus=True,
)

# for network single line diagram generation
SDL_PARAM = pypowsybl.network.SldParameters(
    use_name=True, component_library="Convergence", nodes_infos=True, display_current_feeder_info=True
)

# for network area diagram svg generation
NAD_PARAM = pypowsybl.network.NadParameters(edge_info_along_edge=True, substation_description_displayed=True)
DISTRIBUTED_SLACK = pypowsybl.loadflow.Parameters(
    distributed_slack=True,
    balance_type=pypowsybl.loadflow.BalanceType.PROPORTIONAL_TO_GENERATION_P,
    voltage_init_mode=pypowsybl.loadflow.VoltageInitMode.DC_VALUES,
    provider_parameters={"slackDistributionFailureBehavior": "LEAVE_ON_SLACK_BUS"},
    dc_use_transformer_ratio=True,
)
SINGLE_SLACK = pypowsybl.loadflow.Parameters(distributed_slack=False, dc_use_transformer_ratio=True)
