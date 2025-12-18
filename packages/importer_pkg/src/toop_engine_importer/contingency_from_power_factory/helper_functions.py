# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for the contingency_from_power_factory module.

Author: Benjamin Petrick
Created: 2025-05-13
"""

import logbook
from toop_engine_importer.contingency_from_power_factory.power_factory_data_class import ContingencyMatchSchema

logger = logbook.Logger(__name__)


def get_stats_n1_list_found(processed_n1_definition: ContingencyMatchSchema) -> dict:
    """Get the statistics of the found elements in the n-1 definition.

    This function gets a statistics of the found elements in the n-1 definition.
    It counts the number of elements found and the number of elements not found.
    It also tries to categorize the not found elements into different categories.

    Parameters
    ----------
    processed_n1_definition : pd.DataFrame
        The n-1 definition to get the statistics from.
        The DataFrame must contain the following columns:
        - contingency_name: The name of the contingency from the n-1 definition file.
        - name: The name of the element in the network.
        - type: The type of the element in the network.
        This is a merge (how=left) of the n-1 definition loaded as ContingencyImportSchema and the get_all_element_names().

    Returns
    -------
    stats_dict : dict
        A dictionary containing the statistics of the found elements.
        keys:
            stats : dict
                A dictionary containing the statistics of the found elements.
                keys: unique element types from column "type" in processed_n1_definition
                values: number of elements found
                e.g. {"LINE": 10, "SWITCH": 5, "GENERATOR": 3}
            stats_not_found_name : dict
                A dictionary containing the names of the not found elements.
                keys: guessed categories of not found elements
                values: list of names of the not found elements
                e.g. {"switch": ["SW1", "SW2"], "special_PF_definition": ["PF1"]}
            stats_not_found_count : dict
                A dictionary containing the number of not found elements.
                keys: guessed categories of not found elements
                values: number of not found elements
                e.g. {"switch": 2, "special_PF_definition": 1}
            n_found : int
                The number of elements found in the n-1 definition.
            n_total : int
                The total number of elements (found+not_found) in the n-1 definition.
            n_complete_cases : int
                The number of complete cases in the n-1 definition.
                A complete case is a case where all elements are found.
            n_no_complete_cases : int
                The number of no complete cases in the n-1 definition.
                A not complete case is a case where no elements are found.
            mixed_cases : int
                The number of mixed cases in the n-1 definition.
                A mixed case is a case where some elements are found and some are not found.
    """
    # ###############################
    # Currently only used as information, do not use in production
    # ###############################
    logger.warning("get_stats_n1_list_found is not tested, do not use in production.")

    n_total = len(processed_n1_definition)
    # get the number of elements found
    stats = dict(processed_n1_definition["element_type"].value_counts())
    stats["NA"] = (processed_n1_definition["element_type"].isna()).sum()
    # check if all elements are accounted for
    assert len(processed_n1_definition) == sum([value for value in stats.values()]), (
        f"Not all elements are accounted for: {stats} != {len(processed_n1_definition)}"
    )

    # try to categorize the not found elements
    stats_not_found_name = {}
    not_found = processed_n1_definition[processed_n1_definition["element_type"].isna()]
    len_not_found = len(not_found)
    cond_switch = not_found["power_factory_grid_model_name"].str.endswith(" SW")
    stats_not_found_name["switch"] = list(not_found[cond_switch]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_switch]

    cond_special_power_factory_definition = (
        ~not_found["power_factory_grid_model_name"].str[1].str.isalpha()
        & ~not_found["power_factory_grid_model_name"].str[1].str.isnumeric()
    )
    stats_not_found_name["special_PF_definition"] = list(
        not_found[cond_special_power_factory_definition]["power_factory_grid_model_name"].values
    )
    not_found = not_found[~cond_special_power_factory_definition]

    cond_busbar = not_found["power_factory_grid_model_name"].str.endswith(" BS")
    stats_not_found_name["busbar"] = list(not_found[cond_busbar]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_busbar]

    cond_line = not_found["power_factory_grid_model_name"].str.startswith("DI")
    stats_not_found_name["lines"] = list(not_found[cond_line]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_line]

    cond_trafo = not_found["power_factory_grid_model_name"].str.endswith(" TFO")
    stats_not_found_name["trafo"] = list(not_found[cond_trafo]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_trafo]

    cond_ln = not_found["power_factory_grid_model_name"].str.startswith("LN-")
    stats_not_found_name["LN-"] = list(not_found[cond_ln]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_ln]

    ucte_length = 8
    cond_ucte = not_found["power_factory_grid_model_name"].str.len() == ucte_length
    stats_not_found_name["UCTE? (len 8)"] = list(not_found[cond_ucte]["power_factory_grid_model_name"].values)
    not_found = not_found[~cond_ucte]
    stats_not_found_name["other"] = list(not_found["power_factory_grid_model_name"].values)

    # count the categories
    stats_not_found_count = {}
    len_stats_not_found = 0
    for key, value in stats_not_found_name.items():
        stats_not_found_count[key] = len(value)
        len_stats_not_found += len(value)
    # check if all elements are accounted for
    assert len_not_found == len_stats_not_found, (
        f"Not all elements are accounted for: {len(not_found)} != {len_stats_not_found}"
    )

    # give a summary of found elements
    n_found = len(processed_n1_definition[~processed_n1_definition["element_type"].isna()])
    n_complete_cases = (
        processed_n1_definition.groupby("contingency_id").apply(lambda group: group["element_type"].notna().all()).sum()
    )

    n_mixed_cases = 0
    n_case_not_found = 0
    for contingency_id_not_found in processed_n1_definition[processed_n1_definition["grid_model_id"].isna()][
        "contingency_id"
    ].unique():
        if (
            contingency_id_not_found
            in processed_n1_definition[~processed_n1_definition["grid_model_id"].isna()]["contingency_id"].unique()
        ):
            n_mixed_cases += 1
        else:
            n_case_not_found += 1

    len_cases = len(processed_n1_definition["contingency_id"].unique())
    assert n_complete_cases + n_mixed_cases + n_case_not_found == len_cases, (
        "Not all elements are accounted for: "
        "n_complete_cases + mixed_cases + n_case_not_found == len_cases: "
        f"{n_complete_cases} + {n_mixed_cases} + {n_case_not_found} != {len_cases}"
    )

    stats_dict = {
        "n_found": n_found,
        "n_not_found": len_not_found,
        "n_total": n_total,
        "n_complete_cases": n_complete_cases,
        "n_case_not_found": n_case_not_found,
        "n_mixed_cases": n_mixed_cases,
        "n_total_cases": len_cases,
        "stats": stats,
        "stats_not_found_name": stats_not_found_name,
        "stats_not_found_count": stats_not_found_count,
    }

    return stats_dict
