# Merging UCTE and CGMES Networks with `pypowsybl`

This document demonstrates how to use the `pypowsybl` module to merge UCTE and CGMES networks. The process involves:
- Identifying border lines inside and outside a specified area.
- Determining dangling voltage levels.
- Replacing dangling voltage levels with tie lines.
- Merging the networks.
- Performing an AC load flow on the merged network.

## Example

```python
from importer.pypowsybl_import.merge_ucte_cgmes import replace_line_with_tie
from importer.pypowsybl_import.merge_ucte_cgmes.replace_line_with_tie import get_dangling_voltage_levels
from importer.pypowsybl_import.merge_ucte_cgmes.merge_ucte_cgmes import run_merge_ucte_cgmes
from importer.pypowsybl_import import powsybl_masks
import pypowsybl

# Load the networks
net_ucte = pypowsybl.network.load(uct_file_path)
net_cgmes = pypowsybl.network.load(cgmes_zip)

# Retrieve line data to identify external border lines
lines_df = net_ucte.get_lines(attributes=["voltage_level1_id", "voltage_level2_id"])
area_codes = ["D8"]
cutoff_voltage = 220

# Identify parameters for HV line masks
hv_line_mask = (
    powsybl_masks.get_voltage_from_voltage_level_id(net_ucte, lines_df["voltage_level1_id"])
    >= cutoff_voltage
)
lines_with_limits = powsybl_masks.get_element_has_limits_mask(net_ucte, lines_df)

# Create masks for the voltage level columns corresponding to the area codes
region_columns = ["voltage_level1_id", "voltage_level2_id"]
side_1_in_n1_area = powsybl_masks.get_mask_for_area_codes(lines_df, area_codes, region_columns[0])
side_2_in_n1_area = powsybl_masks.get_mask_for_area_codes(lines_df, area_codes, region_columns[1])

# Identify border lines inside the observed area and outside of it
external_border_mask, _ = powsybl_masks.get_border_line_mask(
    lines_df,
    side_1_in_n1_area,
    side_2_in_n1_area,
    hv_line_mask,
    area_codes=area_codes,
)

# Get the voltage levels of the dangling lines and reduce them to unique values
dangling_voltage_levels = get_dangling_voltage_levels(
    network=net_ucte,
    external_border_mask=external_border_mask,
    area_codes=area_codes
)
dangling_voltage_levels = list(set(dangling_voltage_levels))

# Replace the dangling voltage levels with tie lines
for dangling_voltage_level in dangling_voltage_levels:
    replace_line_with_tie.replace_voltage_level_with_tie_line(
        network=net_ucte,
        voltage_level_id=dangling_voltage_level
    )

# Merge the networks and run AC load flow analysis
net_merged, report = run_merge_ucte_cgmes(
    net_cgmes=net_cgmes,
    net_ucte=net_ucte,
    ucte_area_name="D8"
)
pypowsybl.loadflow.run_ac(net_merged)
```

This example demonstrates the required steps to merge two networks and perform an AC load flow analysis on the merged network.
