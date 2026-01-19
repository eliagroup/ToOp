# **Contingency Import and Matching**

## **Overview**
This project provides tools to import contingency data from PowerFactory files and match them with grid model elements. The focus is on CIM-based grid models, ensuring compatibility with PowerFactory's contingency analysis.

---

## **Modules**

### 1. power_factory_data_class.py
This module is the interface pandera class to define the expected import data

[`power_factory_data_class`][packages.importer_pkg.src.toop_engine_importer.contingency_from_power_factory.power_factory_data_class]

### 2. contingency_from_file.py
This module contains functions to:
- Import contingencies from a file.
- Match contingencies with grid model elements by index or name.

[`contingency_from_file`][packages.importer_pkg.src.toop_engine_importer.contingency_from_power_factory.contingency_from_file]

#### **Main Functions**

[`get_contingencies_from_file`][packages.importer_pkg.src.toop_engine_importer.contingency_from_power_factory.get_contingencies_from_file]

[`match_contingencies`][packages.importer_pkg.src.toop_engine_importer.contingency_from_power_factory.match_contingencies]

### 3. power_factory_data_class.py
This module defines schemas for validating contingency and grid model data using `pandera`.

[`power_factory_data_class`][packages.importer_pkg.src.toop_engine_importer.contingency_from_power_factory.power_factory_data_class]

---

## **Usage**

### **Importing Contingencies**
1. Use `get_contingencies_from_file` to read contingencies from a file.
2. Validate the data using the `ContingencyImportSchema`.

### **Matching Contingencies**
1. Use `match_contingencies` to match contingencies with grid model elements.
2. Choose to match by index or name using the `match_by_name` parameter.
3. Validate the data using the `ContingencyMatchSchema`.

### **Validation**
- All data is validated using `pandera` schemas (`ContingencyImportSchema`, `AllGridElementsSchema`, `ContingencyMatchSchema`).

## **Known Issues**
- UCTE-based grid models have not been tested
