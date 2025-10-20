# Exporter

Welcome to the importing repository: Exporting results

## Asset Topology to UCTE export

The UCTE export is e.g. RES/RAS or PRDx Process. The asset topology is translated into a UCTE format by applying switching changes directly to the original UCTE file. Hence the original UCTE file is needed for this export.

## Asset Topology to DGS export

The DGS export is used by PowerFactory. The asset topology is translated into a DGS format by creating a file that contains the switching actions needed to get into the topology. The Topology can be applied by importing the DGS into Power Factory:

1. Activate Project
2. Activate the Operational Scenario that was used to export the Grid
3. Go to File -> Import -> Import DGS
4. Save new operational Scenario
-> Toggle between before and after to evaluate the Topology
