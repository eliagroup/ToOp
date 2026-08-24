var e=e=>{switch(e){case`dataFlow`:return'---\ntitle: "Technology and data flow"\n---\ngraph LR\n  Client@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }\n  UnprocessedGridStore@{ shape: disk, label: "Unprocessed grid store" }\n  subgraph ProcessedGrid["`Processed grid folder`"]\n    ProcessedGrid.GridSnapshot@{ shape: doc, label: "grid.xiidm / grid.json" }\n    ProcessedGrid.StaticInfo@{ shape: doc, label: "static_information.hdf5" }\n    ProcessedGrid.ActionSet@{ shape: doc, label: "action_set.json + action_set_diffs.hdf5" }\n    ProcessedGrid.Snapshots@{ shape: doc, label: "optimizer_snapshots/ac" }\n  end\n  subgraph Kafka["`Kafka`"]\n    Kafka.ImporterCommands@{ shape: horizontal-cylinder, label: "importer_commands" }\n    Kafka.Commands@{ shape: horizontal-cylinder, label: "commands" }\n    Kafka.ImporterResults@{ shape: horizontal-cylinder, label: "importer_results" }\n    Kafka.ImporterHeartbeat@{ shape: horizontal-cylinder, label: "importer_heartbeat" }\n    Kafka.Results@{ shape: horizontal-cylinder, label: "results" }\n    Kafka.Heartbeat@{ shape: horizontal-cylinder, label: "heartbeat" }\n  end\n  subgraph Toop["`ToOp Engine`"]\n    Toop.Importer@{ shape: rectangle, label: "Importer" }\n    Toop.DcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }\n    Toop.AcValidator@{ shape: rectangle, label: "AC-Validator" }\n    Toop.LfService@{ shape: rectangle, label: "AC loadflow service" }\n  end\n  LoadflowStore@{ shape: disk, label: "Loadflow result store" }\n  Downstream@{ shape: rectangle, label: "Frontend / downstream systems" }\n  Client -. "`StartPreprocessingCommand`" .-> Kafka.ImporterCommands\n  Client -. "`StartOptimizationCommand`" .-> Kafka.Commands\n  UnprocessedGridStore -. "`raw grid file`" .-> Toop.Importer\n  Kafka.ImporterCommands -. "`consumes command`" .-> Toop.Importer\n  ProcessedGrid.GridSnapshot -. "`[...]`" .-> Toop.Importer\n  Toop.Importer -. "`PreprocessingSuccessResult`" .-> Kafka.ImporterResults\n  Toop.Importer -. "`PreprocessHeartbeat per stage`" .-> Kafka.ImporterHeartbeat\n  Toop.Importer -. "`[...]`" .-> ProcessedGrid.StaticInfo\n  Toop.Importer -. "`the same actions as physical switchings`" .-> ProcessedGrid.ActionSet\n  Toop.Importer -. "`normalized snapshot`" .-> ProcessedGrid.GridSnapshot\n  Toop.Importer -. "`initial AC N-1 results`" .-> LoadflowStore\n  Kafka.Commands -. "`consumes command`" .-> Toop.DcOptimizer\n  ProcessedGrid.StaticInfo -. "`loaded onto the GPU at startup`" .-> Toop.DcOptimizer\n  Toop.DcOptimizer -. "`TopologyPushResult per epoch`" .-> Kafka.Results\n  Toop.DcOptimizer -. "`OptimizationStatsHeartbeat`" .-> Kafka.Heartbeat\n  Kafka.Commands -. "`consumes the same command`" .-> Toop.AcValidator\n  Kafka.Results -. "`DC topologies`" .-> Toop.AcValidator\n  ProcessedGrid.ActionSet -. "`to realize topologies`" .-> Toop.AcValidator\n  ProcessedGrid.GridSnapshot -. "`base grid`" .-> Toop.AcValidator\n  LoadflowStore -. "`initial loadflow as baseline`" .-> Toop.AcValidator\n  Toop.AcValidator -. "`AC-validated Strategy`" .-> Kafka.Results\n  Toop.AcValidator -. "`OptimizationStatsHeartbeat`" .-> Kafka.Heartbeat\n  Toop.AcValidator -. "`summaries and diagrams`" .-> ProcessedGrid.Snapshots\n  Toop.AcValidator -. "`AC loadflow results per evaluated topology`" .-> LoadflowStore\n  Kafka.Results -. "`validated topologies for review`" .-> Downstream\n  ProcessedGrid.Snapshots -. "`UCTE, DGS, OpenRAO summaries, single line diagrams`" .-> Downstream\n';case`importerInternals`:return`---
title: "Importer -- what happens to a grid file"
---
graph TB
  subgraph ProcessedGrid["\`Processed grid folder\`"]
    subgraph ProcessedGrid.StaticInfo["\`static_information.hdf5\`"]
      ProcessedGrid.StaticInfo.BranchActionSet@{ shape: doc, label: "BranchActionSet" }
    end
    ProcessedGrid.GridSnapshot@{ shape: doc, label: "grid.xiidm / grid.json" }
    ProcessedGrid.AssetTopoMaster@{ shape: doc, label: "initial_topology/asset_topology_master_data.json" }
    ProcessedGrid.Masks@{ shape: doc, label: "masks/*.npy" }
    ProcessedGrid.ActionSet@{ shape: doc, label: "action_set.json + action_set_diffs.hdf5" }
    ProcessedGrid.Nminus1@{ shape: doc, label: "nminus1_definition.json" }
  end
  UnprocessedGridStore@{ shape: disk, label: "Unprocessed grid store" }
  subgraph ToopImporter["\`Importer\`"]
    subgraph ToopImporter.ImportStage["\`convert_file\`"]
      ToopImporter.ImportStage.LoadGrid@{ shape: rectangle, label: "Load and merge" }
      ToopImporter.ImportStage.Whitelists@{ shape: rectangle, label: "Apply whitelists" }
      ToopImporter.ImportStage.ConvergingParams@{ shape: rectangle, label: "find_converging_loadflow_params" }
      ToopImporter.ImportStage.NetworkMasks@{ shape: rectangle, label: "get_network_masks" }
      ToopImporter.ImportStage.TopologyModel@{ shape: rectangle, label: "get_master_asset_topology_artifact" }
    end
    subgraph ToopImporter.DcPreprocess["\`load_grid (DC preprocessing)\`"]
      ToopImporter.DcPreprocess.Materialize@{ shape: rectangle, label: "get_runtime_asset_topology" }
      ToopImporter.DcPreprocess.Bridges@{ shape: rectangle, label: "compute_bridging_branches" }
      ToopImporter.DcPreprocess.RelevantNodes@{ shape: rectangle, label: "filter_relevant_nodes" }
      ToopImporter.DcPreprocess.Factors@{ shape: rectangle, label: "compute PTDF / PSDF" }
      ToopImporter.DcPreprocess.Reduce@{ shape: rectangle, label: "reduce node / branch dimension" }
      ToopImporter.DcPreprocess.Nminus2Filter@{ shape: rectangle, label: "filter_disconnectable_branches_nminus2" }
      ToopImporter.DcPreprocess.Simplify@{ shape: rectangle, label: "simplify_asset_topology" }
      ToopImporter.DcPreprocess.ElectricalActions@{ shape: rectangle, label: "compute_electrical_actions" }
      ToopImporter.DcPreprocess.StationRealisations@{ shape: rectangle, label: "enumerate_station_realisations" }
      ToopImporter.DcPreprocess.BbOutage@{ shape: rectangle, label: "preprocess_bb_outage" }
    end
    ToopImporter.InitialLoadflow@{ shape: rectangle, label: "run_initial_loadflow" }
  end
  LoadflowStore@{ shape: disk, label: "Loadflow result store" }
  ToopImporter.InitialLoadflow -. "\`initial AC N-1 results\`" .-> LoadflowStore
  ToopImporter.ImportStage.LoadGrid -. "\`parsed network\`" .-> ToopImporter.ImportStage.Whitelists
  ToopImporter.ImportStage.Whitelists -. "\`scoped network\`" .-> ToopImporter.ImportStage.ConvergingParams
  ToopImporter.ImportStage.ConvergingParams -. "\`converging parameters\`" .-> ToopImporter.ImportStage.NetworkMasks
  ToopImporter.ImportStage.NetworkMasks -. "\`masks\`" .-> ToopImporter.ImportStage.TopologyModel
  ToopImporter.DcPreprocess.Materialize -. "\`runtime topology on NetworkData\`" .-> ToopImporter.DcPreprocess.Bridges
  ToopImporter.DcPreprocess.Bridges -. "\`bridge flags\`" .-> ToopImporter.DcPreprocess.RelevantNodes
  ToopImporter.DcPreprocess.RelevantNodes -. "\`switchable subset\`" .-> ToopImporter.DcPreprocess.Factors
  ToopImporter.DcPreprocess.Factors -. "\`PTDF / PSDF\`" .-> ToopImporter.DcPreprocess.Reduce
  ToopImporter.DcPreprocess.Reduce -. "\`reduced dimensions\`" .-> ToopImporter.DcPreprocess.Nminus2Filter
  ToopImporter.DcPreprocess.Nminus2Filter -. "\`final branch and injection ordering\`" .-> ToopImporter.DcPreprocess.Simplify
  ToopImporter.DcPreprocess.Nminus2Filter -. "\`disconnectable set D\`" .-> ToopImporter.DcPreprocess.ElectricalActions
  ToopImporter.DcPreprocess.Simplify -. "\`reduced stations to enumerate in\`" .-> ToopImporter.DcPreprocess.ElectricalActions
  ToopImporter.DcPreprocess.ElectricalActions -. "\`electrical splits\`" .-> ToopImporter.DcPreprocess.StationRealisations
  ToopImporter.DcPreprocess.StationRealisations -. "\`action set A\`" .-> ToopImporter.DcPreprocess.BbOutage
  ProcessedGrid.GridSnapshot -. "\`normalized network\`" .-> ToopImporter.ImportStage.TopologyModel
  ProcessedGrid.GridSnapshot -. "\`live switch, coupler and busbar state\`" .-> ToopImporter.DcPreprocess.Materialize
  ProcessedGrid.AssetTopoMaster -. "\`canonical structure\`" .-> ToopImporter.DcPreprocess.Materialize
  UnprocessedGridStore -. "\`raw grid file\`" .-> ToopImporter.ImportStage
  ToopImporter.ImportStage -. "\`ImportResult\`" .-> ToopImporter.DcPreprocess
  ToopImporter.DcPreprocess -. "\`ready grid folder\`" .-> ToopImporter.InitialLoadflow
  ToopImporter.DcPreprocess -. "\`padded action arrays for the GPU\`" .-> ProcessedGrid.StaticInfo.BranchActionSet
  ToopImporter.DcPreprocess -. "\`the same actions as physical switchings\`" .-> ProcessedGrid.ActionSet
  ToopImporter.ImportStage -. "\`per-asset masks\`" .-> ProcessedGrid.Masks
  ToopImporter.ImportStage -. "\`initial contingency set\`" .-> ProcessedGrid.Nminus1
  ToopImporter.DcPreprocess -. "\`refreshed contingency set\`" .-> ProcessedGrid.Nminus1
`;case`dcWorkerInternals`:return`---
title: "DC-Optimizer -- repertoire, variation, scoring"
---
graph TB
  subgraph ProcessedGridStaticInfo["\`static_information.hdf5\`"]
    ProcessedGridStaticInfo.BranchActionSet@{ shape: doc, label: "BranchActionSet" }
  end
  subgraph ToopDcOptimizer["\`DC-Optimizer\`"]
    ToopDcOptimizer.Scoring@{ shape: rectangle, label: "Scoring" }
    ToopDcOptimizer.Repertoire@{ shape: rectangle, label: "Discrete MAP-Elites repertoire" }
    ToopDcOptimizer.Mutation@{ shape: rectangle, label: "Mutation" }
    ToopDcOptimizer.Crossover@{ shape: rectangle, label: "Crossover" }
    ToopDcOptimizer.Pusher@{ shape: rectangle, label: "Epoch result push" }
    subgraph ToopDcOptimizer.DcSolver["\`GPU DC loadflow solver\`"]
      ToopDcOptimizer.DcSolver.BsdfStage@{ shape: rectangle, label: "compute_bsdf_lodf_static_flows" }
      ToopDcOptimizer.DcSolver.N0Stage@{ shape: rectangle, label: "N-0 flows" }
      ToopDcOptimizer.DcSolver.N1Stage@{ shape: rectangle, label: "Contingency analysis (N-1)" }
      ToopDcOptimizer.DcSolver.ResultExtraction@{ shape: rectangle, label: "Result aggregation and sparsification" }
    end
  end
  subgraph Kafka["\`Kafka\`"]
    Kafka.Results@{ shape: horizontal-cylinder, label: "results" }
  end
  ToopDcOptimizer.Repertoire -. "\`sampled elites\`" .-> ToopDcOptimizer.Mutation
  ToopDcOptimizer.Repertoire -. "\`sampled pairs\`" .-> ToopDcOptimizer.Crossover
  ToopDcOptimizer.Repertoire -. "\`new elites\`" .-> ToopDcOptimizer.Pusher
  ToopDcOptimizer.Scoring -. "\`fitness and descriptors, sorted insert\`" .-> ToopDcOptimizer.Repertoire
  ToopDcOptimizer.DcSolver.BsdfStage -. "\`updated PTDF, LODF, MODF, static flows\`" .-> ToopDcOptimizer.DcSolver.N0Stage
  ToopDcOptimizer.DcSolver.N0Stage -. "\`N-0 flows and nodal injections\`" .-> ToopDcOptimizer.DcSolver.N1Stage
  ToopDcOptimizer.DcSolver.N1Stage -. "\`N-1 matrix\`" .-> ToopDcOptimizer.DcSolver.ResultExtraction
  ToopDcOptimizer.Pusher -. "\`TopologyPushResult per epoch\`" .-> Kafka.Results
  ToopDcOptimizer.Mutation -. "\`candidate batch\`" .-> ToopDcOptimizer.DcSolver
  ToopDcOptimizer.Crossover -. "\`candidate batch\`" .-> ToopDcOptimizer.DcSolver
  ToopDcOptimizer.DcSolver -. "\`N-0 and N-1 flows\`" .-> ToopDcOptimizer.Scoring
  ProcessedGridStaticInfo.BranchActionSet -. "\`sampling space -- indices into these arrays\`" .-> ToopDcOptimizer
`;case`acValidatorInternals`:return`---
title: "AC-Validator -- selection before validation"
---
graph TB
  subgraph Kafka["\`Kafka\`"]
    Kafka.Results@{ shape: horizontal-cylinder, label: "results" }
  end
  subgraph ProcessedGrid["\`Processed grid folder\`"]
    ProcessedGrid.GridSnapshot@{ shape: doc, label: "grid.xiidm / grid.json" }
    ProcessedGrid.ActionSet@{ shape: doc, label: "action_set.json + action_set_diffs.hdf5" }
    ProcessedGrid.Snapshots@{ shape: doc, label: "optimizer_snapshots/ac" }
  end
  subgraph ToopAcValidator["\`AC-Validator\`"]
    ToopAcValidator.ResultListener@{ shape: rectangle, label: "Result listener" }
    subgraph ToopAcValidator.SelectStrategy["\`select_strategy\`"]
      ToopAcValidator.SelectStrategy.Discriminator@{ shape: rectangle, label: "Discriminator filter" }
      ToopAcValidator.SelectStrategy.Dominator@{ shape: rectangle, label: "Dominator filter" }
      ToopAcValidator.SelectStrategy.Median@{ shape: rectangle, label: "Median filter" }
    end
    ToopAcValidator.WorstK@{ shape: rectangle, label: "Worst-k epoch" }
    ToopAcValidator.RemainingCa@{ shape: rectangle, label: "Remaining contingencies" }
    ToopAcValidator.Acceptance@{ shape: rectangle, label: "Acceptance evaluation" }
    ToopAcValidator.SummaryWriter@{ shape: rectangle, label: "Summary writer" }
  end
  ToopContingency@{ shape: rectangle, label: "Contingency analysis" }
  ToopInterfaces@{ shape: rectangle, label: "Interfaces" }
  LoadflowStore@{ shape: disk, label: "Loadflow result store" }
  ToopAcValidator.WorstK -. "\`survivors\`" .-> ToopAcValidator.RemainingCa
  ToopAcValidator.WorstK -. "\`worst-k results\`" .-> ToopAcValidator.Acceptance
  ToopAcValidator.RemainingCa -. "\`full N-1 results\`" .-> ToopAcValidator.Acceptance
  ToopAcValidator.Acceptance -. "\`accepted topologies\`" .-> ToopAcValidator.SummaryWriter
  ToopAcValidator.WorstK -. "\`worst-k contingencies\`" .-> ToopContingency
  ToopAcValidator.RemainingCa -. "\`full N-1\`" .-> ToopContingency
  ToopAcValidator.SummaryWriter -. "\`accepted topology\`" .-> ToopInterfaces
  ToopAcValidator.SelectStrategy.Discriminator -. "\`survivors\`" .-> ToopAcValidator.SelectStrategy.Dominator
  ToopAcValidator.SelectStrategy.Dominator -. "\`survivors\`" .-> ToopAcValidator.SelectStrategy.Median
  Kafka.Results -. "\`DC topologies\`" .-> ToopAcValidator.ResultListener
  ToopAcValidator.SummaryWriter -. "\`summaries and diagrams\`" .-> ProcessedGrid.Snapshots
  ToopInterfaces -. "\`persisted per job\`" .-> LoadflowStore
  ToopAcValidator.ResultListener -. "\`candidate pool\`" .-> ToopAcValidator.SelectStrategy
  ToopAcValidator.SelectStrategy -. "\`selected batch\`" .-> ToopAcValidator.WorstK
  ProcessedGrid.GridSnapshot -. "\`base grid\`" .-> ToopAcValidator
  ProcessedGrid.ActionSet -. "\`to realize topologies\`" .-> ToopAcValidator
  ToopAcValidator -. "\`AC loadflow results per evaluated topology\`" .-> LoadflowStore
  LoadflowStore -. "\`initial loadflow as baseline\`" .-> ToopAcValidator
`;case`contingencyAnalysis`:return`---
title: "Contingency analysis -- what each backend can do"
---
graph TB
  subgraph Toop["\`ToOp Engine\`"]
    subgraph Toop.Contingency["\`Contingency analysis\`"]
      subgraph Toop.Contingency.PwCa["\`run_contingency_analysis_powsybl\`"]
        Toop.Contingency.PwCa.PwLimitCache@{ shape: rectangle, label: "Branch limit cache" }
      end
      Toop.Contingency.Dispatcher@{ shape: rectangle, label: "get_ac_loadflow_results" }
      subgraph Toop.Contingency.PpCa["\`run_contingency_analysis_pandapower\`"]
        Toop.Contingency.PpCa.PpOutageGrouping@{ shape: rectangle, label: "Outage grouping" }
        Toop.Contingency.PpCa.PpSlack@{ shape: rectangle, label: "Slack allocation" }
        Toop.Contingency.PpCa.PpSpps@{ shape: rectangle, label: "SpPS rule engine" }
        Toop.Contingency.PpCa.PpCascade@{ shape: rectangle, label: "Cascade simulation" }
      end
    end
    subgraph Toop.Interfaces["\`Interfaces\`"]
      subgraph Toop.Interfaces.LfResults["\`LoadflowResults\`"]
        Toop.Interfaces.LfResults.BranchRes@{ shape: doc, label: "branch_results" }
        Toop.Interfaces.LfResults.NodeRes@{ shape: doc, label: "node_results" }
        Toop.Interfaces.LfResults.RegRes@{ shape: doc, label: "regulating_element_results" }
        Toop.Interfaces.LfResults.VaDiffRes@{ shape: doc, label: "va_diff_results" }
        Toop.Interfaces.LfResults.ConvergedRes@{ shape: doc, label: "converged" }
        Toop.Interfaces.LfResults.SwitchRes@{ shape: doc, label: "switch_results" }
        Toop.Interfaces.LfResults.ConnectivityRes@{ shape: doc, label: "connectivity_result" }
        Toop.Interfaces.LfResults.SppsRes@{ shape: doc, label: "spps_results" }
        Toop.Interfaces.LfResults.CascadeRes@{ shape: doc, label: "cascade_results" }
      end
    end
    subgraph Toop.Importer["\`Importer\`"]
      Toop.Importer.InitialLoadflow@{ shape: rectangle, label: "run_initial_loadflow" }
    end
    subgraph Toop.AcValidator["\`AC-Validator\`"]
      Toop.AcValidator.WorstK@{ shape: rectangle, label: "Worst-k epoch" }
      Toop.AcValidator.RemainingCa@{ shape: rectangle, label: "Remaining contingencies" }
    end
  end
  LoadflowStore@{ shape: disk, label: "Loadflow result store" }
  Toop.Contingency.PpCa.PpOutageGrouping -. "\`grouped contingencies\`" .-> Toop.Contingency.PpCa.PpSlack
  Toop.Contingency.PpCa.PpSlack -. "\`solvable islands\`" .-> Toop.Contingency.PpCa.PpSpps
  Toop.Contingency.PpCa.PpSpps -. "\`post-scheme state\`" .-> Toop.Contingency.PpCa.PpCascade
  Toop.Importer.InitialLoadflow -. "\`base grid N-1\`" .-> Toop.Contingency.Dispatcher
  Toop.AcValidator.WorstK -. "\`worst-k contingencies\`" .-> Toop.Contingency.Dispatcher
  Toop.AcValidator.WorstK -. "\`survivors\`" .-> Toop.AcValidator.RemainingCa
  Toop.AcValidator.RemainingCa -. "\`full N-1\`" .-> Toop.Contingency.Dispatcher
  Toop.Importer.InitialLoadflow -. "\`initial AC N-1 results\`" .-> LoadflowStore
  Toop.Contingency.Dispatcher -. "\`if pandapowerNet\`" .-> Toop.Contingency.PpCa
  Toop.Contingency.Dispatcher -. "\`if PyPowSyBl Network\`" .-> Toop.Contingency.PwCa
  Toop.Contingency.PpCa -. "\`fills all nine tables\`" .-> Toop.Interfaces.LfResults
  Toop.Contingency.PwCa -. "\`fills the five common tables\`" .-> Toop.Interfaces.LfResults
  Toop.Interfaces.LfResults -. "\`persisted per job\`" .-> LoadflowStore
  Toop.AcValidator -. "\`AC loadflow results per evaluated topology\`" .-> LoadflowStore
  LoadflowStore -. "\`initial loadflow as baseline\`" .-> Toop.AcValidator
`;case`assetTopology`:return`---
title: "Asset topology -- master to runtime to simplified"
---
graph TB
  ProcessedGridGridSnapshot@{ shape: doc, label: "grid.xiidm / grid.json" }
  subgraph ToopImporterImportStageTopologyModel["\`get_master_asset_topology_artifact\`"]
    ToopImporterImportStageTopologyModel.BusBreakerExtract@{ shape: rectangle, label: "get_bus_breaker_master_asset_topology" }
    ToopImporterImportStageTopologyModel.NodeBreakerExtract@{ shape: rectangle, label: "get_node_breaker_master_asset_topology" }
    ToopImporterImportStageTopologyModel.PpExtract@{ shape: rectangle, label: "get_master_asset_topology_from_network" }
  end
  ToopInterfacesAssetTopoMaster@{ shape: doc, label: "1. MasterAssetTopology" }
  ProcessedGridAssetTopoMaster@{ shape: doc, label: "initial_topology/asset_topology_master_data.json" }
  subgraph ToopImporterDcPreprocessMaterialize["\`get_runtime_asset_topology\`"]
    ToopImporterDcPreprocessMaterialize.PwMaterialize@{ shape: rectangle, label: "materialize_runtime_bus_groups_from_network_state" }
    ToopImporterDcPreprocessMaterialize.CompactMaterialize@{ shape: rectangle, label: "materialize_runtime_bus_group_from_runtime_state" }
  end
  ToopInterfacesAssetTopoRuntime@{ shape: doc, label: "2. RuntimeAssetTopology" }
  subgraph ToopImporterDcPreprocessSimplify["\`simplify_asset_topology\`"]
    ToopImporterDcPreprocessSimplify.PrepareSeparation@{ shape: rectangle, label: "prepare_for_separation_set" }
    ToopImporterDcPreprocessSimplify.BbSimplify@{ shape: rectangle, label: "simplify_asset_topology_for_bb_outages" }
  end
  ToopInterfacesAssetTopoSimplified@{ shape: doc, label: "3. SimplifiedAssetTopology" }
  ToopImporterDcPreprocessElectricalActions@{ shape: rectangle, label: "compute_electrical_actions" }
  ToopImporterDcPreprocessStationRealisations@{ shape: rectangle, label: "enumerate_station_realisations" }
  ToopImporterDcPreprocessBbOutage@{ shape: rectangle, label: "preprocess_bb_outage" }
  ToopInterfacesStoredActionSet@{ shape: doc, label: "Stored action set" }
  ToopInterfacesAssetTopoMaster -. "\`serialized once per import\`" .-> ProcessedGridAssetTopoMaster
  ToopInterfacesAssetTopoSimplified -. "\`the geometry splits are enumerated in\`" .-> ToopImporterDcPreprocessElectricalActions
  ToopInterfacesAssetTopoSimplified -. "\`station to realize a split against\`" .-> ToopImporterDcPreprocessStationRealisations
  ToopImporterDcPreprocessElectricalActions -. "\`electrical splits\`" .-> ToopImporterDcPreprocessStationRealisations
  ToopInterfacesAssetTopoSimplified -. "\`reduced again with couplers closed\`" .-> ToopImporterDcPreprocessBbOutage
  ToopImporterDcPreprocessStationRealisations -. "\`action set A\`" .-> ToopImporterDcPreprocessBbOutage
  ToopInterfacesAssetTopoRuntime -. "\`starting_bus_groups -- to reach the real switches\`" .-> ToopInterfacesStoredActionSet
  ToopInterfacesAssetTopoSimplified -. "\`simplified_starting_bus_groups -- the ordering local_actions is indexed against\`" .-> ToopInterfacesStoredActionSet
  ToopImporterDcPreprocessStationRealisations -. "\`physical switchings per action\`" .-> ToopInterfacesStoredActionSet
  ProcessedGridGridSnapshot -. "\`normalized network\`" .-> ToopImporterImportStageTopologyModel
  ToopImporterImportStageTopologyModel -. "\`bus groups, bays, circuit groups, possible connectivity\`" .-> ToopInterfacesAssetTopoMaster
  ProcessedGridGridSnapshot -. "\`live switch, coupler and busbar state\`" .-> ToopImporterDcPreprocessMaterialize
  ProcessedGridAssetTopoMaster -. "\`canonical structure\`" .-> ToopImporterDcPreprocessMaterialize
  ToopImporterDcPreprocessMaterialize -. "\`structure + what is closed now\`" .-> ToopInterfacesAssetTopoRuntime
  ToopInterfacesAssetTopoRuntime -. "\`full physical bus groups\`" .-> ToopImporterDcPreprocessSimplify
  ToopImporterDcPreprocessSimplify -. "\`one reduced slice per electrical node\`" .-> ToopInterfacesAssetTopoSimplified
`;case`loadflowFormat`:return`---
title: "Loadflow result store -- the on-disk format"
---
graph TB
  subgraph LoadflowStore["\`Loadflow result store\`"]
    LoadflowStore.LfMetadata@{ shape: doc, label: "metadata.json" }
    LoadflowStore.LfBranch@{ shape: doc, label: "branch_results.parquet" }
    LoadflowStore.LfNode@{ shape: doc, label: "node_results.parquet" }
    LoadflowStore.LfConverged@{ shape: doc, label: "converged.parquet" }
    LoadflowStore.LfVaDiff@{ shape: doc, label: "va_diff_results.parquet" }
    LoadflowStore.LfReg@{ shape: doc, label: "regulating_element_results.parquet" }
    LoadflowStore.LfSwitch@{ shape: doc, label: "switch_results.parquet" }
    LoadflowStore.LfSpps@{ shape: doc, label: "spps_results.parquet" }
    LoadflowStore.LfCascade@{ shape: doc, label: "cascade_results.parquet" }
  end
  ToopInterfacesLfResults@{ shape: doc, label: "LoadflowResults" }
  ToopImporterInitialLoadflow@{ shape: rectangle, label: "run_initial_loadflow" }
  ToopAcValidator@{ shape: rectangle, label: "AC-Validator" }
  ToopInterfacesLfResults -. "\`persisted per job\`" .-> LoadflowStore
  ToopImporterInitialLoadflow -. "\`initial AC N-1 results\`" .-> LoadflowStore
  LoadflowStore -. "\`initial loadflow as baseline\`" .-> ToopAcValidator
  ToopAcValidator -. "\`AC loadflow results per evaluated topology\`" .-> LoadflowStore
`;case`index`:return'---\ntitle: "System landscape"\n---\ngraph TB\n  Client@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }\n  UnprocessedGridStore@{ shape: disk, label: "Unprocessed grid store" }\n  subgraph Toop["`ToOp Engine`"]\n    Toop.Interfaces@{ shape: rectangle, label: "Interfaces" }\n    Toop.Postprocess@{ shape: rectangle, label: "Postprocessing and export" }\n    Toop.LfService@{ shape: rectangle, label: "AC loadflow service" }\n    Toop.ImporterParams@{ shape: doc, label: "Importer parameters" }\n    Toop.DcParams@{ shape: doc, label: "DC optimizer parameters" }\n    Toop.AcParams@{ shape: doc, label: "AC validator parameters" }\n    Toop.Importer@{ shape: rectangle, label: "Importer" }\n    Toop.DcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }\n    Toop.AcValidator@{ shape: rectangle, label: "AC-Validator" }\n    Toop.Contingency@{ shape: rectangle, label: "Contingency analysis" }\n  end\n  Kafka@{ shape: horizontal-cylinder, label: "Kafka" }\n  LoadflowStore@{ shape: disk, label: "Loadflow result store" }\n  ProcessedGrid@{ shape: disk, label: "Processed grid folder" }\n  Downstream@{ shape: rectangle, label: "Frontend / downstream systems" }\n  Client -. "`[...]`" .-> Kafka\n  Kafka -. "`validated topologies for review`" .-> Downstream\n  ProcessedGrid -. "`UCTE, DGS, OpenRAO summaries, single line diagrams`" .-> Downstream\n  Client -. "`set in StartPreprocessingCommand`" .-> Toop.ImporterParams\n  Client -. "`set in StartOptimizationCommand`" .-> Toop.DcParams\n  Client -. "`set in the same StartOptimizationCommand`" .-> Toop.AcParams\n  Kafka -. "`consumes command`" .-> Toop.Importer\n  Kafka -. "`consumes command`" .-> Toop.DcOptimizer\n  Kafka -. "`[...]`" .-> Toop.AcValidator\n  UnprocessedGridStore -. "`raw grid file`" .-> Toop.Importer\n  ProcessedGrid -. "`[...]`" .-> Toop.Importer\n  ProcessedGrid -. "`[...]`" .-> Toop.DcOptimizer\n  ProcessedGrid -. "`[...]`" .-> Toop.AcValidator\n  LoadflowStore -. "`initial loadflow as baseline`" .-> Toop.AcValidator\n  Toop.Importer -. "`[...]`" .-> Kafka\n  Toop.Importer -. "`[...]`" .-> ProcessedGrid\n  Toop.Importer -. "`initial AC N-1 results`" .-> LoadflowStore\n  Toop.DcOptimizer -. "`[...]`" .-> Kafka\n  Toop.AcValidator -. "`[...]`" .-> Kafka\n  Toop.AcValidator -. "`summaries and diagrams`" .-> ProcessedGrid\n  Toop.AcValidator -. "`AC loadflow results per evaluated topology`" .-> LoadflowStore\n  Toop.Interfaces -. "`serialized once per import`" .-> ProcessedGrid\n  Toop.Interfaces -. "`persisted per job`" .-> LoadflowStore\n  Toop.Postprocess -. "`[...]`" .-> ProcessedGrid\n  Toop.Importer -. "`base grid N-1`" .-> Toop.Contingency\n  Toop.Importer -. "`[...]`" .-> Toop.Interfaces\n  Toop.Interfaces -. "`[...]`" .-> Toop.Importer\n  Toop.ImporterParams -. "`scope, limits, contingencies`" .-> Toop.Importer\n  Toop.DcParams -. "`search bounds, fitness, operator probabilities`" .-> Toop.DcOptimizer\n  Toop.AcValidator -. "`[...]`" .-> Toop.Contingency\n  Toop.AcValidator -. "`accepted topology`" .-> Toop.Interfaces\n  Toop.AcParams -. "`compute budget, pruning, rejection thresholds`" .-> Toop.AcValidator\n  Toop.Contingency -. "`[...]`" .-> Toop.Interfaces\n  Toop.Interfaces -. "`monitored elements and contingencies`" .-> Toop.Contingency\n  Toop.Postprocess -. "`grid with topology applied`" .-> Toop.Contingency\n  Toop.Interfaces -. "`[...]`" .-> Toop.Postprocess\n  Toop.Postprocess -. "`switch id and new state`" .-> Toop.Interfaces\n';case`overview`:return'---\ntitle: "ToOp at a glance"\n---\ngraph LR\n  Client@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }\n  KafkaImporterCommands@{ shape: horizontal-cylinder, label: "importer_commands" }\n  UnprocessedGridStore@{ shape: disk, label: "Unprocessed grid store" }\n  ToopImporter@{ shape: rectangle, label: "Importer" }\n  ProcessedGrid@{ shape: disk, label: "Processed grid folder" }\n  LoadflowStore@{ shape: disk, label: "Loadflow result store" }\n  KafkaImporterResults@{ shape: horizontal-cylinder, label: "importer_results" }\n  KafkaCommands@{ shape: horizontal-cylinder, label: "commands" }\n  ToopDcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }\n  ToopAcValidator@{ shape: rectangle, label: "AC-Validator" }\n  KafkaResults@{ shape: horizontal-cylinder, label: "results" }\n  Downstream@{ shape: rectangle, label: "Frontend / downstream systems" }\n  Client -. "`StartPreprocessingCommand`" .-> KafkaImporterCommands\n  KafkaImporterCommands -. "`picks up the job`" .-> ToopImporter\n  UnprocessedGridStore -. "`UCTE / CGMES / PowerFactory file`" .-> ToopImporter\n  ToopImporter -. "`normalized snapshot, masks, asset topology`" .-> ProcessedGrid\n  ToopImporter -. "`PTDF, action set, contingency set`" .-> ProcessedGrid\n  ToopImporter -. "`initial AC N-1 and reference metrics`" .-> LoadflowStore\n  ToopImporter -. "`PreprocessingSuccessResult`" .-> KafkaImporterResults\n  KafkaImporterResults -. "`data folder is ready`" .-> Client\n  Client -. "`StartOptimizationCommand`" .-> KafkaCommands\n  KafkaCommands -. "`starts the DC run`" .-> ToopDcOptimizer\n  KafkaCommands -. "`starts the AC run`" .-> ToopAcValidator\n  ToopDcOptimizer -. "`loads static information onto the GPU`" .-> ProcessedGrid\n  ToopAcValidator -. "`loads base grid and action set`" .-> ProcessedGrid\n  LoadflowStore -. "`reads the initial loadflow as baseline`" .-> ToopAcValidator\n  ToopDcOptimizer -. "`Strategy, once per epoch`" .-> KafkaResults\n  KafkaResults -. "`DC topologies to validate`" .-> ToopAcValidator\n  ToopAcValidator -. "`prune, worst-k, then full N-1`" .-> ToopAcValidator\n  ToopAcValidator -. "`AC loadflow results per evaluated topology`" .-> LoadflowStore\n  ToopAcValidator -. "`AC-validated Strategy, referencing its loadflow`" .-> KafkaResults\n  ToopAcValidator -. "`summaries, diagrams, loadflow tables`" .-> ProcessedGrid\n  KafkaResults -. "`topologies for review`" .-> Downstream\n  ProcessedGrid -. "`UCTE, DGS, OpenRAO summaries, single line diagrams`" .-> Downstream\n';case`parameters`:return`---
title: "Parameters -- where the constraints live"
---
graph TB
  Client@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }
  subgraph Toop["\`ToOp Engine\`"]
    subgraph Toop.ImporterParams["\`Importer parameters\`"]
      Toop.ImporterParams.PAreaSettings@{ shape: doc, label: "AreaSettings" }
      Toop.ImporterParams.PStationRules@{ shape: doc, label: "RelevantStationRules" }
      Toop.ImporterParams.PLists@{ shape: doc, label: "White / black / ignore lists" }
      Toop.ImporterParams.PContingencies@{ shape: doc, label: "Contingency list" }
      Toop.ImporterParams.PPreprocess@{ shape: doc, label: "PreprocessParameters" }
    end
    subgraph Toop.DcParams["\`DC optimizer parameters\`"]
      Toop.DcParams.PMe@{ shape: doc, label: "BatchedMEParameters (ga_config)" }
      Toop.DcParams.PSolver@{ shape: doc, label: "LoadflowSolverParameters" }
      Toop.DcParams.PDoubleLimits@{ shape: doc, label: "DoubleLimitsSetpoint" }
    end
    subgraph Toop.AcParams["\`AC validator parameters\`"]
      Toop.AcParams.PAcGa@{ shape: doc, label: "ACGAParameters (ga_config)" }
      Toop.AcParams.PRejection@{ shape: doc, label: "Rejection thresholds" }
      Toop.AcParams.PInitialLoadflow@{ shape: doc, label: "initial_loadflow reference" }
    end
    Toop.Importer@{ shape: rectangle, label: "Importer" }
    Toop.DcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }
    Toop.AcValidator@{ shape: rectangle, label: "AC-Validator" }
  end
  Client -. "\`set in StartPreprocessingCommand\`" .-> Toop.ImporterParams
  Toop.ImporterParams -. "\`scope, limits, contingencies\`" .-> Toop.Importer
  Client -. "\`set in StartOptimizationCommand\`" .-> Toop.DcParams
  Toop.DcParams -. "\`search bounds, fitness, operator probabilities\`" .-> Toop.DcOptimizer
  Client -. "\`set in the same StartOptimizationCommand\`" .-> Toop.AcParams
  Toop.AcParams -. "\`compute budget, pruning, rejection thresholds\`" .-> Toop.AcValidator
`;default:throw Error(`Unknown viewId: `+e)}};export{e as mmdSource};