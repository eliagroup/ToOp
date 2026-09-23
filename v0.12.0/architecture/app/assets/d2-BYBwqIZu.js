var e=e=>{switch(e){case`dataFlow`:return`direction: right

Client: {
  label: "Operator / orchestration client"
  shape: c4-person
}
UnprocessedGridStore: {
  label: "Unprocessed grid store"
  shape: stored_data
}
ProcessedGrid: {
  label: "Processed grid folder"
  shape: stored_data

  GridSnapshot: {
    label: "grid.xiidm / grid.json"
    shape: document
  }
  StaticInfo: {
    label: "static_information.hdf5"
    shape: document
  }
  ActionSet: {
    label: "action_set.json + action_set_diffs.hdf5"
    shape: document
  }
  Snapshots: {
    label: "optimizer_snapshots/ac"
    shape: document
  }
}
Kafka: {
  label: "Kafka"
  shape: queue

  ImporterCommands: {
    label: "importer_commands"
    shape: queue
  }
  Commands: {
    label: "commands"
    shape: queue
  }
  ImporterResults: {
    label: "importer_results"
    shape: queue
  }
  ImporterHeartbeat: {
    label: "importer_heartbeat"
    shape: queue
  }
  Results: {
    label: "results"
    shape: queue
  }
  Heartbeat: {
    label: "heartbeat"
    shape: queue
  }
}
Toop: {
  label: "ToOp Engine"

  Importer: {
    label: "Importer"
  }
  DcOptimizer: {
    label: "DC-Optimizer"
  }
  AcValidator: {
    label: "AC-Validator"
  }
  LfService: {
    label: "AC loadflow service"
  }
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}
Downstream: {
  label: "Frontend / downstream systems"
}

Client -> Kafka.ImporterCommands: "StartPreprocessingCommand"
Client -> Kafka.Commands: "StartOptimizationCommand"
UnprocessedGridStore -> Toop.Importer: "raw grid file"
Kafka.ImporterCommands -> Toop.Importer: "consumes command"
ProcessedGrid.GridSnapshot -> Toop.Importer: "[...]"
Toop.Importer -> Kafka.ImporterResults: "PreprocessingSuccessResult"
Toop.Importer -> Kafka.ImporterHeartbeat: "PreprocessHeartbeat per stage"
Toop.Importer -> ProcessedGrid.StaticInfo: "[...]"
Toop.Importer -> ProcessedGrid.ActionSet: "the same actions as physical switchings"
Toop.Importer -> ProcessedGrid.GridSnapshot: "normalized snapshot"
Toop.Importer -> LoadflowStore: "initial AC N-1 results"
Kafka.Commands -> Toop.DcOptimizer: "consumes command"
ProcessedGrid.StaticInfo -> Toop.DcOptimizer: "loaded onto the GPU at startup"
Toop.DcOptimizer -> Kafka.Results: "TopologyPushResult per epoch"
Toop.DcOptimizer -> Kafka.Heartbeat: "OptimizationStatsHeartbeat"
Kafka.Commands -> Toop.AcValidator: "consumes the same command"
Kafka.Results -> Toop.AcValidator: "DC topologies"
ProcessedGrid.ActionSet -> Toop.AcValidator: "to realize topologies"
ProcessedGrid.GridSnapshot -> Toop.AcValidator: "base grid"
LoadflowStore -> Toop.AcValidator: "initial loadflow as baseline"
Toop.AcValidator -> Kafka.Results: "AC-validated Strategy"
Toop.AcValidator -> Kafka.Heartbeat: "OptimizationStatsHeartbeat"
Toop.AcValidator -> ProcessedGrid.Snapshots: "summaries and diagrams"
Toop.AcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
Kafka.Results -> Downstream: "validated topologies for review"
ProcessedGrid.Snapshots -> Downstream: "UCTE, DGS, OpenRAO summaries, single line diagrams"
`;case`importerInternals`:return`direction: down

ProcessedGrid: {
  label: "Processed grid folder"
  shape: stored_data

  StaticInfo: {
    label: "static_information.hdf5"
    shape: document

    BranchActionSet: {
      label: "BranchActionSet"
      shape: document
    }
  }
  GridSnapshot: {
    label: "grid.xiidm / grid.json"
    shape: document
  }
  AssetTopoMaster: {
    label: "initial_topology/asset_topology_master_data.json"
    shape: document
  }
  Masks: {
    label: "masks/*.npy"
    shape: document
  }
  ActionSet: {
    label: "action_set.json + action_set_diffs.hdf5"
    shape: document
  }
  Nminus1: {
    label: "nminus1_definition.json"
    shape: document
  }
}
UnprocessedGridStore: {
  label: "Unprocessed grid store"
  shape: stored_data
}
ToopImporter: {
  label: "Importer"

  ImportStage: {
    label: "convert_file"

    LoadGrid: {
      label: "Load and merge"
    }
    Whitelists: {
      label: "Apply whitelists"
    }
    ConvergingParams: {
      label: "find_converging_loadflow_params"
    }
    NetworkMasks: {
      label: "get_network_masks"
    }
    TopologyModel: {
      label: "get_master_asset_topology_artifact"
    }
  }
  DcPreprocess: {
    label: "load_grid (DC preprocessing)"

    Materialize: {
      label: "get_runtime_asset_topology"
    }
    Bridges: {
      label: "compute_bridging_branches"
    }
    RelevantNodes: {
      label: "filter_relevant_nodes"
    }
    Factors: {
      label: "compute PTDF / PSDF"
    }
    Reduce: {
      label: "reduce node / branch dimension"
    }
    Nminus2Filter: {
      label: "filter_disconnectable_branches_nminus2"
    }
    Simplify: {
      label: "simplify_asset_topology"
    }
    ElectricalActions: {
      label: "compute_electrical_actions"
    }
    StationRealisations: {
      label: "enumerate_station_realisations"
    }
    BbOutage: {
      label: "preprocess_bb_outage"
    }
  }
  InitialLoadflow: {
    label: "run_initial_loadflow"
  }
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}

ToopImporter.InitialLoadflow -> LoadflowStore: "initial AC N-1 results"
ToopImporter.ImportStage.LoadGrid -> ToopImporter.ImportStage.Whitelists: "parsed network"
ToopImporter.ImportStage.Whitelists -> ToopImporter.ImportStage.ConvergingParams: "scoped network"
ToopImporter.ImportStage.ConvergingParams -> ToopImporter.ImportStage.NetworkMasks: "converging parameters"
ToopImporter.ImportStage.NetworkMasks -> ToopImporter.ImportStage.TopologyModel: "masks"
ToopImporter.DcPreprocess.Materialize -> ToopImporter.DcPreprocess.Bridges: "runtime topology on NetworkData"
ToopImporter.DcPreprocess.Bridges -> ToopImporter.DcPreprocess.RelevantNodes: "bridge flags"
ToopImporter.DcPreprocess.RelevantNodes -> ToopImporter.DcPreprocess.Factors: "switchable subset"
ToopImporter.DcPreprocess.Factors -> ToopImporter.DcPreprocess.Reduce: "PTDF / PSDF"
ToopImporter.DcPreprocess.Reduce -> ToopImporter.DcPreprocess.Nminus2Filter: "reduced dimensions"
ToopImporter.DcPreprocess.Nminus2Filter -> ToopImporter.DcPreprocess.Simplify: "final branch and injection ordering"
ToopImporter.DcPreprocess.Nminus2Filter -> ToopImporter.DcPreprocess.ElectricalActions: "disconnectable set D"
ToopImporter.DcPreprocess.Simplify -> ToopImporter.DcPreprocess.ElectricalActions: "reduced stations to enumerate in"
ToopImporter.DcPreprocess.ElectricalActions -> ToopImporter.DcPreprocess.StationRealisations: "electrical splits"
ToopImporter.DcPreprocess.StationRealisations -> ToopImporter.DcPreprocess.BbOutage: "action set A"
ProcessedGrid.GridSnapshot -> ToopImporter.ImportStage.TopologyModel: "normalized network"
ProcessedGrid.GridSnapshot -> ToopImporter.DcPreprocess.Materialize: "live switch, coupler and busbar state"
ProcessedGrid.AssetTopoMaster -> ToopImporter.DcPreprocess.Materialize: "canonical structure"
UnprocessedGridStore -> ToopImporter.ImportStage: "raw grid file"
ToopImporter.ImportStage -> ToopImporter.DcPreprocess: "ImportResult"
ToopImporter.DcPreprocess -> ToopImporter.InitialLoadflow: "ready grid folder"
ToopImporter.DcPreprocess -> ProcessedGrid.StaticInfo.BranchActionSet: "padded action arrays for the GPU"
ToopImporter.DcPreprocess -> ProcessedGrid.ActionSet: "the same actions as physical switchings"
ToopImporter.ImportStage -> ProcessedGrid.Masks: "per-asset masks"
ToopImporter.ImportStage -> ProcessedGrid.Nminus1: "initial contingency set"
ToopImporter.DcPreprocess -> ProcessedGrid.Nminus1: "refreshed contingency set"
`;case`dcWorkerInternals`:return`direction: down

ProcessedGridStaticInfo: {
  label: "static_information.hdf5"
  shape: document

  BranchActionSet: {
    label: "BranchActionSet"
    shape: document
  }
}
ToopDcOptimizer: {
  label: "DC-Optimizer"

  Scoring: {
    label: "Scoring"
  }
  Repertoire: {
    label: "Discrete MAP-Elites repertoire"
  }
  Mutation: {
    label: "Mutation"
  }
  Crossover: {
    label: "Crossover"
  }
  Pusher: {
    label: "Epoch result push"
  }
  DcSolver: {
    label: "GPU DC loadflow solver"

    BsdfStage: {
      label: "compute_bsdf_lodf_static_flows"
    }
    N0Stage: {
      label: "N-0 flows"
    }
    N1Stage: {
      label: "Contingency analysis (N-1)"
    }
    ResultExtraction: {
      label: "Result aggregation and sparsification"
    }
  }
}
Kafka: {
  label: "Kafka"
  shape: queue

  Results: {
    label: "results"
    shape: queue
  }
}

ToopDcOptimizer.Repertoire -> ToopDcOptimizer.Mutation: "sampled elites"
ToopDcOptimizer.Repertoire -> ToopDcOptimizer.Crossover: "sampled pairs"
ToopDcOptimizer.Repertoire -> ToopDcOptimizer.Pusher: "new elites"
ToopDcOptimizer.Scoring -> ToopDcOptimizer.Repertoire: "fitness and descriptors, sorted insert"
ToopDcOptimizer.DcSolver.BsdfStage -> ToopDcOptimizer.DcSolver.N0Stage: "updated PTDF, LODF, MODF, static flows"
ToopDcOptimizer.DcSolver.N0Stage -> ToopDcOptimizer.DcSolver.N1Stage: "N-0 flows and nodal injections"
ToopDcOptimizer.DcSolver.N1Stage -> ToopDcOptimizer.DcSolver.ResultExtraction: "N-1 matrix"
ToopDcOptimizer.Pusher -> Kafka.Results: "TopologyPushResult per epoch"
ToopDcOptimizer.Mutation -> ToopDcOptimizer.DcSolver: "candidate batch"
ToopDcOptimizer.Crossover -> ToopDcOptimizer.DcSolver: "candidate batch"
ToopDcOptimizer.DcSolver -> ToopDcOptimizer.Scoring: "N-0 and N-1 flows"
ProcessedGridStaticInfo.BranchActionSet -> ToopDcOptimizer: "sampling space -- indices into these arrays"
`;case`acValidatorInternals`:return`direction: down

Kafka: {
  label: "Kafka"
  shape: queue

  Results: {
    label: "results"
    shape: queue
  }
}
ProcessedGrid: {
  label: "Processed grid folder"
  shape: stored_data

  GridSnapshot: {
    label: "grid.xiidm / grid.json"
    shape: document
  }
  ActionSet: {
    label: "action_set.json + action_set_diffs.hdf5"
    shape: document
  }
  Snapshots: {
    label: "optimizer_snapshots/ac"
    shape: document
  }
}
ToopAcValidator: {
  label: "AC-Validator"

  ResultListener: {
    label: "Result listener"
  }
  SelectStrategy: {
    label: "select_strategy"

    Discriminator: {
      label: "Discriminator filter"
    }
    Dominator: {
      label: "Dominator filter"
    }
    Median: {
      label: "Median filter"
    }
  }
  WorstK: {
    label: "Worst-k epoch"
  }
  RemainingCa: {
    label: "Remaining contingencies"
  }
  Acceptance: {
    label: "Acceptance evaluation"
  }
  SummaryWriter: {
    label: "Summary writer"
  }
}
ToopContingency: {
  label: "Contingency analysis"
}
ToopInterfaces: {
  label: "Interfaces"
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}

ToopAcValidator.WorstK -> ToopAcValidator.RemainingCa: "survivors"
ToopAcValidator.WorstK -> ToopAcValidator.Acceptance: "worst-k results"
ToopAcValidator.RemainingCa -> ToopAcValidator.Acceptance: "full N-1 results"
ToopAcValidator.Acceptance -> ToopAcValidator.SummaryWriter: "accepted topologies"
ToopAcValidator.WorstK -> ToopContingency: "worst-k contingencies"
ToopAcValidator.RemainingCa -> ToopContingency: "full N-1"
ToopAcValidator.SummaryWriter -> ToopInterfaces: "accepted topology"
ToopAcValidator.SelectStrategy.Discriminator -> ToopAcValidator.SelectStrategy.Dominator: "survivors"
ToopAcValidator.SelectStrategy.Dominator -> ToopAcValidator.SelectStrategy.Median: "survivors"
Kafka.Results -> ToopAcValidator.ResultListener: "DC topologies"
ToopAcValidator.SummaryWriter -> ProcessedGrid.Snapshots: "summaries and diagrams"
ToopInterfaces -> LoadflowStore: "persisted per job"
ToopAcValidator.ResultListener -> ToopAcValidator.SelectStrategy: "candidate pool"
ToopAcValidator.SelectStrategy -> ToopAcValidator.WorstK: "selected batch"
ProcessedGrid.GridSnapshot -> ToopAcValidator: "base grid"
ProcessedGrid.ActionSet -> ToopAcValidator: "to realize topologies"
ToopAcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
LoadflowStore -> ToopAcValidator: "initial loadflow as baseline"
`;case`contingencyAnalysis`:return`direction: down

Toop: {
  label: "ToOp Engine"

  Contingency: {
    label: "Contingency analysis"

    PwCa: {
      label: "run_contingency_analysis_powsybl"

      PwLimitCache: {
        label: "Branch limit cache"
      }
    }
    Dispatcher: {
      label: "get_ac_loadflow_results"
    }
    PpCa: {
      label: "run_contingency_analysis_pandapower"

      PpOutageGrouping: {
        label: "Outage grouping"
      }
      PpSlack: {
        label: "Slack allocation"
      }
      PpSpps: {
        label: "SpPS rule engine"
      }
      PpCascade: {
        label: "Cascade simulation"
      }
    }
  }
  Interfaces: {
    label: "Interfaces"

    LfResults: {
      label: "LoadflowResults"
      shape: document

      BranchRes: {
        label: "branch_results"
        shape: document
      }
      NodeRes: {
        label: "node_results"
        shape: document
      }
      RegRes: {
        label: "regulating_element_results"
        shape: document
      }
      VaDiffRes: {
        label: "va_diff_results"
        shape: document
      }
      ConvergedRes: {
        label: "converged"
        shape: document
      }
      SwitchRes: {
        label: "switch_results"
        shape: document
      }
      ConnectivityRes: {
        label: "connectivity_result"
        shape: document
      }
      SppsRes: {
        label: "spps_results"
        shape: document
      }
      CascadeRes: {
        label: "cascade_results"
        shape: document
      }
    }
  }
  Importer: {
    label: "Importer"

    InitialLoadflow: {
      label: "run_initial_loadflow"
    }
  }
  AcValidator: {
    label: "AC-Validator"

    WorstK: {
      label: "Worst-k epoch"
    }
    RemainingCa: {
      label: "Remaining contingencies"
    }
  }
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}

Toop.Contingency.PpCa.PpOutageGrouping -> Toop.Contingency.PpCa.PpSlack: "grouped contingencies"
Toop.Contingency.PpCa.PpSlack -> Toop.Contingency.PpCa.PpSpps: "solvable islands"
Toop.Contingency.PpCa.PpSpps -> Toop.Contingency.PpCa.PpCascade: "post-scheme state"
Toop.Importer.InitialLoadflow -> Toop.Contingency.Dispatcher: "base grid N-1"
Toop.AcValidator.WorstK -> Toop.Contingency.Dispatcher: "worst-k contingencies"
Toop.AcValidator.WorstK -> Toop.AcValidator.RemainingCa: "survivors"
Toop.AcValidator.RemainingCa -> Toop.Contingency.Dispatcher: "full N-1"
Toop.Importer.InitialLoadflow -> LoadflowStore: "initial AC N-1 results"
Toop.Contingency.Dispatcher -> Toop.Contingency.PpCa: "if pandapowerNet"
Toop.Contingency.Dispatcher -> Toop.Contingency.PwCa: "if PyPowSyBl Network"
Toop.Contingency.PpCa -> Toop.Interfaces.LfResults: "fills all nine tables"
Toop.Contingency.PwCa -> Toop.Interfaces.LfResults: "fills the five common tables"
Toop.Interfaces.LfResults -> LoadflowStore: "persisted per job"
Toop.AcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
LoadflowStore -> Toop.AcValidator: "initial loadflow as baseline"
`;case`assetTopology`:return`direction: down

ProcessedGridGridSnapshot: {
  label: "grid.xiidm / grid.json"
  shape: document
}
ToopImporterImportStageTopologyModel: {
  label: "get_master_asset_topology_artifact"

  BusBreakerExtract: {
    label: "get_bus_breaker_master_asset_topology"
  }
  NodeBreakerExtract: {
    label: "get_node_breaker_master_asset_topology"
  }
  PpExtract: {
    label: "get_master_asset_topology_from_network"
  }
}
ToopInterfacesAssetTopoMaster: {
  label: "1. MasterAssetTopology"
  shape: document
}
ProcessedGridAssetTopoMaster: {
  label: "initial_topology/asset_topology_master_data.json"
  shape: document
}
ToopImporterDcPreprocessMaterialize: {
  label: "get_runtime_asset_topology"

  PwMaterialize: {
    label: "materialize_runtime_bus_groups_from_network_state"
  }
  CompactMaterialize: {
    label: "materialize_runtime_bus_group_from_runtime_state"
  }
}
ToopInterfacesAssetTopoRuntime: {
  label: "2. RuntimeAssetTopology"
  shape: document
}
ToopImporterDcPreprocessSimplify: {
  label: "simplify_asset_topology"

  PrepareSeparation: {
    label: "prepare_for_separation_set"
  }
  BbSimplify: {
    label: "simplify_asset_topology_for_bb_outages"
  }
}
ToopInterfacesAssetTopoSimplified: {
  label: "3. SimplifiedAssetTopology"
  shape: document
}
ToopImporterDcPreprocessElectricalActions: {
  label: "compute_electrical_actions"
}
ToopImporterDcPreprocessStationRealisations: {
  label: "enumerate_station_realisations"
}
ToopImporterDcPreprocessBbOutage: {
  label: "preprocess_bb_outage"
}
ToopInterfacesStoredActionSet: {
  label: "Stored action set"
  shape: document
}

ToopInterfacesAssetTopoMaster -> ProcessedGridAssetTopoMaster: "serialized once per import"
ToopInterfacesAssetTopoSimplified -> ToopImporterDcPreprocessElectricalActions: "the geometry splits are enumerated in"
ToopInterfacesAssetTopoSimplified -> ToopImporterDcPreprocessStationRealisations: "station to realize a split against"
ToopImporterDcPreprocessElectricalActions -> ToopImporterDcPreprocessStationRealisations: "electrical splits"
ToopInterfacesAssetTopoSimplified -> ToopImporterDcPreprocessBbOutage: "reduced again with couplers closed"
ToopImporterDcPreprocessStationRealisations -> ToopImporterDcPreprocessBbOutage: "action set A"
ToopInterfacesAssetTopoRuntime -> ToopInterfacesStoredActionSet: "starting_bus_groups -- to reach the real switches"
ToopInterfacesAssetTopoSimplified -> ToopInterfacesStoredActionSet: "simplified_starting_bus_groups -- the ordering local_actions is indexed against"
ToopImporterDcPreprocessStationRealisations -> ToopInterfacesStoredActionSet: "physical switchings per action"
ProcessedGridGridSnapshot -> ToopImporterImportStageTopologyModel: "normalized network"
ToopImporterImportStageTopologyModel -> ToopInterfacesAssetTopoMaster: "bus groups, bays, circuit groups, possible connectivity"
ProcessedGridGridSnapshot -> ToopImporterDcPreprocessMaterialize: "live switch, coupler and busbar state"
ProcessedGridAssetTopoMaster -> ToopImporterDcPreprocessMaterialize: "canonical structure"
ToopImporterDcPreprocessMaterialize -> ToopInterfacesAssetTopoRuntime: "structure + what is closed now"
ToopInterfacesAssetTopoRuntime -> ToopImporterDcPreprocessSimplify: "full physical bus groups"
ToopImporterDcPreprocessSimplify -> ToopInterfacesAssetTopoSimplified: "one reduced slice per electrical node"
`;case`loadflowFormat`:return`direction: down

LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data

  LfMetadata: {
    label: "metadata.json"
    shape: document
  }
  LfBranch: {
    label: "branch_results.parquet"
    shape: document
  }
  LfNode: {
    label: "node_results.parquet"
    shape: document
  }
  LfConverged: {
    label: "converged.parquet"
    shape: document
  }
  LfVaDiff: {
    label: "va_diff_results.parquet"
    shape: document
  }
  LfReg: {
    label: "regulating_element_results.parquet"
    shape: document
  }
  LfSwitch: {
    label: "switch_results.parquet"
    shape: document
  }
  LfSpps: {
    label: "spps_results.parquet"
    shape: document
  }
  LfCascade: {
    label: "cascade_results.parquet"
    shape: document
  }
}
ToopInterfacesLfResults: {
  label: "LoadflowResults"
  shape: document
}
ToopImporterInitialLoadflow: {
  label: "run_initial_loadflow"
}
ToopAcValidator: {
  label: "AC-Validator"
}

ToopInterfacesLfResults -> LoadflowStore: "persisted per job"
ToopImporterInitialLoadflow -> LoadflowStore: "initial AC N-1 results"
LoadflowStore -> ToopAcValidator: "initial loadflow as baseline"
ToopAcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
`;case`index`:return`direction: down

Client: {
  label: "Operator / orchestration client"
  shape: c4-person
}
UnprocessedGridStore: {
  label: "Unprocessed grid store"
  shape: stored_data
}
Toop: {
  label: "ToOp Engine"

  Interfaces: {
    label: "Interfaces"
  }
  Postprocess: {
    label: "Postprocessing and export"
  }
  LfService: {
    label: "AC loadflow service"
  }
  ImporterParams: {
    label: "Importer parameters"
    shape: document
  }
  DcParams: {
    label: "DC optimizer parameters"
    shape: document
  }
  AcParams: {
    label: "AC validator parameters"
    shape: document
  }
  Importer: {
    label: "Importer"
  }
  DcOptimizer: {
    label: "DC-Optimizer"
  }
  AcValidator: {
    label: "AC-Validator"
  }
  Contingency: {
    label: "Contingency analysis"
  }
}
Kafka: {
  label: "Kafka"
  shape: queue
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}
ProcessedGrid: {
  label: "Processed grid folder"
  shape: stored_data
}
Downstream: {
  label: "Frontend / downstream systems"
}

Client -> Kafka: "[...]"
Kafka -> Downstream: "validated topologies for review"
ProcessedGrid -> Downstream: "UCTE, DGS, OpenRAO summaries, single line diagrams"
Client -> Toop.ImporterParams: "set in StartPreprocessingCommand"
Client -> Toop.DcParams: "set in StartOptimizationCommand"
Client -> Toop.AcParams: "set in the same StartOptimizationCommand"
Kafka -> Toop.Importer: "consumes command"
Kafka -> Toop.DcOptimizer: "consumes command"
Kafka -> Toop.AcValidator: "[...]"
UnprocessedGridStore -> Toop.Importer: "raw grid file"
ProcessedGrid -> Toop.Importer: "[...]"
ProcessedGrid -> Toop.DcOptimizer: "[...]"
ProcessedGrid -> Toop.AcValidator: "[...]"
LoadflowStore -> Toop.AcValidator: "initial loadflow as baseline"
Toop.Importer -> Kafka: "[...]"
Toop.Importer -> ProcessedGrid: "[...]"
Toop.Importer -> LoadflowStore: "initial AC N-1 results"
Toop.DcOptimizer -> Kafka: "[...]"
Toop.AcValidator -> Kafka: "[...]"
Toop.AcValidator -> ProcessedGrid: "summaries and diagrams"
Toop.AcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
Toop.Interfaces -> ProcessedGrid: "serialized once per import"
Toop.Interfaces -> LoadflowStore: "persisted per job"
Toop.Postprocess -> ProcessedGrid: "[...]"
Toop.Importer -> Toop.Contingency: "base grid N-1"
Toop.Importer -> Toop.Interfaces: "[...]"
Toop.Interfaces -> Toop.Importer: "[...]"
Toop.ImporterParams -> Toop.Importer: "scope, limits, contingencies"
Toop.DcParams -> Toop.DcOptimizer: "search bounds, fitness, operator probabilities"
Toop.AcValidator -> Toop.Contingency: "[...]"
Toop.AcValidator -> Toop.Interfaces: "accepted topology"
Toop.AcParams -> Toop.AcValidator: "compute budget, pruning, rejection thresholds"
Toop.Contingency -> Toop.Interfaces: "[...]"
Toop.Interfaces -> Toop.Contingency: "monitored elements and contingencies"
Toop.Postprocess -> Toop.Contingency: "grid with topology applied"
Toop.Interfaces -> Toop.Postprocess: "[...]"
Toop.Postprocess -> Toop.Interfaces: "switch id and new state"
`;case`overview`:return`direction: right

Client: {
  label: "Operator / orchestration client"
  shape: c4-person
}
KafkaImporterCommands: {
  label: "importer_commands"
  shape: queue
}
UnprocessedGridStore: {
  label: "Unprocessed grid store"
  shape: stored_data
}
ToopImporter: {
  label: "Importer"
}
ProcessedGrid: {
  label: "Processed grid folder"
  shape: stored_data
}
LoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}
KafkaImporterResults: {
  label: "importer_results"
  shape: queue
}
KafkaCommands: {
  label: "commands"
  shape: queue
}
ToopDcOptimizer: {
  label: "DC-Optimizer"
}
ToopAcValidator: {
  label: "AC-Validator"
}
KafkaResults: {
  label: "results"
  shape: queue
}
Downstream: {
  label: "Frontend / downstream systems"
}

Client -> KafkaImporterCommands: "StartPreprocessingCommand"
KafkaImporterCommands -> ToopImporter: "picks up the job"
UnprocessedGridStore -> ToopImporter: "UCTE / CGMES / PowerFactory file"
ToopImporter -> ProcessedGrid: "normalized snapshot, masks, asset topology"
ToopImporter -> ProcessedGrid: "PTDF, action set, contingency set"
ToopImporter -> LoadflowStore: "initial AC N-1 and reference metrics"
ToopImporter -> KafkaImporterResults: "PreprocessingSuccessResult"
KafkaImporterResults -> Client: "data folder is ready"
Client -> KafkaCommands: "StartOptimizationCommand"
KafkaCommands -> ToopDcOptimizer: "starts the DC run"
KafkaCommands -> ToopAcValidator: "starts the AC run"
ToopDcOptimizer -> ProcessedGrid: "loads static information onto the GPU"
ToopAcValidator -> ProcessedGrid: "loads base grid and action set"
LoadflowStore -> ToopAcValidator: "reads the initial loadflow as baseline"
ToopDcOptimizer -> KafkaResults: "Strategy, once per epoch"
KafkaResults -> ToopAcValidator: "DC topologies to validate"
ToopAcValidator -> ToopAcValidator: "prune, worst-k, then full N-1"
ToopAcValidator -> LoadflowStore: "AC loadflow results per evaluated topology"
ToopAcValidator -> KafkaResults: "AC-validated Strategy, referencing its loadflow"
ToopAcValidator -> ProcessedGrid: "summaries, diagrams, loadflow tables"
KafkaResults -> Downstream: "topologies for review"
ProcessedGrid -> Downstream: "UCTE, DGS, OpenRAO summaries, single line diagrams"
`;case`parameters`:return`direction: down

Client: {
  label: "Operator / orchestration client"
  shape: c4-person
}
Toop: {
  label: "ToOp Engine"

  ImporterParams: {
    label: "Importer parameters"
    shape: document

    PAreaSettings: {
      label: "AreaSettings"
      shape: document
    }
    PStationRules: {
      label: "RelevantStationRules"
      shape: document
    }
    PLists: {
      label: "White / black / ignore lists"
      shape: document
    }
    PContingencies: {
      label: "Contingency list"
      shape: document
    }
    PPreprocess: {
      label: "PreprocessParameters"
      shape: document
    }
  }
  DcParams: {
    label: "DC optimizer parameters"
    shape: document

    PMe: {
      label: "BatchedMEParameters (ga_config)"
      shape: document
    }
    PSolver: {
      label: "LoadflowSolverParameters"
      shape: document
    }
    PDoubleLimits: {
      label: "DoubleLimitsSetpoint"
      shape: document
    }
  }
  AcParams: {
    label: "AC validator parameters"
    shape: document

    PAcGa: {
      label: "ACGAParameters (ga_config)"
      shape: document
    }
    PRejection: {
      label: "Rejection thresholds"
      shape: document
    }
    PInitialLoadflow: {
      label: "initial_loadflow reference"
      shape: document
    }
  }
  Importer: {
    label: "Importer"
  }
  DcOptimizer: {
    label: "DC-Optimizer"
  }
  AcValidator: {
    label: "AC-Validator"
  }
}

Client -> Toop.ImporterParams: "set in StartPreprocessingCommand"
Toop.ImporterParams -> Toop.Importer: "scope, limits, contingencies"
Client -> Toop.DcParams: "set in StartOptimizationCommand"
Toop.DcParams -> Toop.DcOptimizer: "search bounds, fitness, operator probabilities"
Client -> Toop.AcParams: "set in the same StartOptimizationCommand"
Toop.AcParams -> Toop.AcValidator: "compute budget, pruning, rejection thresholds"
`;default:throw Error(`Unknown viewId: `+e)}};export{e as d2Source};