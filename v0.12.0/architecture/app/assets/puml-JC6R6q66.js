var e=e=>{switch(e){case`dataFlow`:return`@startuml
title "Technology and data flow"
left to right direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Client>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam database<<UnprocessedGridStore>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridGridSnapshot>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam queue<<KafkaImporterCommands>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam queue<<KafkaCommands>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopImporter>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam queue<<KafkaImporterResults>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam queue<<KafkaImporterHeartbeat>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ProcessedGridStaticInfo>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ProcessedGridActionSet>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopDcOptimizer>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam queue<<KafkaResults>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopAcValidator>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam queue<<KafkaHeartbeat>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ProcessedGridSnapshots>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopLfService>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<Downstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<Client>> as Client
database "==Unprocessed grid store\\n<size:10>[fsspec AbstractFileSystem (unprocessed_gridfile_fs)]</size>\\n\\nWhere the source grid files land before anything touches them. The same\\nkind of thing as the loadflow result store -- an fsspec filesystem the\\nworker is handed, local disk or object storage depending on how it is\\nwired -- not an external system the engine talks to.\\n\\nThe importer reads from it and never writes back. The concrete folder\\ncomes from the import command, as a path relative to this filesystem." <<UnprocessedGridStore>> as UnprocessedGridStore
rectangle "Processed grid folder" <<ProcessedGrid>> as ProcessedGrid {
  skinparam RectangleBorderColor<<ProcessedGrid>> #428a4f
  skinparam RectangleFontColor<<ProcessedGrid>> #428a4f
  skinparam RectangleBorderStyle<<ProcessedGrid>> dashed

  rectangle "==grid.xiidm / grid.json\\n\\nThe normalized backend grid, written by the importer." <<ProcessedGridGridSnapshot>> as ProcessedGridGridSnapshot
  rectangle "==static_information.hdf5\\n\\nThe critical asset: everything the GPU needs, and nothing it does not.\\nOne serialized StaticInformation -- a SolverConfig, which is static and\\npart of the JIT signature, plus a DynamicInformation holding every\\narray. Expand for what is inside." <<ProcessedGridStaticInfo>> as ProcessedGridStaticInfo
  rectangle "==action_set.json + action_set_diffs.hdf5\\n\\nThe same action space in physical terms: station-local reconfigurations\\nA and disconnectable branches D, expressed as switch positions against\\nthe asset topology.\\n\\nThis is the postprocessing and export side -- realizing a topology,\\nwriting DGS or UCTE. The optimizer does not read it; it samples the\\nBranchActionSet in static_information.hdf5 instead. The two are built\\nfrom the same enumeration and share an ordering, which is what lets an\\naction index found on the GPU be translated back into switches." <<ProcessedGridActionSet>> as ProcessedGridActionSet
  rectangle "==optimizer_snapshots/ac\\n\\nRepertoire, realized asset topologies, AC/DC loadflow tables, SLDs, OpenRAO summaries." <<ProcessedGridSnapshots>> as ProcessedGridSnapshots
}
rectangle "Kafka" <<Kafka>> as Kafka {
  skinparam RectangleBorderColor<<Kafka>> #A35829
  skinparam RectangleFontColor<<Kafka>> #A35829
  skinparam RectangleBorderStyle<<Kafka>> dashed

  queue "==importer_commands\\n\\nStartPreprocessingCommand, ShutdownCommand. 24 partitions." <<KafkaImporterCommands>> as KafkaImporterCommands
  queue "==commands\\n\\nStartOptimizationCommand, ShutdownCommand. 4 partitions." <<KafkaCommands>> as KafkaCommands
  queue "==importer_results\\n\\nPreprocessingStartedResult, PreprocessingSuccessResult, ErrorResult" <<KafkaImporterResults>> as KafkaImporterResults
  queue "==importer_heartbeat\\n\\nPreprocessHeartbeat carrying the current PreprocessStage" <<KafkaImporterHeartbeat>> as KafkaImporterHeartbeat
  queue "==results\\n\\nThe one shared topic. Both stages publish topologies here and the\\nAC-Validator also consumes it to pick up DC candidates." <<KafkaResults>> as KafkaResults
  queue "==heartbeat\\n\\nHeartbeat tagged with OptimizerType.DC or OptimizerType.AC" <<KafkaHeartbeat>> as KafkaHeartbeat
}
rectangle "ToOp Engine" <<Toop>> as Toop {
  skinparam RectangleBorderColor<<Toop>> #64748b
  skinparam RectangleFontColor<<Toop>> #64748b
  skinparam RectangleBorderStyle<<Toop>> dashed

  rectangle "==Importer\\n<size:10>[Python, PyPowSyBl, pandapower, JAX]</size>\\n\\nNormalizes a raw grid into a processed grid folder and derives the\\nsolver artifacts. Most of it depends only on the initial grid topology,\\nso it can run before the forecast is available. Runs one job at a time." <<ToopImporter>> as ToopImporter
  rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<ToopDcOptimizer>> as ToopDcOptimizer
  rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<ToopAcValidator>> as ToopAcValidator
  rectangle "==AC loadflow service\\n<size:10>[Python, PyPowSyBl]</size>\\n\\nA standalone N-1 service on its own loadflow_commands / loadflow_results\\n/ loadflow_heartbeat topics. Present in the codebase but off the main\\npath: dev-deployment does not create its topics, and the importer only\\nborrows a grid-loading helper from it." <<ToopLfService>> as ToopLfService
}
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<Downstream>> as Downstream

Client .[#8D8D8D,thickness=2].> KafkaImporterCommands : <color:#8D8D8D>StartPreprocessingCommand
Client .[#8D8D8D,thickness=2].> KafkaCommands : <color:#8D8D8D>StartOptimizationCommand
UnprocessedGridStore .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>raw grid file
KafkaImporterCommands .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>consumes command
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>[...]
ToopImporter .[#8D8D8D,thickness=2].> KafkaImporterResults : <color:#8D8D8D>PreprocessingSuccessResult
ToopImporter .[#8D8D8D,thickness=2].> KafkaImporterHeartbeat : <color:#8D8D8D>PreprocessHeartbeat per stage
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGridStaticInfo : <color:#8D8D8D>[...]
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGridActionSet : <color:#8D8D8D>the same actions as physical switchings
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGridGridSnapshot : <color:#8D8D8D>normalized snapshot
ToopImporter .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 results
KafkaCommands .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>consumes command
ProcessedGridStaticInfo .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>loaded onto the GPU at startup
ToopDcOptimizer .[#8D8D8D,thickness=2].> KafkaResults : <color:#8D8D8D>TopologyPushResult per epoch
ToopDcOptimizer .[#8D8D8D,thickness=2].> KafkaHeartbeat : <color:#8D8D8D>OptimizationStatsHeartbeat
KafkaCommands .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>consumes the same command
KafkaResults .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>DC topologies
ProcessedGridActionSet .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>to realize topologies
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>base grid
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>initial loadflow as baseline
ToopAcValidator .[#8D8D8D,thickness=2].> KafkaResults : <color:#8D8D8D>AC-validated Strategy
ToopAcValidator .[#8D8D8D,thickness=2].> KafkaHeartbeat : <color:#8D8D8D>OptimizationStatsHeartbeat
ToopAcValidator .[#8D8D8D,thickness=2].> ProcessedGridSnapshots : <color:#8D8D8D>summaries and diagrams
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
KafkaResults .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>validated topologies for review
ProcessedGridSnapshots .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>UCTE, DGS, OpenRAO summaries, single line diagrams
@enduml
`;case`importerInternals`:return`@startuml
title "Importer -- what happens to a grid file"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam database<<UnprocessedGridStore>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridGridSnapshot>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridAssetTopoMaster>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageLoadGrid>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ProcessedGridMasks>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageWhitelists>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterDcPreprocessMaterialize>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterInitialLoadflow>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<ProcessedGridStaticInfoBranchActionSet>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridActionSet>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridNminus1>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageConvergingParams>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterDcPreprocessBridges>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageNetworkMasks>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterDcPreprocessRelevantNodes>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterImportStageTopologyModel>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterDcPreprocessFactors>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessReduce>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessNminus2Filter>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessSimplify>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessElectricalActions>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessStationRealisations>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporterDcPreprocessBbOutage>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
rectangle "Processed grid folder" <<ProcessedGrid>> as ProcessedGrid {
  skinparam RectangleBorderColor<<ProcessedGrid>> #64748b
  skinparam RectangleFontColor<<ProcessedGrid>> #64748b
  skinparam RectangleBorderStyle<<ProcessedGrid>> dashed

  rectangle "static_information.hdf5" <<ProcessedGridStaticInfo>> as ProcessedGridStaticInfo {
    skinparam RectangleBorderColor<<ProcessedGridStaticInfo>> #64748b
    skinparam RectangleFontColor<<ProcessedGridStaticInfo>> #64748b
    skinparam RectangleBorderStyle<<ProcessedGridStaticInfo>> dashed

    rectangle "==BranchActionSet\\n\\nWhat the DC-Optimizer actually samples from -- a different asset from\\naction_set.json, in a different format and a different file.\\n\\nPadded boolean arrays (branch_actions, inj_actions) concatenated\\nacross substations, with n_actions_per_sub and action_start_indices\\nto index into them, the unsplit_action_mask, and the precomputed\\nreassignment_distance per action. A genome is an index into this, not\\ninto the JSON.\\n\\nAlso carries RelBBOutageData: the busbar outage data for relevant\\nstations lives here rather than with the other outage sets, because it\\nis per-action." <<ProcessedGridStaticInfoBranchActionSet>> as ProcessedGridStaticInfoBranchActionSet
  }
  rectangle "==grid.xiidm / grid.json\\n\\nThe normalized backend grid, written by the importer." <<ProcessedGridGridSnapshot>> as ProcessedGridGridSnapshot
  rectangle "==initial_topology/asset_topology_master_data.json\\n\\nA serialized MasterAssetTopology, and the only form of the asset\\ntopology that gets a file of its own. Written by the importer, read back\\nat the start of DC preprocessing. The runtime and simplified forms are\\nderived from it in memory and reach disk only inside action_set.json." <<ProcessedGridAssetTopoMaster>> as ProcessedGridAssetTopoMaster
  rectangle "==masks/*.npy\\n\\n~35 boolean and weight masks per asset class: which branches count for\\nN-1, which are disconnectable, overload weights, TSO/DSO borders,\\nblacklists." <<ProcessedGridMasks>> as ProcessedGridMasks
  rectangle "==action_set.json + action_set_diffs.hdf5\\n\\nThe same action space in physical terms: station-local reconfigurations\\nA and disconnectable branches D, expressed as switch positions against\\nthe asset topology.\\n\\nThis is the postprocessing and export side -- realizing a topology,\\nwriting DGS or UCTE. The optimizer does not read it; it samples the\\nBranchActionSet in static_information.hdf5 instead. The two are built\\nfrom the same enumeration and share an ordering, which is what lets an\\naction index found on the GPU be translated back into switches." <<ProcessedGridActionSet>> as ProcessedGridActionSet
  rectangle "==nminus1_definition.json\\n\\nThe contingency set, written by the importer and refreshed by DC preprocessing." <<ProcessedGridNminus1>> as ProcessedGridNminus1
}
database "==Unprocessed grid store\\n<size:10>[fsspec AbstractFileSystem (unprocessed_gridfile_fs)]</size>\\n\\nWhere the source grid files land before anything touches them. The same\\nkind of thing as the loadflow result store -- an fsspec filesystem the\\nworker is handed, local disk or object storage depending on how it is\\nwired -- not an external system the engine talks to.\\n\\nThe importer reads from it and never writes back. The concrete folder\\ncomes from the import command, as a path relative to this filesystem." <<UnprocessedGridStore>> as UnprocessedGridStore
rectangle "Importer" <<ToopImporter>> as ToopImporter {
  skinparam RectangleBorderColor<<ToopImporter>> #64748b
  skinparam RectangleFontColor<<ToopImporter>> #64748b
  skinparam RectangleBorderStyle<<ToopImporter>> dashed

  rectangle "convert_file" <<ToopImporterImportStage>> as ToopImporterImportStage {
    skinparam RectangleBorderColor<<ToopImporterImportStage>> #3b82f6
    skinparam RectangleFontColor<<ToopImporterImportStage>> #3b82f6
    skinparam RectangleBorderStyle<<ToopImporterImportStage>> dashed

    rectangle "==Load and merge\\n\\nParse UCTE/CGMES/PowerFactory. Dominates importer runtime on CGMES." <<ToopImporterImportStageLoadGrid>> as ToopImporterImportStageLoadGrid
    rectangle "==Apply whitelists\\n\\nApply the CB / black- and whitelists that scope the switchable area." <<ToopImporterImportStageWhitelists>> as ToopImporterImportStageWhitelists
    rectangle "==find_converging_loadflow_params\\n\\nSweep loadflow parameters and voltage init methods until the base\\ncase converges. Some grid files do not converge on defaults." <<ToopImporterImportStageConvergingParams>> as ToopImporterImportStageConvergingParams
    rectangle "==get_network_masks\\n\\nBuild the per-asset masks, then derive the initial N-1 definition from them." <<ToopImporterImportStageNetworkMasks>> as ToopImporterImportStageNetworkMasks
    rectangle "==get_master_asset_topology_artifact\\n\\nExtraction. Dispatches on the importer data_type and hands off to\\none of the readers below -- which is the whole reason the rest of\\nthe engine never has to know which framework parsed the file." <<ToopImporterImportStageTopologyModel>> as ToopImporterImportStageTopologyModel
  }
  rectangle "load_grid (DC preprocessing)" <<ToopImporterDcPreprocess>> as ToopImporterDcPreprocess {
    skinparam RectangleBorderColor<<ToopImporterDcPreprocess>> #6366f1
    skinparam RectangleFontColor<<ToopImporterDcPreprocess>> #6366f1
    skinparam RectangleBorderStyle<<ToopImporterDcPreprocess>> dashed

    rectangle "==get_runtime_asset_topology\\n\\nTransition 1. Reads the master data back and materializes it against\\nthe loaded network: structure from the importer artifact, switch and\\nbusbar states from the grid file. Backend-specific, and it refuses\\nto continue if the live state comes out narrower than the\\nconnectivity the master data declared -- silently losing a possible\\nbusbar assignment here would silently shrink the action space later.\\n\\nWorth knowing that the two inputs are separable by construction\\nrather than by accident: structure arrives as a MasterAssetTopology\\nand live state as a compact RuntimeSwitchingState overlay, through\\nseparate parameters, and neither reader reaches back for the other.\\nToday both are read from the same normalized snapshot, so the seam\\nis invisible -- but nothing here requires one file, and a case\\nderived elsewhere (outages in force, concrete nodal injections)\\nwould enter exactly here, against unchanged master structure." <<ToopImporterDcPreprocessMaterialize>> as ToopImporterDcPreprocessMaterialize
    rectangle "==compute_bridging_branches\\n\\nTarjan bridge finding. A branch whose removal islands the grid,\\nunder N-0 or any contingency, cannot be disconnected." <<ToopImporterDcPreprocessBridges>> as ToopImporterDcPreprocessBridges
    rectangle "==filter_relevant_nodes\\n\\nDrop substations that are not worth switching: too few branches,\\nno assets, or double connections." <<ToopImporterDcPreprocessRelevantNodes>> as ToopImporterDcPreprocessRelevantNodes
    rectangle "==compute PTDF / PSDF\\n\\nThe reference PTDF matrix, solved once. Every topology the optimizer\\nlater evaluates is a low-rank update of it rather than a refactorization." <<ToopImporterDcPreprocessFactors>> as ToopImporterDcPreprocessFactors
    rectangle "==reduce node / branch dimension\\n\\nCollapse nodes that never change into a single static column and\\ndrop branches that are neither monitored, outaged nor switched.\\nDirectly shrinks the PTDF the GPU has to hold." <<ToopImporterDcPreprocessReduce>> as ToopImporterDcPreprocessReduce
    rectangle "==filter_disconnectable_branches_nminus2\\n\\nExclude branches that island the grid in combination with a contingency." <<ToopImporterDcPreprocessNminus2Filter>> as ToopImporterDcPreprocessNminus2Filter
    rectangle "==simplify_asset_topology\\n\\nTransition 2. Projects each relevant station onto one electrical\\nnode at a time and runs prepare_for_separation_set on the slice.\\nStations that survive become the simplified topology; the ones that\\ndo not are removed from the relevant set here rather than failing\\nthe run.\\n\\nRuns late, after the dimension reductions, because it needs the\\nfinal branch and injection ordering to project onto. Busbar-outage\\npreprocessing runs the same reduction a second time with couplers\\nforced closed, for a separate view of its own." <<ToopImporterDcPreprocessSimplify>> as ToopImporterDcPreprocessSimplify
    rectangle "==compute_electrical_actions\\n\\nStage one of action set enumeration: every electrically distinct\\ntwo-node split of a station, filtered for islanding and\\nconnectivity, clipped if a station exceeds the configured limit." <<ToopImporterDcPreprocessElectricalActions>> as ToopImporterDcPreprocessElectricalActions
    rectangle "==enumerate_station_realisations\\n\\nStage two: map each electrical split onto a reachable node-breaker\\nrealization and precompute its reassignment distance. Splits with\\nno valid realization are discarded." <<ToopImporterDcPreprocessStationRealisations>> as ToopImporterDcPreprocessStationRealisations
    rectangle "==preprocess_bb_outage\\n\\nOptional busbar outage contingencies, used by the do-not-make-it-worse criterion." <<ToopImporterDcPreprocessBbOutage>> as ToopImporterDcPreprocessBbOutage
  }
  rectangle "==run_initial_loadflow\\n<size:10>[PyPowSyBl]</size>\\n\\nFull AC N-1 on the unmodified grid. Produces the reference metrics\\nevery proposed topology is later compared against." <<ToopImporterInitialLoadflow>> as ToopImporterInitialLoadflow
}
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore

ToopImporterInitialLoadflow .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 results
ToopImporterImportStageLoadGrid .[#8D8D8D,thickness=2].> ToopImporterImportStageWhitelists : <color:#8D8D8D>parsed network
ToopImporterImportStageWhitelists .[#8D8D8D,thickness=2].> ToopImporterImportStageConvergingParams : <color:#8D8D8D>scoped network
ToopImporterImportStageConvergingParams .[#8D8D8D,thickness=2].> ToopImporterImportStageNetworkMasks : <color:#8D8D8D>converging parameters
ToopImporterImportStageNetworkMasks .[#8D8D8D,thickness=2].> ToopImporterImportStageTopologyModel : <color:#8D8D8D>masks
ToopImporterDcPreprocessMaterialize .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessBridges : <color:#8D8D8D>runtime topology on NetworkData
ToopImporterDcPreprocessBridges .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessRelevantNodes : <color:#8D8D8D>bridge flags
ToopImporterDcPreprocessRelevantNodes .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessFactors : <color:#8D8D8D>switchable subset
ToopImporterDcPreprocessFactors .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessReduce : <color:#8D8D8D>PTDF / PSDF
ToopImporterDcPreprocessReduce .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessNminus2Filter : <color:#8D8D8D>reduced dimensions
ToopImporterDcPreprocessNminus2Filter .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessSimplify : <color:#8D8D8D>final branch and injection ordering
ToopImporterDcPreprocessNminus2Filter .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessElectricalActions : <color:#8D8D8D>disconnectable set D
ToopImporterDcPreprocessSimplify .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessElectricalActions : <color:#8D8D8D>reduced stations to enumerate in
ToopImporterDcPreprocessElectricalActions .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessStationRealisations : <color:#8D8D8D>electrical splits
ToopImporterDcPreprocessStationRealisations .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessBbOutage : <color:#8D8D8D>action set A
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopImporterImportStageTopologyModel : <color:#8D8D8D>normalized network
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessMaterialize : <color:#8D8D8D>live switch, coupler and busbar state
ProcessedGridAssetTopoMaster .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessMaterialize : <color:#8D8D8D>canonical structure
UnprocessedGridStore .[#8D8D8D,thickness=2].> ToopImporterImportStage : <color:#8D8D8D>raw grid file
ToopImporterImportStage .[#8D8D8D,thickness=2].> ToopImporterDcPreprocess : <color:#8D8D8D>ImportResult
ToopImporterDcPreprocess .[#8D8D8D,thickness=2].> ToopImporterInitialLoadflow : <color:#8D8D8D>ready grid folder
ToopImporterDcPreprocess .[#8D8D8D,thickness=2].> ProcessedGridStaticInfoBranchActionSet : <color:#8D8D8D>padded action arrays for the GPU
ToopImporterDcPreprocess .[#8D8D8D,thickness=2].> ProcessedGridActionSet : <color:#8D8D8D>the same actions as physical switchings
ToopImporterImportStage .[#8D8D8D,thickness=2].> ProcessedGridMasks : <color:#8D8D8D>per-asset masks
ToopImporterImportStage .[#8D8D8D,thickness=2].> ProcessedGridNminus1 : <color:#8D8D8D>initial contingency set
ToopImporterDcPreprocess .[#8D8D8D,thickness=2].> ProcessedGridNminus1 : <color:#8D8D8D>refreshed contingency set
@enduml
`;case`dcWorkerInternals`:return`@startuml
title "DC-Optimizer -- repertoire, variation, scoring"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<ProcessedGridStaticInfoBranchActionSet>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopDcOptimizerScoring>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopDcOptimizerRepertoire>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopDcOptimizerMutation>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopDcOptimizerCrossover>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopDcOptimizerPusher>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopDcOptimizerDcSolverBsdfStage>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam queue<<KafkaResults>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopDcOptimizerDcSolverN0Stage>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopDcOptimizerDcSolverN1Stage>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopDcOptimizerDcSolverResultExtraction>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
rectangle "static_information.hdf5" <<ProcessedGridStaticInfo>> as ProcessedGridStaticInfo {
  skinparam RectangleBorderColor<<ProcessedGridStaticInfo>> #64748b
  skinparam RectangleFontColor<<ProcessedGridStaticInfo>> #64748b
  skinparam RectangleBorderStyle<<ProcessedGridStaticInfo>> dashed

  rectangle "==BranchActionSet\\n\\nWhat the DC-Optimizer actually samples from -- a different asset from\\naction_set.json, in a different format and a different file.\\n\\nPadded boolean arrays (branch_actions, inj_actions) concatenated\\nacross substations, with n_actions_per_sub and action_start_indices\\nto index into them, the unsplit_action_mask, and the precomputed\\nreassignment_distance per action. A genome is an index into this, not\\ninto the JSON.\\n\\nAlso carries RelBBOutageData: the busbar outage data for relevant\\nstations lives here rather than with the other outage sets, because it\\nis per-action." <<ProcessedGridStaticInfoBranchActionSet>> as ProcessedGridStaticInfoBranchActionSet
}
rectangle "DC-Optimizer" <<ToopDcOptimizer>> as ToopDcOptimizer {
  skinparam RectangleBorderColor<<ToopDcOptimizer>> #64748b
  skinparam RectangleFontColor<<ToopDcOptimizer>> #64748b
  skinparam RectangleBorderStyle<<ToopDcOptimizer>> dashed

  rectangle "==Scoring\\n\\nTurns raw flows into the metric vector -- overload energy, critical\\nbranch counts under N-0 and N-1, busbar outage penalty -- and\\naggregates it into the scalar fitness that is maximized. Overload\\nenergy dominates; the branch counts are weighted penalties on top." <<ToopDcOptimizerScoring>> as ToopDcOptimizerScoring
  rectangle "==Discrete MAP-Elites repertoire\\n\\nCells indexed by the switching-distance descriptors: disconnections,\\nsplit substations and reassignment distance. Each cell keeps its own\\nelites (cell_depth), so a conservative topology is never outcompeted\\nby an aggressive one -- that is what illuminates the Pareto front\\ninstead of collapsing to a single optimum." <<ToopDcOptimizerRepertoire>> as ToopDcOptimizerRepertoire
  rectangle "==Mutation\\n\\nPer genome, a Poisson-sampled number of substation mutations followed\\nby one disconnection mutation, each drawing ADD / CHANGE / REMOVE /\\nIDENTITY. Feasibility is enforced while sampling: no station split\\ntwice, no branch disconnected twice. PST taps mutate separately." <<ToopDcOptimizerMutation>> as ToopDcOptimizerMutation
  rectangle "==Crossover\\n\\nBuilds an offspring by sampling actions and disconnections from the\\nunion of two parents, biased toward the first parent." <<ToopDcOptimizerCrossover>> as ToopDcOptimizerCrossover
  rectangle "==Epoch result push\\n\\nHow topologies leave the DC stage. At the end of each epoch the new\\nelites are pulled off the GPU, converted to TopologyPushResult\\nmessages and produced to the \`results\` topic -- this is the only\\nthing that publishes DC topologies.\\n\\nBatching by epoch is the point: pushing per iteration would break the\\nGPU loop for a host transfer every time. Per epoch it lands every few\\nseconds." <<ToopDcOptimizerPusher>> as ToopDcOptimizerPusher
  rectangle "GPU DC loadflow solver" <<ToopDcOptimizerDcSolver>> as ToopDcOptimizerDcSolver {
    skinparam RectangleBorderColor<<ToopDcOptimizerDcSolver>> #6366f1
    skinparam RectangleFontColor<<ToopDcOptimizerDcSolver>> #6366f1
    skinparam RectangleBorderStyle<<ToopDcOptimizerDcSolver>> dashed

    rectangle "==compute_bsdf_lodf_static_flows\\n\\nEverything that changes the PTDF, in one pass per branch topology:\\nBSDF for each busbar split, MODF for the remedial disconnections,\\nthe LODF matrix, and the static flows. Optionally applies non-linear\\nPST tap susceptances. Returns TopologyResults." <<ToopDcOptimizerDcSolverBsdfStage>> as ToopDcOptimizerDcSolverBsdfStage
    rectangle "==N-0 flows\\n\\nNodal injections against the already-updated PTDF, plus the\\ncross-coupler flows across each split, corrected for disconnections\\nand PST taps. Cheap, because the PTDF is fixed by this point." <<ToopDcOptimizerDcSolverN0Stage>> as ToopDcOptimizerDcSolverN0Stage
    rectangle "==Contingency analysis (N-1)\\n\\nThe N-1 matrix from the LODF and multi-outage factors, plus busbar\\noutage and injection outage cases. Runs over the whole batch." <<ToopDcOptimizerDcSolverN1Stage>> as ToopDcOptimizerDcSolverN1Stage
    rectangle "==Result aggregation and sparsification\\n\\nThe full N-1 matrix is far too large to keep per topology, so only\\nthe worst entries survive: a top-k over the flattened matrix for\\nstorage, and a per-case worst-k that tells the AC-Validator which\\ncontingencies to re-check first." <<ToopDcOptimizerDcSolverResultExtraction>> as ToopDcOptimizerDcSolverResultExtraction
  }
}
rectangle "Kafka" <<Kafka>> as Kafka {
  skinparam RectangleBorderColor<<Kafka>> #64748b
  skinparam RectangleFontColor<<Kafka>> #64748b
  skinparam RectangleBorderStyle<<Kafka>> dashed

  queue "==results\\n\\nThe one shared topic. Both stages publish topologies here and the\\nAC-Validator also consumes it to pick up DC candidates." <<KafkaResults>> as KafkaResults
}

ToopDcOptimizerRepertoire .[#8D8D8D,thickness=2].> ToopDcOptimizerMutation : <color:#8D8D8D>sampled elites
ToopDcOptimizerRepertoire .[#8D8D8D,thickness=2].> ToopDcOptimizerCrossover : <color:#8D8D8D>sampled pairs
ToopDcOptimizerRepertoire .[#8D8D8D,thickness=2].> ToopDcOptimizerPusher : <color:#8D8D8D>new elites
ToopDcOptimizerScoring .[#8D8D8D,thickness=2].> ToopDcOptimizerRepertoire : <color:#8D8D8D>fitness and descriptors, sorted insert
ToopDcOptimizerDcSolverBsdfStage .[#8D8D8D,thickness=2].> ToopDcOptimizerDcSolverN0Stage : <color:#8D8D8D>updated PTDF, LODF, MODF, static flows
ToopDcOptimizerDcSolverN0Stage .[#8D8D8D,thickness=2].> ToopDcOptimizerDcSolverN1Stage : <color:#8D8D8D>N-0 flows and nodal injections
ToopDcOptimizerDcSolverN1Stage .[#8D8D8D,thickness=2].> ToopDcOptimizerDcSolverResultExtraction : <color:#8D8D8D>N-1 matrix
ToopDcOptimizerPusher .[#8D8D8D,thickness=2].> KafkaResults : <color:#8D8D8D>TopologyPushResult per epoch
ToopDcOptimizerMutation .[#8D8D8D,thickness=2].> ToopDcOptimizerDcSolver : <color:#8D8D8D>candidate batch
ToopDcOptimizerCrossover .[#8D8D8D,thickness=2].> ToopDcOptimizerDcSolver : <color:#8D8D8D>candidate batch
ToopDcOptimizerDcSolver .[#8D8D8D,thickness=2].> ToopDcOptimizerScoring : <color:#8D8D8D>N-0 and N-1 flows
ProcessedGridStaticInfoBranchActionSet .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>sampling space -- indices into these arrays
@enduml
`;case`acValidatorInternals`:return`@startuml
title "AC-Validator -- selection before validation"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<ProcessedGridGridSnapshot>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridActionSet>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam queue<<KafkaResults>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidatorResultListener>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidatorSelectStrategyDiscriminator>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopAcValidatorWorstK>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<ToopAcValidatorSelectStrategyDominator>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopAcValidatorRemainingCa>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<ToopAcValidatorSelectStrategyMedian>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopAcValidatorAcceptance>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopContingency>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidatorSummaryWriter>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfaces>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ProcessedGridSnapshots>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "Kafka" <<Kafka>> as Kafka {
  skinparam RectangleBorderColor<<Kafka>> #64748b
  skinparam RectangleFontColor<<Kafka>> #64748b
  skinparam RectangleBorderStyle<<Kafka>> dashed

  queue "==results\\n\\nThe one shared topic. Both stages publish topologies here and the\\nAC-Validator also consumes it to pick up DC candidates." <<KafkaResults>> as KafkaResults
}
rectangle "Processed grid folder" <<ProcessedGrid>> as ProcessedGrid {
  skinparam RectangleBorderColor<<ProcessedGrid>> #64748b
  skinparam RectangleFontColor<<ProcessedGrid>> #64748b
  skinparam RectangleBorderStyle<<ProcessedGrid>> dashed

  rectangle "==grid.xiidm / grid.json\\n\\nThe normalized backend grid, written by the importer." <<ProcessedGridGridSnapshot>> as ProcessedGridGridSnapshot
  rectangle "==action_set.json + action_set_diffs.hdf5\\n\\nThe same action space in physical terms: station-local reconfigurations\\nA and disconnectable branches D, expressed as switch positions against\\nthe asset topology.\\n\\nThis is the postprocessing and export side -- realizing a topology,\\nwriting DGS or UCTE. The optimizer does not read it; it samples the\\nBranchActionSet in static_information.hdf5 instead. The two are built\\nfrom the same enumeration and share an ordering, which is what lets an\\naction index found on the GPU be translated back into switches." <<ProcessedGridActionSet>> as ProcessedGridActionSet
  rectangle "==optimizer_snapshots/ac\\n\\nRepertoire, realized asset topologies, AC/DC loadflow tables, SLDs, OpenRAO summaries." <<ProcessedGridSnapshots>> as ProcessedGridSnapshots
}
rectangle "AC-Validator" <<ToopAcValidator>> as ToopAcValidator {
  skinparam RectangleBorderColor<<ToopAcValidator>> #64748b
  skinparam RectangleFontColor<<ToopAcValidator>> #64748b
  skinparam RectangleBorderStyle<<ToopAcValidator>> dashed

  rectangle "==Result listener\\n<size:10>[SQLite (in-memory), SQLModel]</size>\\n\\nSpools the results topic into a local database, at startup and\\nbetween epochs, so candidates are already staged when a run begins." <<ToopAcValidatorResultListener>> as ToopAcValidatorResultListener
  rectangle "select_strategy" <<ToopAcValidatorSelectStrategy>> as ToopAcValidatorSelectStrategy {
    skinparam RectangleBorderColor<<ToopAcValidatorSelectStrategy>> #A35829
    skinparam RectangleFontColor<<ToopAcValidatorSelectStrategy>> #A35829
    skinparam RectangleBorderStyle<<ToopAcValidatorSelectStrategy>> dashed

    rectangle "==Discriminator filter\\n\\nDrop candidates too close to something already validated." <<ToopAcValidatorSelectStrategyDiscriminator>> as ToopAcValidatorSelectStrategyDiscriminator
    rectangle "==Dominator filter\\n\\nDrop a candidate if another topology reaches similar or better DC\\nfitness at a lower switching distance." <<ToopAcValidatorSelectStrategyDominator>> as ToopAcValidatorSelectStrategyDominator
    rectangle "==Median filter\\n\\nDrop candidates whose fitness is below the median of their descriptor cell." <<ToopAcValidatorSelectStrategyMedian>> as ToopAcValidatorSelectStrategyMedian
  }
  rectangle "==Worst-k epoch\\n<size:10>[PyPowSyBl]</size>\\n\\nReruns only the handful of contingencies the DC stage flagged as worst\\nfor this topology. A candidate that already fails there, or converges\\npoorly, is rejected without the full analysis. CPU parallelism is\\ndisabled here -- at this few cases it costs more than it saves." <<ToopAcValidatorWorstK>> as ToopAcValidatorWorstK
  rectangle "==Remaining contingencies\\n<size:10>[PyPowSyBl security analysis, multiprocess]</size>\\n\\nFull AC N-1 on the survivors, batched over runner processes. Hundreds\\nof contingencies rather than a handful." <<ToopAcValidatorRemainingCa>> as ToopAcValidatorRemainingCa
  rectangle "==Acceptance evaluation\\n<size:10>[polars LazyFrame]</size>\\n\\nDetects constraint violations across the loadflow tables and decides\\nwhether a topology passes. Polars because the result volume is the\\nbottleneck, not the check itself." <<ToopAcValidatorAcceptance>> as ToopAcValidatorAcceptance
  rectangle "==Summary writer\\n\\nRealized asset topologies, loadflow tables, SLDs and OpenRAO summaries." <<ToopAcValidatorSummaryWriter>> as ToopAcValidatorSummaryWriter
}
rectangle "==Contingency analysis\\n<size:10>[toop_engine_contingency_analysis]</size>\\n\\nRuns an N-1 analysis against whichever backend holds the grid, and\\nnormalizes both into the same result object. The two backends are not\\nat feature parity, so which one you import with decides what you can\\nmeasure afterwards." <<ToopContingency>> as ToopContingency
rectangle "==Interfaces\\n<size:10>[toop_engine_interfaces]</size>\\n\\nThe shared vocabulary. Everything here exists so that two packages can\\nagree on a payload without depending on each other." <<ToopInterfaces>> as ToopInterfaces
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore

ToopAcValidatorWorstK .[#8D8D8D,thickness=2].> ToopAcValidatorRemainingCa : <color:#8D8D8D>survivors
ToopAcValidatorWorstK .[#8D8D8D,thickness=2].> ToopAcValidatorAcceptance : <color:#8D8D8D>worst-k results
ToopAcValidatorRemainingCa .[#8D8D8D,thickness=2].> ToopAcValidatorAcceptance : <color:#8D8D8D>full N-1 results
ToopAcValidatorAcceptance .[#8D8D8D,thickness=2].> ToopAcValidatorSummaryWriter : <color:#8D8D8D>accepted topologies
ToopAcValidatorWorstK .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>worst-k contingencies
ToopAcValidatorRemainingCa .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>full N-1
ToopAcValidatorSummaryWriter .[#8D8D8D,thickness=2].> ToopInterfaces : <color:#8D8D8D>accepted topology
ToopAcValidatorSelectStrategyDiscriminator .[#8D8D8D,thickness=2].> ToopAcValidatorSelectStrategyDominator : <color:#8D8D8D>survivors
ToopAcValidatorSelectStrategyDominator .[#8D8D8D,thickness=2].> ToopAcValidatorSelectStrategyMedian : <color:#8D8D8D>survivors
KafkaResults .[#8D8D8D,thickness=2].> ToopAcValidatorResultListener : <color:#8D8D8D>DC topologies
ToopAcValidatorSummaryWriter .[#8D8D8D,thickness=2].> ProcessedGridSnapshots : <color:#8D8D8D>summaries and diagrams
ToopInterfaces .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>persisted per job
ToopAcValidatorResultListener .[#8D8D8D,thickness=2].> ToopAcValidatorSelectStrategy : <color:#8D8D8D>candidate pool
ToopAcValidatorSelectStrategy .[#8D8D8D,thickness=2].> ToopAcValidatorWorstK : <color:#8D8D8D>selected batch
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>base grid
ProcessedGridActionSet .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>to realize topologies
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>initial loadflow as baseline
@enduml
`;case`contingencyAnalysis`:return`@startuml
title "Contingency analysis -- what each backend can do"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<ToopContingencyPwCaPwLimitCache>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfacesLfResultsBranchRes>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfacesLfResultsNodeRes>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfacesLfResultsRegRes>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfacesLfResultsVaDiffRes>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfacesLfResultsConvergedRes>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfacesLfResultsSwitchRes>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopInterfacesLfResultsConnectivityRes>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopInterfacesLfResultsSppsRes>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopInterfacesLfResultsCascadeRes>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopImporterInitialLoadflow>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidatorWorstK>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidatorRemainingCa>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopContingencyDispatcher>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopContingencyPpCaPpOutageGrouping>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopContingencyPpCaPpSlack>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopContingencyPpCaPpSpps>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopContingencyPpCaPpCascade>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "ToOp Engine" <<Toop>> as Toop {
  skinparam RectangleBorderColor<<Toop>> #64748b
  skinparam RectangleFontColor<<Toop>> #64748b
  skinparam RectangleBorderStyle<<Toop>> dashed

  rectangle "Contingency analysis" <<ToopContingency>> as ToopContingency {
    skinparam RectangleBorderColor<<ToopContingency>> #64748b
    skinparam RectangleFontColor<<ToopContingency>> #64748b
    skinparam RectangleBorderStyle<<ToopContingency>> dashed

    rectangle "run_contingency_analysis_powsybl" <<ToopContingencyPwCa>> as ToopContingencyPwCa {
      skinparam RectangleBorderColor<<ToopContingencyPwCa>> #6366f1
      skinparam RectangleFontColor<<ToopContingencyPwCa>> #6366f1
      skinparam RectangleBorderStyle<<ToopContingencyPwCa>> dashed

      rectangle "==Branch limit cache\\n\\nCaches operational limits across runs on the same network." <<ToopContingencyPwCaPwLimitCache>> as ToopContingencyPwCaPwLimitCache
    }
    rectangle "==get_ac_loadflow_results\\n\\nThe single entry point. Dispatches on the *type of the network\\nobject* -- a pandapowerNet goes one way, a PyPowSyBl Network the\\nother -- and raises if it is neither. Both branches return the same\\nresult type, which is what keeps callers backend-agnostic." <<ToopContingencyDispatcher>> as ToopContingencyDispatcher
    rectangle "run_contingency_analysis_pandapower" <<ToopContingencyPpCa>> as ToopContingencyPpCa {
      skinparam RectangleBorderColor<<ToopContingencyPpCa>> #A35829
      skinparam RectangleFontColor<<ToopContingencyPpCa>> #A35829
      skinparam RectangleBorderStyle<<ToopContingencyPpCa>> dashed

      rectangle "==Outage grouping\\n\\nExpands a contingency into every element that goes out with it.\\nOff by default; then each contingency is its own group." <<ToopContingencyPpCaPpOutageGrouping>> as ToopContingencyPpCaPpOutageGrouping
      rectangle "==Slack allocation\\n\\nGives each surviving island its own slack bus, above a minimum\\nisland size. Without this an islanded contingency simply fails." <<ToopContingencyPpCaPpSlack>> as ToopContingencyPpCaPpSlack
      rectangle "==SpPS rule engine\\n\\nSpecial Protection Schemes as a condition/action rule engine.\\nA scheme whose conditions pass applies its actions to the network\\nand the loadflow re-runs, so the next iteration sees the new state." <<ToopContingencyPpCaPpSpps>> as ToopContingencyPpCaPpSpps
      rectangle "==Cascade simulation\\n\\nIterative follow-on outage simulation: overload and distance\\nprotection detection, outage grouping, re-solve, repeat. Produces\\nan event log rather than a single post-contingency state." <<ToopContingencyPpCaPpCascade>> as ToopContingencyPpCaPpCascade
    }
  }
  rectangle "Interfaces" <<ToopInterfaces>> as ToopInterfaces {
    skinparam RectangleBorderColor<<ToopInterfaces>> #64748b
    skinparam RectangleFontColor<<ToopInterfaces>> #64748b
    skinparam RectangleBorderStyle<<ToopInterfaces>> dashed

    rectangle "LoadflowResults" <<ToopInterfacesLfResults>> as ToopInterfacesLfResults {
      skinparam RectangleBorderColor<<ToopInterfacesLfResults>> #64748b
      skinparam RectangleFontColor<<ToopInterfacesLfResults>> #64748b
      skinparam RectangleBorderStyle<<ToopInterfacesLfResults>> dashed

      rectangle "==branch_results\\n\\nFlows and loading per monitored branch, per contingency and timestep." <<ToopInterfacesLfResultsBranchRes>> as ToopInterfacesLfResultsBranchRes
      rectangle "==node_results\\n\\nVoltage magnitude and angle per monitored node." <<ToopInterfacesLfResultsNodeRes>> as ToopInterfacesLfResultsNodeRes
      rectangle "==regulating_element_results\\n\\nTap positions and setpoints of regulating elements." <<ToopInterfacesLfResultsRegRes>> as ToopInterfacesLfResultsRegRes
      rectangle "==va_diff_results\\n\\nVoltage angle differences across the ends of an outaged branch and\\nacross open switches. What tells you whether a split can be closed\\nagain." <<ToopInterfacesLfResultsVaDiffRes>> as ToopInterfacesLfResultsVaDiffRes
      rectangle "==converged\\n\\nConvergence status per contingency and timestep. The index of what actually ran." <<ToopInterfacesLfResultsConvergedRes>> as ToopInterfacesLfResultsConvergedRes
      rectangle "==switch_results\\n\\nPower through each monitored switch, aggregated from everything connected to one side." <<ToopInterfacesLfResultsSwitchRes>> as ToopInterfacesLfResultsSwitchRes
      rectangle "==connectivity_result\\n\\nWhich elements each contingency takes out. Populated by outage grouping." <<ToopInterfacesLfResultsConnectivityRes>> as ToopInterfacesLfResultsConnectivityRes
      rectangle "==spps_results\\n\\nPer-case SpPS run summaries." <<ToopInterfacesLfResultsSppsRes>> as ToopInterfacesLfResultsSppsRes
      rectangle "==cascade_results\\n\\nOne row per cascade event. Empty when cascade screening is off." <<ToopInterfacesLfResultsCascadeRes>> as ToopInterfacesLfResultsCascadeRes
    }
  }
  rectangle "Importer" <<ToopImporter>> as ToopImporter {
    skinparam RectangleBorderColor<<ToopImporter>> #64748b
    skinparam RectangleFontColor<<ToopImporter>> #64748b
    skinparam RectangleBorderStyle<<ToopImporter>> dashed

    rectangle "==run_initial_loadflow\\n<size:10>[PyPowSyBl]</size>\\n\\nFull AC N-1 on the unmodified grid. Produces the reference metrics\\nevery proposed topology is later compared against." <<ToopImporterInitialLoadflow>> as ToopImporterInitialLoadflow
  }
  rectangle "AC-Validator" <<ToopAcValidator>> as ToopAcValidator {
    skinparam RectangleBorderColor<<ToopAcValidator>> #64748b
    skinparam RectangleFontColor<<ToopAcValidator>> #64748b
    skinparam RectangleBorderStyle<<ToopAcValidator>> dashed

    rectangle "==Worst-k epoch\\n<size:10>[PyPowSyBl]</size>\\n\\nReruns only the handful of contingencies the DC stage flagged as worst\\nfor this topology. A candidate that already fails there, or converges\\npoorly, is rejected without the full analysis. CPU parallelism is\\ndisabled here -- at this few cases it costs more than it saves." <<ToopAcValidatorWorstK>> as ToopAcValidatorWorstK
    rectangle "==Remaining contingencies\\n<size:10>[PyPowSyBl security analysis, multiprocess]</size>\\n\\nFull AC N-1 on the survivors, batched over runner processes. Hundreds\\nof contingencies rather than a handful." <<ToopAcValidatorRemainingCa>> as ToopAcValidatorRemainingCa
  }
}
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore

ToopContingencyPpCaPpOutageGrouping .[#8D8D8D,thickness=2].> ToopContingencyPpCaPpSlack : <color:#8D8D8D>grouped contingencies
ToopContingencyPpCaPpSlack .[#8D8D8D,thickness=2].> ToopContingencyPpCaPpSpps : <color:#8D8D8D>solvable islands
ToopContingencyPpCaPpSpps .[#8D8D8D,thickness=2].> ToopContingencyPpCaPpCascade : <color:#8D8D8D>post-scheme state
ToopImporterInitialLoadflow .[#8D8D8D,thickness=2].> ToopContingencyDispatcher : <color:#8D8D8D>base grid N-1
ToopAcValidatorWorstK .[#8D8D8D,thickness=2].> ToopContingencyDispatcher : <color:#8D8D8D>worst-k contingencies
ToopAcValidatorWorstK .[#8D8D8D,thickness=2].> ToopAcValidatorRemainingCa : <color:#8D8D8D>survivors
ToopAcValidatorRemainingCa .[#8D8D8D,thickness=2].> ToopContingencyDispatcher : <color:#8D8D8D>full N-1
ToopImporterInitialLoadflow .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 results
ToopContingencyDispatcher .[#8D8D8D,thickness=2].> ToopContingencyPpCa : <color:#8D8D8D>if pandapowerNet
ToopContingencyDispatcher .[#8D8D8D,thickness=2].> ToopContingencyPwCa : <color:#8D8D8D>if PyPowSyBl Network
ToopContingencyPpCa .[#8D8D8D,thickness=2].> ToopInterfacesLfResults : <color:#8D8D8D>fills all nine tables
ToopContingencyPwCa .[#8D8D8D,thickness=2].> ToopInterfacesLfResults : <color:#8D8D8D>fills the five common tables
ToopInterfacesLfResults .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>persisted per job
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>initial loadflow as baseline
@enduml
`;case`assetTopology`:return`@startuml
title "Asset topology -- master to runtime to simplified"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<ProcessedGridGridSnapshot>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageTopologyModelBusBreakerExtract>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageTopologyModelNodeBreakerExtract>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterImportStageTopologyModelPpExtract>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfacesAssetTopoMaster>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ProcessedGridAssetTopoMaster>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterDcPreprocessMaterializePwMaterialize>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterDcPreprocessMaterializeCompactMaterialize>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfacesAssetTopoRuntime>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopImporterDcPreprocessSimplifyPrepareSeparation>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterDcPreprocessSimplifyBbSimplify>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfacesAssetTopoSimplified>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterDcPreprocessElectricalActions>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterDcPreprocessStationRealisations>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterDcPreprocessBbOutage>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopInterfacesStoredActionSet>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "==grid.xiidm / grid.json\\n\\nThe normalized backend grid, written by the importer." <<ProcessedGridGridSnapshot>> as ProcessedGridGridSnapshot
rectangle "get_master_asset_topology_artifact" <<ToopImporterImportStageTopologyModel>> as ToopImporterImportStageTopologyModel {
  skinparam RectangleBorderColor<<ToopImporterImportStageTopologyModel>> #64748b
  skinparam RectangleFontColor<<ToopImporterImportStageTopologyModel>> #64748b
  skinparam RectangleBorderStyle<<ToopImporterImportStageTopologyModel>> dashed

  rectangle "==get_bus_breaker_master_asset_topology\\n\\nUCTE. Bus-breaker source, so bays and busbars have to be inferred rather than read." <<ToopImporterImportStageTopologyModelBusBreakerExtract>> as ToopImporterImportStageTopologyModelBusBreakerExtract
  rectangle "==get_node_breaker_master_asset_topology\\n\\nCGMES. Node-breaker source, walked as a station graph -- the richest input, and the one the model is shaped after." <<ToopImporterImportStageTopologyModelNodeBreakerExtract>> as ToopImporterImportStageTopologyModelNodeBreakerExtract
  rectangle "==get_master_asset_topology_from_network\\n\\npandapower nets, read through the pandapower switch and bus tables." <<ToopImporterImportStageTopologyModelPpExtract>> as ToopImporterImportStageTopologyModelPpExtract
}
rectangle "==1. MasterAssetTopology\\n\\nStructure, no state. Bus groups with their busbars, couplers, asset\\nbays and circuit groups, the branch and injection assets they\\nconnect, and branch_connectivity / injection_connectivity: every\\nbusbar assignment the station is physically wired for.\\n\\nRuntime state is stripped rather than merely missing -- every busbar\\nreads as in service and every coupler as closed, by construction.\\nNothing in here moves when a switch does, which is what makes it the\\none form worth writing to disk and reusing across grid states.\\n\\nThat independence is the design, not a side effect. Station wiring\\nchanges on an asset-management timescale; outages and injections\\nchange per case. Keeping the two in separate objects is what lets\\none master survive many cases -- and what would let the case a\\ntopology is materialized against come from somewhere other than the\\nfile the structure was read from." <<ToopInterfacesAssetTopoMaster>> as ToopInterfacesAssetTopoMaster
rectangle "==initial_topology/asset_topology_master_data.json\\n\\nA serialized MasterAssetTopology, and the only form of the asset\\ntopology that gets a file of its own. Written by the importer, read back\\nat the start of DC preprocessing. The runtime and simplified forms are\\nderived from it in memory and reach disk only inside action_set.json." <<ProcessedGridAssetTopoMaster>> as ProcessedGridAssetTopoMaster
rectangle "get_runtime_asset_topology" <<ToopImporterDcPreprocessMaterialize>> as ToopImporterDcPreprocessMaterialize {
  skinparam RectangleBorderColor<<ToopImporterDcPreprocessMaterialize>> #64748b
  skinparam RectangleFontColor<<ToopImporterDcPreprocessMaterialize>> #64748b
  skinparam RectangleBorderStyle<<ToopImporterDcPreprocessMaterialize>> dashed

  rectangle "==materialize_runtime_bus_groups_from_network_state\\n\\nReads switch positions straight off the node-breaker network, station by station." <<ToopImporterDcPreprocessMaterializePwMaterialize>> as ToopImporterDcPreprocessMaterializePwMaterialize
  rectangle "==materialize_runtime_bus_group_from_runtime_state\\n\\nThe backend-neutral half: a canonical bus group plus a compact\\nRuntimeSwitchingState overlay in, one runtime bus group out. The\\npandapower path runs through here, and so does the PyPowSyBl\\nfallback for legacy bay-less stations." <<ToopImporterDcPreprocessMaterializeCompactMaterialize>> as ToopImporterDcPreprocessMaterializeCompactMaterialize
}
rectangle "==2. RuntimeAssetTopology\\n\\nThe same structure, materialized against one grid state. What it\\nadds is a second pair of matrices: branch_switching_table and\\ninjection_switching_table say what is closed *now*, alongside the\\ninherited connectivity tables saying what *could* be closed. Plus\\nbusbar in_service, coupler open state, and the bus-branch bus id\\neach busbar currently belongs to.\\n\\nBoth matrices have to be present at once, and that is the point of\\nthis stage: enumerating a split needs to know what is reachable, and\\nscoring the switching distance needs to know where things stand.\\n\\nMaterialized at the start of DC preprocessing and not written out as\\na file of its own on the main path -- it lives in\\nNetworkData.asset_topology and travels onward inside the action set." <<ToopInterfacesAssetTopoRuntime>> as ToopInterfacesAssetTopoRuntime
rectangle "simplify_asset_topology" <<ToopImporterDcPreprocessSimplify>> as ToopImporterDcPreprocessSimplify {
  skinparam RectangleBorderColor<<ToopImporterDcPreprocessSimplify>> #64748b
  skinparam RectangleFontColor<<ToopImporterDcPreprocessSimplify>> #64748b
  skinparam RectangleBorderStyle<<ToopImporterDcPreprocessSimplify>> dashed

  rectangle "==prepare_for_separation_set\\n\\nWhere the reduction actually happens, one bus group at a time:\\norder assets to the solver index order, drop out-of-service assets\\nand disconnected busbars, remove duplicate open couplers, fuse\\ndisconnector couplers, and pick one bus for assets hanging off\\nseveral. Each step is best-effort and reports what it had to fix." <<ToopImporterDcPreprocessSimplifyPrepareSeparation>> as ToopImporterDcPreprocessSimplifyPrepareSeparation
  rectangle "==simplify_asset_topology_for_bb_outages\\n\\nThe second reduction, for busbar-outage preprocessing, run with\\ncouplers forced closed. Yields a separate simplified topology\\nrather than replacing the first one." <<ToopImporterDcPreprocessSimplifyBbSimplify>> as ToopImporterDcPreprocessSimplifyBbSimplify
}
rectangle "==3. SimplifiedAssetTopology\\n\\nThe runtime form reduced to what the DC solver can search -- and a\\n*subclass* of it, so the reduction is carried in the type system: a\\nfunction that needs a simplified bus group cannot be handed a raw\\none by accident. That is the whole reason the subtype exists; it\\nadds no fields.\\n\\nReduced per relevant station and per electrical node slice of it:\\nassets reordered into the solver index order, out-of-service assets\\nand disconnected busbars dropped, disconnector couplers fused,\\nduplicate open couplers removed, and one bus picked for assets that\\nhang off several. Stations that cannot be reduced are dropped from\\nthe relevant set outright, so this step also shrinks the search\\nspace.\\n\\nLossy on purpose, and therefore not interchangeable with the runtime\\nform: it is the geometry every enumerated action is indexed against,\\nwhich is why the action set has to store both." <<ToopInterfacesAssetTopoSimplified>> as ToopInterfacesAssetTopoSimplified
rectangle "==compute_electrical_actions\\n\\nStage one of action set enumeration: every electrically distinct\\ntwo-node split of a station, filtered for islanding and\\nconnectivity, clipped if a station exceeds the configured limit." <<ToopImporterDcPreprocessElectricalActions>> as ToopImporterDcPreprocessElectricalActions
rectangle "==enumerate_station_realisations\\n\\nStage two: map each electrical split onto a reachable node-breaker\\nrealization and precompute its reassignment distance. Splits with\\nno valid realization are discarded." <<ToopImporterDcPreprocessStationRealisations>> as ToopImporterDcPreprocessStationRealisations
rectangle "==preprocess_bb_outage\\n\\nOptional busbar outage contingencies, used by the do-not-make-it-worse criterion." <<ToopImporterDcPreprocessBbOutage>> as ToopImporterDcPreprocessBbOutage
rectangle "==Stored action set\\n\\nThe action set in physical terms, keyed to the asset topology, as\\nopposed to the electrical index form the JAX solver uses. Two\\nrepresentations of one thing: the JAX one is what gets searched, this\\none is what can be exported." <<ToopInterfacesStoredActionSet>> as ToopInterfacesStoredActionSet

ToopInterfacesAssetTopoMaster .[#8D8D8D,thickness=2].> ProcessedGridAssetTopoMaster : <color:#8D8D8D>serialized once per import
ToopInterfacesAssetTopoSimplified .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessElectricalActions : <color:#8D8D8D>the geometry splits are enumerated in
ToopInterfacesAssetTopoSimplified .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessStationRealisations : <color:#8D8D8D>station to realize a split against
ToopImporterDcPreprocessElectricalActions .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessStationRealisations : <color:#8D8D8D>electrical splits
ToopInterfacesAssetTopoSimplified .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessBbOutage : <color:#8D8D8D>reduced again with couplers closed
ToopImporterDcPreprocessStationRealisations .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessBbOutage : <color:#8D8D8D>action set A
ToopInterfacesAssetTopoRuntime .[#8D8D8D,thickness=2].> ToopInterfacesStoredActionSet : <color:#8D8D8D>starting_bus_groups -- to reach the real switches
ToopInterfacesAssetTopoSimplified .[#8D8D8D,thickness=2].> ToopInterfacesStoredActionSet : <color:#8D8D8D>simplified_starting_bus_groups -- the ordering local_actions is indexed against
ToopImporterDcPreprocessStationRealisations .[#8D8D8D,thickness=2].> ToopInterfacesStoredActionSet : <color:#8D8D8D>physical switchings per action
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopImporterImportStageTopologyModel : <color:#8D8D8D>normalized network
ToopImporterImportStageTopologyModel .[#8D8D8D,thickness=2].> ToopInterfacesAssetTopoMaster : <color:#8D8D8D>bus groups, bays, circuit groups, possible connectivity
ProcessedGridGridSnapshot .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessMaterialize : <color:#8D8D8D>live switch, coupler and busbar state
ProcessedGridAssetTopoMaster .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessMaterialize : <color:#8D8D8D>canonical structure
ToopImporterDcPreprocessMaterialize .[#8D8D8D,thickness=2].> ToopInterfacesAssetTopoRuntime : <color:#8D8D8D>structure + what is closed now
ToopInterfacesAssetTopoRuntime .[#8D8D8D,thickness=2].> ToopImporterDcPreprocessSimplify : <color:#8D8D8D>full physical bus groups
ToopImporterDcPreprocessSimplify .[#8D8D8D,thickness=2].> ToopInterfacesAssetTopoSimplified : <color:#8D8D8D>one reduced slice per electrical node
@enduml
`;case`loadflowFormat`:return`@startuml
title "Loadflow result store -- the on-disk format"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<LoadflowStoreLfMetadata>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<LoadflowStoreLfBranch>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<LoadflowStoreLfNode>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<LoadflowStoreLfConverged>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<LoadflowStoreLfVaDiff>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<LoadflowStoreLfReg>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<LoadflowStoreLfSwitch>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<LoadflowStoreLfSpps>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<LoadflowStoreLfCascade>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopInterfacesLfResults>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterInitialLoadflow>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidator>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "Loadflow result store" <<LoadflowStore>> as LoadflowStore {
  skinparam RectangleBorderColor<<LoadflowStore>> #64748b
  skinparam RectangleFontColor<<LoadflowStore>> #64748b
  skinparam RectangleBorderStyle<<LoadflowStore>> dashed

  rectangle "==metadata.json\\n\\nThe only non-Parquet file: job_id and the global warnings list. Written\\nfirst, so its presence marks the folder as started." <<LoadflowStoreLfMetadata>> as LoadflowStoreLfMetadata
  rectangle "==branch_results.parquet\\n\\nindex: timestep, contingency, element, side\\ncolumns: i, p, q, loading, element_name, contingency_name\\n\\nIndexed per branch *end*, so a branch appears twice per case. \`loading\`\\nis what the overload metrics are computed from." <<LoadflowStoreLfBranch>> as LoadflowStoreLfBranch
  rectangle "==node_results.parquet\\n\\nindex: timestep, contingency, element\\ncolumns: vm, vm_loading, va, p, q, vm_basecase_deviation, element_name, contingency_name" <<LoadflowStoreLfNode>> as LoadflowStoreLfNode
  rectangle "==converged.parquet\\n\\nindex: timestep, contingency\\ncolumns: status, iteration_count, warnings, contingency_name\\n\\nThe index of what actually ran. Read this first: non-converging cases\\nare missing from the other tables rather than NaN-filled." <<LoadflowStoreLfConverged>> as LoadflowStoreLfConverged
  rectangle "==va_diff_results.parquet\\n\\nindex: timestep, contingency, element\\ncolumns: va_diff, element_name, contingency_name" <<LoadflowStoreLfVaDiff>> as LoadflowStoreLfVaDiff
  rectangle "==regulating_element_results.parquet\\n\\nindex: timestep, contingency, element\\ncolumns: value, regulating_element_type, element_name, contingency_name" <<LoadflowStoreLfReg>> as LoadflowStoreLfReg
  rectangle "==switch_results.parquet\\n\\nindex: timestep, contingency, element\\ncolumns: p, q, vm, i, element_name, contingency_name, side\\n\\nOptional -- the file is absent unless the table was populated. The one\\ntable that keeps rows for non-converging cases and NaNs the values." <<LoadflowStoreLfSwitch>> as LoadflowStoreLfSwitch
  rectangle "==spps_results.parquet\\n\\nindex: timestep, contingency\\ncolumns: iterations, activated_schemes_per_iter, max_iterations_reached, power_flow_failed\\n\\nOptional." <<LoadflowStoreLfSpps>> as LoadflowStoreLfSpps
  rectangle "==cascade_results.parquet\\n\\nindex: timestep, contingency, cascade_number, element_mrid\\ncolumns: element_id, contingency_outage_id, element_outage_group_id,\\nelement_name, contingency_name, cascade_reason, loading, r_ohm, x_ohm,\\ndistance_protection_severity, activated_schemes_per_iter\\n\\nOptional. The only table with a per-event index rather than a\\nper-element one, because a cascade is a sequence." <<LoadflowStoreLfCascade>> as LoadflowStoreLfCascade
}
rectangle "==LoadflowResults\\n\\nOne container per computation job, holding a pandera-validated frame\\nper result family, plus warnings. Mirrored by LoadflowResultsPolars,\\nwhose schemas subclass the pandas ones so the column contracts cannot\\ndrift apart.\\n\\nConvention worth knowing: a non-converging case is *omitted* from the\\nresult tables rather than filled with NaN, and \`converged\` is what\\ntells you it happened. Switch results are the exception -- they keep\\nthe rows and set NaN." <<ToopInterfacesLfResults>> as ToopInterfacesLfResults
rectangle "==run_initial_loadflow\\n<size:10>[PyPowSyBl]</size>\\n\\nFull AC N-1 on the unmodified grid. Produces the reference metrics\\nevery proposed topology is later compared against." <<ToopImporterInitialLoadflow>> as ToopImporterInitialLoadflow
rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<ToopAcValidator>> as ToopAcValidator

ToopInterfacesLfResults .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>persisted per job
ToopImporterInitialLoadflow .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 results
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>initial loadflow as baseline
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
@enduml
`;case`index`:return`@startuml
title "System landscape"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Client>>{
  BackgroundColor #0284c7
  FontColor #f0f9ff
  BorderColor #0369a1
}
skinparam database<<UnprocessedGridStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopInterfaces>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam queue<<Kafka>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopPostprocess>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam database<<ProcessedGrid>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<Downstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopLfService>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterParams>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopDcParams>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcParams>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporter>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopDcOptimizer>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopAcValidator>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopContingency>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<Client>> as Client
database "==Unprocessed grid store\\n<size:10>[fsspec AbstractFileSystem (unprocessed_gridfile_fs)]</size>\\n\\nWhere the source grid files land before anything touches them. The same\\nkind of thing as the loadflow result store -- an fsspec filesystem the\\nworker is handed, local disk or object storage depending on how it is\\nwired -- not an external system the engine talks to.\\n\\nThe importer reads from it and never writes back. The concrete folder\\ncomes from the import command, as a path relative to this filesystem." <<UnprocessedGridStore>> as UnprocessedGridStore
rectangle "ToOp Engine" <<Toop>> as Toop {
  skinparam RectangleBorderColor<<Toop>> #3b82f6
  skinparam RectangleFontColor<<Toop>> #3b82f6
  skinparam RectangleBorderStyle<<Toop>> dashed

  rectangle "==Interfaces\\n<size:10>[toop_engine_interfaces]</size>\\n\\nThe shared vocabulary. Everything here exists so that two packages can\\nagree on a payload without depending on each other." <<ToopInterfaces>> as ToopInterfaces
  rectangle "==Postprocessing and export\\n<size:10>[toop_engine_dc_solver.postprocess, toop_engine_importer.exporter]</size>\\n\\nTurns an action index back into something a grid tool can open." <<ToopPostprocess>> as ToopPostprocess
  rectangle "==AC loadflow service\\n<size:10>[Python, PyPowSyBl]</size>\\n\\nA standalone N-1 service on its own loadflow_commands / loadflow_results\\n/ loadflow_heartbeat topics. Present in the codebase but off the main\\npath: dev-deployment does not create its topics, and the importer only\\nborrows a grid-loading helper from it." <<ToopLfService>> as ToopLfService
  rectangle "==Importer parameters\\n\\nCarried by the StartPreprocessingCommand. Fixes the scope of the whole\\nrun before any search happens: which grid, which area, which stations\\nmay be switched, which contingencies, and how hard to work at\\nenumerating the action space." <<ToopImporterParams>> as ToopImporterParams
  rectangle "==DC optimizer parameters\\n\\nCarried by the StartOptimizationCommand. Everything about how the\\nsearch behaves and what it optimizes for." <<ToopDcParams>> as ToopDcParams
  rectangle "==AC validator parameters\\n\\nCarried by the same StartOptimizationCommand as the DC parameters.\\nMostly about what to reject and how much compute to spend." <<ToopAcParams>> as ToopAcParams
  rectangle "==Importer\\n<size:10>[Python, PyPowSyBl, pandapower, JAX]</size>\\n\\nNormalizes a raw grid into a processed grid folder and derives the\\nsolver artifacts. Most of it depends only on the initial grid topology,\\nso it can run before the forecast is available. Runs one job at a time." <<ToopImporter>> as ToopImporter
  rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<ToopDcOptimizer>> as ToopDcOptimizer
  rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<ToopAcValidator>> as ToopAcValidator
  rectangle "==Contingency analysis\\n<size:10>[toop_engine_contingency_analysis]</size>\\n\\nRuns an N-1 analysis against whichever backend holds the grid, and\\nnormalizes both into the same result object. The two backends are not\\nat feature parity, so which one you import with decides what you can\\nmeasure afterwards." <<ToopContingency>> as ToopContingency
}
queue "==Kafka\\n<size:10>[confluent-kafka]</size>\\n\\nSix topics, created by dev-deployment/docker-compose.yaml. Every payload\\nis a pydantic model dumped to JSON and wrapped in a single protobuf\\nMessageWrapper whose only field is that JSON string -- the envelope is\\nprotobuf, the schema is pydantic." <<Kafka>> as Kafka
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore
database "==Processed grid folder\\n<size:10>[fsspec AbstractFileSystem]</size>\\n\\nOne folder per import job, shared by all three stages and the only large\\npayload that never travels through Kafka. fsspec keeps the backend open:\\nlocal disk in the dev setup, object storage or an NFS share in a\\ndeployment -- no bucket driver is a declared dependency, so whoever wires\\nup the worker chooses it." <<ProcessedGrid>> as ProcessedGrid
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<Downstream>> as Downstream

Client .[#8D8D8D,thickness=2].> Kafka : <color:#8D8D8D>[...]
Kafka .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>validated topologies for review
ProcessedGrid .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>UCTE, DGS, OpenRAO summaries, single line diagrams
Client .[#8D8D8D,thickness=2].> ToopImporterParams : <color:#8D8D8D>set in StartPreprocessingCommand
Client .[#8D8D8D,thickness=2].> ToopDcParams : <color:#8D8D8D>set in StartOptimizationCommand
Client .[#8D8D8D,thickness=2].> ToopAcParams : <color:#8D8D8D>set in the same StartOptimizationCommand
Kafka .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>consumes command
Kafka .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>consumes command
Kafka .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>[...]
UnprocessedGridStore .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>raw grid file
ProcessedGrid .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>[...]
ProcessedGrid .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>[...]
ProcessedGrid .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>[...]
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>initial loadflow as baseline
ToopImporter .[#8D8D8D,thickness=2].> Kafka : <color:#8D8D8D>[...]
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>[...]
ToopImporter .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 results
ToopDcOptimizer .[#8D8D8D,thickness=2].> Kafka : <color:#8D8D8D>[...]
ToopAcValidator .[#8D8D8D,thickness=2].> Kafka : <color:#8D8D8D>[...]
ToopAcValidator .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>summaries and diagrams
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
ToopInterfaces .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>serialized once per import
ToopInterfaces .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>persisted per job
ToopPostprocess .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>[...]
ToopImporter .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>base grid N-1
ToopImporter .[#8D8D8D,thickness=2].> ToopInterfaces : <color:#8D8D8D>[...]
ToopInterfaces .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>[...]
ToopImporterParams .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>scope, limits, contingencies
ToopDcParams .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>search bounds, fitness, operator probabilities
ToopAcValidator .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>[...]
ToopAcValidator .[#8D8D8D,thickness=2].> ToopInterfaces : <color:#8D8D8D>accepted topology
ToopAcParams .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>compute budget, pruning, rejection thresholds
ToopContingency .[#8D8D8D,thickness=2].> ToopInterfaces : <color:#8D8D8D>[...]
ToopInterfaces .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>monitored elements and contingencies
ToopPostprocess .[#8D8D8D,thickness=2].> ToopContingency : <color:#8D8D8D>grid with topology applied
ToopInterfaces .[#8D8D8D,thickness=2].> ToopPostprocess : <color:#8D8D8D>[...]
ToopPostprocess .[#8D8D8D,thickness=2].> ToopInterfaces : <color:#8D8D8D>switch id and new state
@enduml
`;case`overview`:return`@startuml
title "ToOp at a glance"
left to right direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Client>>{
  BackgroundColor #0284c7
  FontColor #f0f9ff
  BorderColor #0369a1
}
skinparam queue<<KafkaImporterCommands>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam database<<UnprocessedGridStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<ToopImporter>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<ProcessedGrid>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam database<<LoadflowStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam queue<<KafkaImporterResults>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam queue<<KafkaCommands>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<ToopDcOptimizer>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopAcValidator>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam queue<<KafkaResults>>{
  BackgroundColor #A35829
  FontColor #FFE0C2
  BorderColor #7E451D
}
skinparam rectangle<<Downstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<Client>> as Client
queue "==importer_commands\\n\\nStartPreprocessingCommand, ShutdownCommand. 24 partitions." <<KafkaImporterCommands>> as KafkaImporterCommands
database "==Unprocessed grid store\\n<size:10>[fsspec AbstractFileSystem (unprocessed_gridfile_fs)]</size>\\n\\nWhere the source grid files land before anything touches them. The same\\nkind of thing as the loadflow result store -- an fsspec filesystem the\\nworker is handed, local disk or object storage depending on how it is\\nwired -- not an external system the engine talks to.\\n\\nThe importer reads from it and never writes back. The concrete folder\\ncomes from the import command, as a path relative to this filesystem." <<UnprocessedGridStore>> as UnprocessedGridStore
rectangle "==Importer\\n<size:10>[Python, PyPowSyBl, pandapower, JAX]</size>\\n\\nNormalizes a raw grid into a processed grid folder and derives the\\nsolver artifacts. Most of it depends only on the initial grid topology,\\nso it can run before the forecast is available. Runs one job at a time." <<ToopImporter>> as ToopImporter
database "==Processed grid folder\\n<size:10>[fsspec AbstractFileSystem]</size>\\n\\nOne folder per import job, shared by all three stages and the only large\\npayload that never travels through Kafka. fsspec keeps the backend open:\\nlocal disk in the dev setup, object storage or an NFS share in a\\ndeployment -- no bucket driver is a declared dependency, so whoever wires\\nup the worker chooses it." <<ProcessedGrid>> as ProcessedGrid
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the initial N-1 on the unmodified grid, which the\\nvalidator reads back as its baseline -- and computes and stores itself if\\nno reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<LoadflowStore>> as LoadflowStore
queue "==importer_results\\n\\nPreprocessingStartedResult, PreprocessingSuccessResult, ErrorResult" <<KafkaImporterResults>> as KafkaImporterResults
queue "==commands\\n\\nStartOptimizationCommand, ShutdownCommand. 4 partitions." <<KafkaCommands>> as KafkaCommands
rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<ToopDcOptimizer>> as ToopDcOptimizer
rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<ToopAcValidator>> as ToopAcValidator
queue "==results\\n\\nThe one shared topic. Both stages publish topologies here and the\\nAC-Validator also consumes it to pick up DC candidates." <<KafkaResults>> as KafkaResults
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<Downstream>> as Downstream

Client .[#8D8D8D,thickness=2].> KafkaImporterCommands : <color:#8D8D8D>StartPreprocessingCommand
KafkaImporterCommands .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>picks up the job
UnprocessedGridStore .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>UCTE / CGMES / PowerFactory file
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>normalized snapshot, masks, asset topology
ToopImporter .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>PTDF, action set, contingency set
ToopImporter .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>initial AC N-1 and reference metrics
ToopImporter .[#8D8D8D,thickness=2].> KafkaImporterResults : <color:#8D8D8D>PreprocessingSuccessResult
KafkaImporterResults .[#8D8D8D,thickness=2].> Client : <color:#8D8D8D>data folder is ready
Client .[#8D8D8D,thickness=2].> KafkaCommands : <color:#8D8D8D>StartOptimizationCommand
KafkaCommands .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>starts the DC run
KafkaCommands .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>starts the AC run
ToopDcOptimizer .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>loads static information onto the GPU
ToopAcValidator .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>loads base grid and action set
LoadflowStore .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>reads the initial loadflow as baseline
ToopDcOptimizer .[#8D8D8D,thickness=2].> KafkaResults : <color:#8D8D8D>Strategy, once per epoch
KafkaResults .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>DC topologies to validate
ToopAcValidator .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>prune, worst-k, then full N-1
ToopAcValidator .[#8D8D8D,thickness=2].> LoadflowStore : <color:#8D8D8D>AC loadflow results per evaluated topology
ToopAcValidator .[#8D8D8D,thickness=2].> KafkaResults : <color:#8D8D8D>AC-validated Strategy, referencing its loadflow
ToopAcValidator .[#8D8D8D,thickness=2].> ProcessedGrid : <color:#8D8D8D>summaries, diagrams, loadflow tables
KafkaResults .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>topologies for review
ProcessedGrid .[#8D8D8D,thickness=2].> Downstream : <color:#8D8D8D>UCTE, DGS, OpenRAO summaries, single line diagrams
@enduml
`;case`parameters`:return`@startuml
title "Parameters -- where the constraints live"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Client>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopImporterParamsPAreaSettings>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterParamsPStationRules>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterParamsPLists>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterParamsPContingencies>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopImporterParamsPPreprocess>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<ToopDcParamsPMe>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopDcParamsPSolver>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopDcParamsPDoubleLimits>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<ToopImporter>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopDcOptimizer>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcValidator>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<ToopAcParamsPAcGa>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<ToopAcParamsPRejection>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<ToopAcParamsPInitialLoadflow>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<Client>> as Client
rectangle "ToOp Engine" <<Toop>> as Toop {
  skinparam RectangleBorderColor<<Toop>> #64748b
  skinparam RectangleFontColor<<Toop>> #64748b
  skinparam RectangleBorderStyle<<Toop>> dashed

  rectangle "Importer parameters" <<ToopImporterParams>> as ToopImporterParams {
    skinparam RectangleBorderColor<<ToopImporterParams>> #3b82f6
    skinparam RectangleFontColor<<ToopImporterParams>> #3b82f6
    skinparam RectangleBorderStyle<<ToopImporterParams>> dashed

    rectangle "==AreaSettings\\n\\ncontrol_area and view_area, plus cutoff_voltage (220 kV by default).\\n\\nAlso where the limit adjustments live: dso_trafo_factors and\\nborder_line_factors, each a LimitAdjustmentParameters, with a\\ndso_trafo_weight and border_line_weight beside them. Those factors\\nare the artificial limits, and those weights are the branch weights\\nthe fitness later uses -- they are set here, not in the optimizer." <<ToopImporterParamsPAreaSettings>> as ToopImporterParamsPAreaSettings
    rectangle "==RelevantStationRules\\n\\nWhat makes a station worth switching: min_busbars (2),\\nmin_connected_branches (4), min_connected_elements (4). This decides\\nthe set of switchable substations." <<ToopImporterParamsPStationRules>> as ToopImporterParamsPStationRules
    rectangle "==White / black / ignore lists\\n\\nwhite_list_file, black_list_file, ignore_list_file and\\nselect_by_voltage_level_id_list. Operator overrides on top of the\\narea rules, applied during convert_file." <<ToopImporterParamsPLists>> as ToopImporterParamsPLists
    rectangle "==Contingency list\\n\\ncontingency_list_file plus its schema_format, either the PowerFactory\\nimport schema or the generic one. Becomes the N-1 definition. Without\\nit, contingencies are derived from the network masks instead." <<ToopImporterParamsPContingencies>> as ToopImporterParamsPContingencies
    rectangle "==PreprocessParameters\\n\\nHow hard to work on the action space. action_set_clip caps a station\\nat 2^23 configurations; action_set_filter_bridge_lookup and\\naction_set_filter_bsdf_lodf drop splits that island the grid;\\nelectrical_reassignment_limits caps reassignments per station;\\nseparation_set_clip_hamming_distance and _clip_at_size bound the\\nseparation set; realise_station_busbar_choice_heuristic picks which\\nnode-breaker realization a split gets." <<ToopImporterParamsPPreprocess>> as ToopImporterParamsPPreprocess
  }
  rectangle "DC optimizer parameters" <<ToopDcParams>> as ToopDcParams {
    skinparam RectangleBorderColor<<ToopDcParams>> #6366f1
    skinparam RectangleFontColor<<ToopDcParams>> #6366f1
    skinparam RectangleBorderStyle<<ToopDcParams>> dashed

    rectangle "==BatchedMEParameters (ga_config)\\n\\nThe search itself. runtime_seconds and iterations_per_epoch set the\\nbudget and how often results are pushed. target_metrics and\\nobserved_metrics define the fitness -- overload_energy_n_1 weighted\\n1.0 by default. me_descriptors and cell_depth define the repertoire\\ncells. The mutation and crossover probabilities live here too:\\nn_subs_mutated_lambda, add/change/remove split and disconnection\\nprobabilities, the PST mutation sigma, proportion_crossover.\\n\\nn_worst_contingencies (20) is the handoff to AC -- it decides how\\nmany worst cases the AC-Validator re-checks first. enable_bb_outage\\nand its penalties turn on busbar outage scoring." <<ToopDcParamsPMe>> as ToopDcParamsPMe
    rectangle "==LoadflowSolverParameters\\n\\nThe shape of the search space and the batch. max_num_splits (4) and\\nmax_num_disconnections cap the genome; batch_size sets how many\\ntopologies the GPU evaluates at once; distributed spreads over\\ndevices; cross_coupler_flow turns on the cross-coupler computation\\nand its penalty." <<ToopDcParamsPSolver>> as ToopDcParamsPSolver
    rectangle "==DoubleLimitsSetpoint\\n\\nOptional. Separate permanent and temporary branch limits, so N-0 and\\nN-1 can be judged against different ratings." <<ToopDcParamsPDoubleLimits>> as ToopDcParamsPDoubleLimits
  }
  rectangle "AC validator parameters" <<ToopAcParams>> as ToopAcParams {
    skinparam RectangleBorderColor<<ToopAcParams>> #AC4D39
    skinparam RectangleFontColor<<ToopAcParams>> #AC4D39
    skinparam RectangleBorderStyle<<ToopAcParams>> dashed

    rectangle "==ACGAParameters (ga_config)\\n\\nruntime_seconds (180) and max_initial_wait_seconds bound the run;\\nrunner_processes, contingency_processes and their worst_k\\ncounterparts set the CPU parallelism, separately for the worst-k pass\\nand the full analysis, because parallelism does not pay off at 20\\ncontingencies.\\n\\nfilter_strategy configures the discriminator / dominator / median\\npruning. remaining_loadflow_wait_seconds decides when to run a\\npartial batch rather than wait for a full one." <<ToopAcParamsPAcGa>> as ToopAcParamsPAcGa
    rectangle "==Rejection thresholds\\n\\nWhat counts as a failure. enable_ac_rejection switches the gate on,\\nthen reject_overload_threshold (0.95), reject_critical_branch_threshold\\n(1.1), reject_convergence_threshold, reject_voltage_jump_threshold and\\nreject_critical_va_diff_threshold decide against what.\\n\\nThe voltage criteria are the ones DC cannot see at all:\\ncritical_voltage_jump_percent (5%) and critical_va_diff_degree (20\\ndegrees), behind enable_critical_voltage_rejection." <<ToopAcParamsPRejection>> as ToopAcParamsPRejection
    rectangle "==initial_loadflow reference\\n\\nAn optional StoredLoadflowReference to the baseline. If absent the\\nvalidator computes and stores the initial AC N-1 itself." <<ToopAcParamsPInitialLoadflow>> as ToopAcParamsPInitialLoadflow
  }
  rectangle "==Importer\\n<size:10>[Python, PyPowSyBl, pandapower, JAX]</size>\\n\\nNormalizes a raw grid into a processed grid folder and derives the\\nsolver artifacts. Most of it depends only on the initial grid topology,\\nso it can run before the forecast is available. Runs one job at a time." <<ToopImporter>> as ToopImporter
  rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<ToopDcOptimizer>> as ToopDcOptimizer
  rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<ToopAcValidator>> as ToopAcValidator
}

Client .[#8D8D8D,thickness=2].> ToopImporterParams : <color:#8D8D8D>set in StartPreprocessingCommand
ToopImporterParams .[#8D8D8D,thickness=2].> ToopImporter : <color:#8D8D8D>scope, limits, contingencies
Client .[#8D8D8D,thickness=2].> ToopDcParams : <color:#8D8D8D>set in StartOptimizationCommand
ToopDcParams .[#8D8D8D,thickness=2].> ToopDcOptimizer : <color:#8D8D8D>search bounds, fitness, operator probabilities
Client .[#8D8D8D,thickness=2].> ToopAcParams : <color:#8D8D8D>set in the same StartOptimizationCommand
ToopAcParams .[#8D8D8D,thickness=2].> ToopAcValidator : <color:#8D8D8D>compute budget, pruning, rejection thresholds
@enduml
`;default:throw Error(`Unknown viewId: `+e)}};export{e as pumlSource};