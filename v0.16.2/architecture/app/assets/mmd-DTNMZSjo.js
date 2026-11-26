var e=e=>{switch(e){case`index`:return`---
title: "Landscape view"
---
graph TB
  OptimizerDb@{ shape: disk, label: "Optimizer database" }
  AcFastFail@{ shape: rectangle, label: "AC fast-failing stage" }
  @toopToop@{ shape: rectangle, label: "ToOp Engine" }
  @toopDownstream@{ shape: rectangle, label: "Frontend / downstream systems" }
  @toopClient@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }
  OptimizerDb -. "\`open work items to select from and claim\`" .-> AcFastFail
  OptimizerDb -. "\`open work items to claim\`" .-> @toopToop
  OptimizerDb -. "\`accepted topologies for review\`" .-> @toopDownstream
  OptimizerDb -. "\`[...]\`" .-> @toopClient
  AcFastFail -. "\`[...]\`" .-> OptimizerDb
  @toopToop -. "\`[...]\`" .-> OptimizerDb
  @toopClient -. "\`[...]\`" .-> OptimizerDb
`;case`postgresCommands`:return'---\ntitle: "Postgres instead of Kafka"\n---\ngraph LR\n  subgraph OptimizerDb["`Optimizer database`"]\n    subgraph OptimizerDb.ResultTables["`Result tables`"]\n      OptimizerDb.ResultTables.Strategy@{ shape: doc, label: "strategy" }\n      OptimizerDb.ResultTables.Topology@{ shape: doc, label: "topology" }\n      OptimizerDb.ResultTables.TopologyWorkItem@{ shape: doc, label: "topology_work_item" }\n      OptimizerDb.ResultTables.StageTopologyResult@{ shape: doc, label: "stage_topology_result" }\n    end\n    subgraph OptimizerDb.CommandTables["`Command tables`"]\n      OptimizerDb.CommandTables.ActiveWorker@{ shape: doc, label: "active_worker" }\n      OptimizerDb.CommandTables.OptimizationJob@{ shape: doc, label: "optimization_job" }\n      OptimizerDb.CommandTables.StageWorkItem@{ shape: doc, label: "stage_work_item" }\n      OptimizerDb.CommandTables.StageExecutionHistory@{ shape: doc, label: "stage_execution_history" }\n    end\n  end\n  subgraph @toopToop["`ToOp Engine`"]\n    @toopToop.DcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }\n    @toopToop.AcValidator@{ shape: rectangle, label: "AC-Validator" }\n  end\n  @toopClient@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }\n  AcFastFail@{ shape: rectangle, label: "AC fast-failing stage" }\n  @toopDownstream@{ shape: rectangle, label: "Frontend / downstream systems" }\n  @toopClient -. "`start_optimization: insert the request`" .-> OptimizerDb.CommandTables.OptimizationJob\n  @toopClient -. "`one TRIGGERED row per stage; cancel_optimization sets CANCELLED`" .-> OptimizerDb.CommandTables.StageWorkItem\n  OptimizerDb.CommandTables.StageWorkItem -. "`job status inferred from the work items`" .-> @toopClient\n  OptimizerDb.CommandTables.ActiveWorker -. "`which workers are alive`" .-> @toopClient\n  @toopToop.DcOptimizer -. "`claims the DC stage with SKIP LOCKED, refreshes the lease every epoch`" .-> OptimizerDb.CommandTables.StageWorkItem\n  @toopToop.DcOptimizer -. "`attempt row, KPIs per epoch`" .-> OptimizerDb.CommandTables.StageExecutionHistory\n  @toopToop.DcOptimizer -. "`heartbeat every few seconds`" .-> OptimizerDb.CommandTables.ActiveWorker\n  @toopToop.DcOptimizer -. "`new elites, immutable`" .-> OptimizerDb.ResultTables.Topology\n  @toopToop.DcOptimizer -. "`work item for AC_FAST_FAILING`" .-> OptimizerDb.ResultTables.TopologyWorkItem\n  @toopToop.DcOptimizer -. "`DC fitness, metrics, worst-k cases`" .-> OptimizerDb.ResultTables.StageTopologyResult\n  OptimizerDb.ResultTables.TopologyWorkItem -. "`open work items to select from and claim`" .-> AcFastFail\n  AcFastFail -. "`reads the shared state of the stage`" .-> OptimizerDb.CommandTables.StageWorkItem\n  AcFastFail -. "`attempt row, KPIs`" .-> OptimizerDb.CommandTables.StageExecutionHistory\n  AcFastFail -. "`heartbeat every few seconds, the job it is on`" .-> OptimizerDb.CommandTables.ActiveWorker\n  AcFastFail -. "`claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN`" .-> OptimizerDb.ResultTables.TopologyWorkItem\n  AcFastFail -. "`worst-k AC result`" .-> OptimizerDb.ResultTables.StageTopologyResult\n  OptimizerDb.ResultTables.TopologyWorkItem -. "`open work items to claim`" .-> @toopToop.AcValidator\n  @toopToop.AcValidator -. "`reads the shared state of the stage`" .-> OptimizerDb.CommandTables.StageWorkItem\n  @toopToop.AcValidator -. "`attempt row, KPIs`" .-> OptimizerDb.CommandTables.StageExecutionHistory\n  @toopToop.AcValidator -. "`heartbeat every few seconds, the job it is on`" .-> OptimizerDb.CommandTables.ActiveWorker\n  @toopToop.AcValidator -. "`claims items; deletes its item with the result`" .-> OptimizerDb.ResultTables.TopologyWorkItem\n  @toopToop.AcValidator -. "`full AC result with loadflow reference`" .-> OptimizerDb.ResultTables.StageTopologyResult\n  OptimizerDb.ResultTables.StageTopologyResult -. "`accepted topologies for review`" .-> @toopDownstream\n';case`postgresCommandsTables`:return`---
title: "Tables of the optimizer database"
---
graph TB
  subgraph OptimizerDb["\`Optimizer database\`"]
    subgraph OptimizerDb.ResultTables["\`Result tables\`"]
      OptimizerDb.ResultTables.TopologyWorkItem@{ shape: doc, label: "topology_work_item" }
      OptimizerDb.ResultTables.StageTopologyResult@{ shape: doc, label: "stage_topology_result" }
      OptimizerDb.ResultTables.Topology@{ shape: doc, label: "topology" }
      OptimizerDb.ResultTables.Strategy@{ shape: doc, label: "strategy" }
    end
    subgraph OptimizerDb.CommandTables["\`Command tables\`"]
      OptimizerDb.CommandTables.StageExecutionHistory@{ shape: doc, label: "stage_execution_history" }
      OptimizerDb.CommandTables.StageWorkItem@{ shape: doc, label: "stage_work_item" }
      OptimizerDb.CommandTables.ActiveWorker@{ shape: doc, label: "active_worker" }
      OptimizerDb.CommandTables.OptimizationJob@{ shape: doc, label: "optimization_job" }
    end
  end
  OptimizerDb.CommandTables.StageWorkItem -. "\`job, one row per stage\`" .-> OptimizerDb.CommandTables.OptimizationJob
  OptimizerDb.CommandTables.ActiveWorker -. "\`job it is on\`" .-> OptimizerDb.CommandTables.OptimizationJob
  OptimizerDb.CommandTables.StageExecutionHistory -. "\`one row per attempt\`" .-> OptimizerDb.CommandTables.StageWorkItem
  OptimizerDb.CommandTables.StageExecutionHistory -. "\`worker of the attempt\`" .-> OptimizerDb.CommandTables.ActiveWorker
  OptimizerDb.ResultTables.Topology -. "\`job, unique with the hash\`" .-> OptimizerDb.CommandTables.OptimizationJob
  OptimizerDb.ResultTables.TopologyWorkItem -. "\`claimed by, open again once its heartbeat is older than the timeout\`" .-> OptimizerDb.CommandTables.ActiveWorker
  OptimizerDb.ResultTables.Topology -. "\`grouped into\`" .-> OptimizerDb.ResultTables.Strategy
  OptimizerDb.ResultTables.TopologyWorkItem -. "\`topology to evaluate, per stage\`" .-> OptimizerDb.ResultTables.Topology
  OptimizerDb.ResultTables.StageTopologyResult -. "\`result of, per stage\`" .-> OptimizerDb.ResultTables.Topology
`;case`postgresCommandsLiveness`:return`---
title: "Worker liveness and claims"
---
graph TB
  subgraph @toopToop["\`ToOp Engine\`"]
    @toopToop.DcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }
    @toopToop.AcValidator@{ shape: rectangle, label: "AC-Validator" }
  end
  AcFastFail@{ shape: rectangle, label: "AC fast-failing stage" }
  subgraph OptimizerDb["\`Optimizer database\`"]
    subgraph OptimizerDb.ResultTables["\`Result tables\`"]
      OptimizerDb.ResultTables.TopologyWorkItem@{ shape: doc, label: "topology_work_item" }
    end
    subgraph OptimizerDb.CommandTables["\`Command tables\`"]
      OptimizerDb.CommandTables.StageWorkItem@{ shape: doc, label: "stage_work_item" }
      OptimizerDb.CommandTables.ActiveWorker@{ shape: doc, label: "active_worker" }
      OptimizerDb.CommandTables.OptimizationJob@{ shape: doc, label: "optimization_job" }
    end
  end
  OptimizerDb.CommandTables.StageWorkItem -. "\`job, one row per stage\`" .-> OptimizerDb.CommandTables.OptimizationJob
  OptimizerDb.CommandTables.ActiveWorker -. "\`job it is on\`" .-> OptimizerDb.CommandTables.OptimizationJob
  OptimizerDb.ResultTables.TopologyWorkItem -. "\`claimed by, open again once its heartbeat is older than the timeout\`" .-> OptimizerDb.CommandTables.ActiveWorker
  @toopToop.DcOptimizer -. "\`claims the DC stage with SKIP LOCKED, refreshes the lease every epoch\`" .-> OptimizerDb.CommandTables.StageWorkItem
  @toopToop.DcOptimizer -. "\`heartbeat every few seconds\`" .-> OptimizerDb.CommandTables.ActiveWorker
  AcFastFail -. "\`reads the shared state of the stage\`" .-> OptimizerDb.CommandTables.StageWorkItem
  AcFastFail -. "\`heartbeat every few seconds, the job it is on\`" .-> OptimizerDb.CommandTables.ActiveWorker
  AcFastFail -. "\`claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN\`" .-> OptimizerDb.ResultTables.TopologyWorkItem
  @toopToop.AcValidator -. "\`reads the shared state of the stage\`" .-> OptimizerDb.CommandTables.StageWorkItem
  @toopToop.AcValidator -. "\`heartbeat every few seconds, the job it is on\`" .-> OptimizerDb.CommandTables.ActiveWorker
  @toopToop.AcValidator -. "\`claims items; deletes its item with the result\`" .-> OptimizerDb.ResultTables.TopologyWorkItem
`;case`postgresCommandsLifecycle`:return'---\ntitle: "One job on the database"\n---\ngraph LR\n  @toopClient@{ icon: "fa:user", shape: rounded, label: "Operator / orchestration client" }\n  OptimizerDbCommandTablesOptimizationJob@{ shape: doc, label: "optimization_job" }\n  @toopToopDcOptimizer@{ shape: rectangle, label: "DC-Optimizer" }\n  AcFastFail@{ shape: rectangle, label: "AC fast-failing stage" }\n  @toopToopAcValidator@{ shape: rectangle, label: "AC-Validator" }\n  OptimizerDbCommandTablesStageWorkItem@{ shape: doc, label: "stage_work_item" }\n  OptimizerDbCommandTablesStageExecutionHistory@{ shape: doc, label: "stage_execution_history" }\n  OptimizerDbResultTablesTopology@{ shape: doc, label: "topology" }\n  OptimizerDbResultTablesStageTopologyResult@{ shape: doc, label: "stage_topology_result" }\n  OptimizerDbResultTablesTopologyWorkItem@{ shape: doc, label: "topology_work_item" }\n  @toopLoadflowStore@{ shape: disk, label: "Loadflow result store" }\n  @toopDownstream@{ shape: rectangle, label: "Frontend / downstream systems" }\n  @toopClient -. "`start_optimization: the request, write-once`" .-> OptimizerDbCommandTablesOptimizationJob\n  @toopClient -. "`one work item per stage, TRIGGERED`" .-> OptimizerDbCommandTablesStageWorkItem\n  @toopToopDcOptimizer -. "`claims DC: RUNNING, attempt 1, lease`" .-> OptimizerDbCommandTablesStageWorkItem\n  AcFastFail -. "`claims AC_FAST_FAILING`" .-> OptimizerDbCommandTablesStageWorkItem\n  @toopToopAcValidator -. "`claims AC`" .-> OptimizerDbCommandTablesStageWorkItem\n  @toopToopDcOptimizer -. "`attempt row with worker_id and claimed_at`" .-> OptimizerDbCommandTablesStageExecutionHistory\n  @toopToopDcOptimizer -. "`every epoch: refresh lease, check for CANCELLED`" .-> OptimizerDbCommandTablesStageWorkItem\n  @toopToopDcOptimizer -. "`every epoch: KPIs`" .-> OptimizerDbCommandTablesStageExecutionHistory\n  @toopToopDcOptimizer -. "`new elites as immutable rows`" .-> OptimizerDbResultTablesTopology\n  @toopToopDcOptimizer -. "`DC fitness and worst-k cases`" .-> OptimizerDbResultTablesStageTopologyResult\n  @toopToopDcOptimizer -. "`work item for AC_FAST_FAILING`" .-> OptimizerDbResultTablesTopologyWorkItem\n  OptimizerDbResultTablesTopologyWorkItem -. "`selects among the open items and claims them`" .-> AcFastFail\n  AcFastFail -. "`worst-k AC result`" .-> OptimizerDbResultTablesStageTopologyResult\n  AcFastFail -. "`deletes its item; ACCEPTED or WARN: item for AC`" .-> OptimizerDbResultTablesTopologyWorkItem\n  OptimizerDbResultTablesTopologyWorkItem -. "`claims an open item`" .-> @toopToopAcValidator\n  @toopToopAcValidator -. "`full AC N-1 per topology`" .-> @toopLoadflowStore\n  @toopToopAcValidator -. "`fitness, metrics, loadflow reference`" .-> OptimizerDbResultTablesStageTopologyResult\n  @toopToopAcValidator -. "`deletes its item`" .-> OptimizerDbResultTablesTopologyWorkItem\n  @toopToopDcOptimizer -. "`COMPLETED, lease cleared`" .-> OptimizerDbCommandTablesStageWorkItem\n  AcFastFail -. "`COMPLETED`" .-> OptimizerDbCommandTablesStageWorkItem\n  @toopToopAcValidator -. "`COMPLETED`" .-> OptimizerDbCommandTablesStageWorkItem\n  OptimizerDbCommandTablesStageWorkItem -. "`all stages COMPLETED: job is done`" .-> @toopClient\n  OptimizerDbResultTablesStageTopologyResult -. "`accepted topologies for review`" .-> @toopDownstream\n';default:throw Error(`Unknown viewId: `+e)}};export{e as mmdSource};