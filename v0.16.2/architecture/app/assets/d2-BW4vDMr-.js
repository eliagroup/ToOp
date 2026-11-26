var e=e=>{switch(e){case`index`:return`direction: down

OptimizerDb: {
  label: "Optimizer database"
  shape: stored_data
}
AcFastFail: {
  label: "AC fast-failing stage"
}
@toopToop: {
  label: "ToOp Engine"
}
@toopDownstream: {
  label: "Frontend / downstream systems"
}
@toopClient: {
  label: "Operator / orchestration client"
  shape: c4-person
}

OptimizerDb -> AcFastFail: "open work items to select from and claim"
OptimizerDb -> @toopToop: "open work items to claim"
OptimizerDb -> @toopDownstream: "accepted topologies for review"
OptimizerDb -> @toopClient: "[...]"
AcFastFail -> OptimizerDb: "[...]"
@toopToop -> OptimizerDb: "[...]"
@toopClient -> OptimizerDb: "[...]"
`;case`postgresCommands`:return`direction: right

OptimizerDb: {
  label: "Optimizer database"
  shape: stored_data

  ResultTables: {
    label: "Result tables"

    Strategy: {
      label: "strategy"
      shape: document
    }
    Topology: {
      label: "topology"
      shape: document
    }
    TopologyWorkItem: {
      label: "topology_work_item"
      shape: document
    }
    StageTopologyResult: {
      label: "stage_topology_result"
      shape: document
    }
  }
  CommandTables: {
    label: "Command tables"

    ActiveWorker: {
      label: "active_worker"
      shape: document
    }
    OptimizationJob: {
      label: "optimization_job"
      shape: document
    }
    StageWorkItem: {
      label: "stage_work_item"
      shape: document
    }
    StageExecutionHistory: {
      label: "stage_execution_history"
      shape: document
    }
  }
}
@toopToop: {
  label: "ToOp Engine"

  DcOptimizer: {
    label: "DC-Optimizer"
  }
  AcValidator: {
    label: "AC-Validator"
  }
}
@toopClient: {
  label: "Operator / orchestration client"
  shape: c4-person
}
AcFastFail: {
  label: "AC fast-failing stage"
}
@toopDownstream: {
  label: "Frontend / downstream systems"
}

@toopClient -> OptimizerDb.CommandTables.OptimizationJob: "start_optimization: insert the request"
@toopClient -> OptimizerDb.CommandTables.StageWorkItem: "one TRIGGERED row per stage; cancel_optimization sets CANCELLED"
OptimizerDb.CommandTables.StageWorkItem -> @toopClient: "job status inferred from the work items"
OptimizerDb.CommandTables.ActiveWorker -> @toopClient: "which workers are alive"
@toopToop.DcOptimizer -> OptimizerDb.CommandTables.StageWorkItem: "claims the DC stage with SKIP LOCKED, refreshes the lease every epoch"
@toopToop.DcOptimizer -> OptimizerDb.CommandTables.StageExecutionHistory: "attempt row, KPIs per epoch"
@toopToop.DcOptimizer -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds"
@toopToop.DcOptimizer -> OptimizerDb.ResultTables.Topology: "new elites, immutable"
@toopToop.DcOptimizer -> OptimizerDb.ResultTables.TopologyWorkItem: "work item for AC_FAST_FAILING"
@toopToop.DcOptimizer -> OptimizerDb.ResultTables.StageTopologyResult: "DC fitness, metrics, worst-k cases"
OptimizerDb.ResultTables.TopologyWorkItem -> AcFastFail: "open work items to select from and claim"
AcFastFail -> OptimizerDb.CommandTables.StageWorkItem: "reads the shared state of the stage"
AcFastFail -> OptimizerDb.CommandTables.StageExecutionHistory: "attempt row, KPIs"
AcFastFail -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds, the job it is on"
AcFastFail -> OptimizerDb.ResultTables.TopologyWorkItem: "claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN"
AcFastFail -> OptimizerDb.ResultTables.StageTopologyResult: "worst-k AC result"
OptimizerDb.ResultTables.TopologyWorkItem -> @toopToop.AcValidator: "open work items to claim"
@toopToop.AcValidator -> OptimizerDb.CommandTables.StageWorkItem: "reads the shared state of the stage"
@toopToop.AcValidator -> OptimizerDb.CommandTables.StageExecutionHistory: "attempt row, KPIs"
@toopToop.AcValidator -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds, the job it is on"
@toopToop.AcValidator -> OptimizerDb.ResultTables.TopologyWorkItem: "claims items; deletes its item with the result"
@toopToop.AcValidator -> OptimizerDb.ResultTables.StageTopologyResult: "full AC result with loadflow reference"
OptimizerDb.ResultTables.StageTopologyResult -> @toopDownstream: "accepted topologies for review"
`;case`postgresCommandsTables`:return`direction: down

OptimizerDb: {
  label: "Optimizer database"
  shape: stored_data

  ResultTables: {
    label: "Result tables"

    TopologyWorkItem: {
      label: "topology_work_item"
      shape: document
    }
    StageTopologyResult: {
      label: "stage_topology_result"
      shape: document
    }
    Topology: {
      label: "topology"
      shape: document
    }
    Strategy: {
      label: "strategy"
      shape: document
    }
  }
  CommandTables: {
    label: "Command tables"

    StageExecutionHistory: {
      label: "stage_execution_history"
      shape: document
    }
    StageWorkItem: {
      label: "stage_work_item"
      shape: document
    }
    ActiveWorker: {
      label: "active_worker"
      shape: document
    }
    OptimizationJob: {
      label: "optimization_job"
      shape: document
    }
  }
}

OptimizerDb.CommandTables.StageWorkItem -> OptimizerDb.CommandTables.OptimizationJob: "job, one row per stage"
OptimizerDb.CommandTables.ActiveWorker -> OptimizerDb.CommandTables.OptimizationJob: "job it is on"
OptimizerDb.CommandTables.StageExecutionHistory -> OptimizerDb.CommandTables.StageWorkItem: "one row per attempt"
OptimizerDb.CommandTables.StageExecutionHistory -> OptimizerDb.CommandTables.ActiveWorker: "worker of the attempt"
OptimizerDb.ResultTables.Topology -> OptimizerDb.CommandTables.OptimizationJob: "job, unique with the hash"
OptimizerDb.ResultTables.TopologyWorkItem -> OptimizerDb.CommandTables.ActiveWorker: "claimed by, open again once its heartbeat is older than the timeout"
OptimizerDb.ResultTables.Topology -> OptimizerDb.ResultTables.Strategy: "grouped into"
OptimizerDb.ResultTables.TopologyWorkItem -> OptimizerDb.ResultTables.Topology: "topology to evaluate, per stage"
OptimizerDb.ResultTables.StageTopologyResult -> OptimizerDb.ResultTables.Topology: "result of, per stage"
`;case`postgresCommandsLiveness`:return`direction: down

@toopToop: {
  label: "ToOp Engine"

  DcOptimizer: {
    label: "DC-Optimizer"
  }
  AcValidator: {
    label: "AC-Validator"
  }
}
AcFastFail: {
  label: "AC fast-failing stage"
}
OptimizerDb: {
  label: "Optimizer database"
  shape: stored_data

  ResultTables: {
    label: "Result tables"

    TopologyWorkItem: {
      label: "topology_work_item"
      shape: document
    }
  }
  CommandTables: {
    label: "Command tables"

    StageWorkItem: {
      label: "stage_work_item"
      shape: document
    }
    ActiveWorker: {
      label: "active_worker"
      shape: document
    }
    OptimizationJob: {
      label: "optimization_job"
      shape: document
    }
  }
}

OptimizerDb.CommandTables.StageWorkItem -> OptimizerDb.CommandTables.OptimizationJob: "job, one row per stage"
OptimizerDb.CommandTables.ActiveWorker -> OptimizerDb.CommandTables.OptimizationJob: "job it is on"
OptimizerDb.ResultTables.TopologyWorkItem -> OptimizerDb.CommandTables.ActiveWorker: "claimed by, open again once its heartbeat is older than the timeout"
@toopToop.DcOptimizer -> OptimizerDb.CommandTables.StageWorkItem: "claims the DC stage with SKIP LOCKED, refreshes the lease every epoch"
@toopToop.DcOptimizer -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds"
AcFastFail -> OptimizerDb.CommandTables.StageWorkItem: "reads the shared state of the stage"
AcFastFail -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds, the job it is on"
AcFastFail -> OptimizerDb.ResultTables.TopologyWorkItem: "claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN"
@toopToop.AcValidator -> OptimizerDb.CommandTables.StageWorkItem: "reads the shared state of the stage"
@toopToop.AcValidator -> OptimizerDb.CommandTables.ActiveWorker: "heartbeat every few seconds, the job it is on"
@toopToop.AcValidator -> OptimizerDb.ResultTables.TopologyWorkItem: "claims items; deletes its item with the result"
`;case`postgresCommandsLifecycle`:return`direction: right

@toopClient: {
  label: "Operator / orchestration client"
  shape: c4-person
}
OptimizerDbCommandTablesOptimizationJob: {
  label: "optimization_job"
  shape: document
}
@toopToopDcOptimizer: {
  label: "DC-Optimizer"
}
AcFastFail: {
  label: "AC fast-failing stage"
}
@toopToopAcValidator: {
  label: "AC-Validator"
}
OptimizerDbCommandTablesStageWorkItem: {
  label: "stage_work_item"
  shape: document
}
OptimizerDbCommandTablesStageExecutionHistory: {
  label: "stage_execution_history"
  shape: document
}
OptimizerDbResultTablesTopology: {
  label: "topology"
  shape: document
}
OptimizerDbResultTablesStageTopologyResult: {
  label: "stage_topology_result"
  shape: document
}
OptimizerDbResultTablesTopologyWorkItem: {
  label: "topology_work_item"
  shape: document
}
@toopLoadflowStore: {
  label: "Loadflow result store"
  shape: stored_data
}
@toopDownstream: {
  label: "Frontend / downstream systems"
}

@toopClient -> OptimizerDbCommandTablesOptimizationJob: "start_optimization: the request, write-once"
@toopClient -> OptimizerDbCommandTablesStageWorkItem: "one work item per stage, TRIGGERED"
@toopToopDcOptimizer -> OptimizerDbCommandTablesStageWorkItem: "claims DC: RUNNING, attempt 1, lease"
AcFastFail -> OptimizerDbCommandTablesStageWorkItem: "claims AC_FAST_FAILING"
@toopToopAcValidator -> OptimizerDbCommandTablesStageWorkItem: "claims AC"
@toopToopDcOptimizer -> OptimizerDbCommandTablesStageExecutionHistory: "attempt row with worker_id and claimed_at"
@toopToopDcOptimizer -> OptimizerDbCommandTablesStageWorkItem: "every epoch: refresh lease, check for CANCELLED"
@toopToopDcOptimizer -> OptimizerDbCommandTablesStageExecutionHistory: "every epoch: KPIs"
@toopToopDcOptimizer -> OptimizerDbResultTablesTopology: "new elites as immutable rows"
@toopToopDcOptimizer -> OptimizerDbResultTablesStageTopologyResult: "DC fitness and worst-k cases"
@toopToopDcOptimizer -> OptimizerDbResultTablesTopologyWorkItem: "work item for AC_FAST_FAILING"
OptimizerDbResultTablesTopologyWorkItem -> AcFastFail: "selects among the open items and claims them"
AcFastFail -> OptimizerDbResultTablesStageTopologyResult: "worst-k AC result"
AcFastFail -> OptimizerDbResultTablesTopologyWorkItem: "deletes its item; ACCEPTED or WARN: item for AC"
OptimizerDbResultTablesTopologyWorkItem -> @toopToopAcValidator: "claims an open item"
@toopToopAcValidator -> @toopLoadflowStore: "full AC N-1 per topology"
@toopToopAcValidator -> OptimizerDbResultTablesStageTopologyResult: "fitness, metrics, loadflow reference"
@toopToopAcValidator -> OptimizerDbResultTablesTopologyWorkItem: "deletes its item"
@toopToopDcOptimizer -> OptimizerDbCommandTablesStageWorkItem: "COMPLETED, lease cleared"
AcFastFail -> OptimizerDbCommandTablesStageWorkItem: "COMPLETED"
@toopToopAcValidator -> OptimizerDbCommandTablesStageWorkItem: "COMPLETED"
OptimizerDbCommandTablesStageWorkItem -> @toopClient: "all stages COMPLETED: job is done"
OptimizerDbResultTablesStageTopologyResult -> @toopDownstream: "accepted topologies for review"
`;default:throw Error(`Unknown viewId: `+e)}};export{e as d2Source};