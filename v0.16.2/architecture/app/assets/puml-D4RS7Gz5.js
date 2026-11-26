var e=e=>{switch(e){case`index`:return`@startuml
title "Landscape view"
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

skinparam database<<OptimizerDb>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<AcFastFail>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<@toopToop>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<@toopDownstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam person<<@toopClient>>{
  BackgroundColor #0284c7
  FontColor #f0f9ff
  BorderColor #0369a1
}
database "==Optimizer database\\n<size:10>[PostgreSQL, SQLModel]</size>\\n\\nReplaces the commands, results and heartbeat topics. Request payload,\\nper-stage work item and execution history are separate tables so that\\neach can evolve on its own. Workers poll with SELECT FOR UPDATE SKIP\\nLOCKED and hold a lease instead of a hard lock." <<OptimizerDb>> as OptimizerDb
rectangle "==AC fast-failing stage\\n<size:10>[Kafka worker]</size>\\n\\nNew stage, OptimizerType.AC_FAST_FAILING: computes only the worst-k N-1\\ncases the DC stage reported, to reject topologies cheaply before the\\nfull AC run." <<AcFastFail>> as AcFastFail
rectangle "==ToOp Engine\\n\\nTopology optimization engine: Importer, DC-Optimizer, AC-Validator." <<@toopToop>> as @toopToop
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<@toopDownstream>> as @toopDownstream
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<@toopClient>> as @toopClient

OptimizerDb .[#8D8D8D,thickness=2].> AcFastFail : <color:#8D8D8D>open work items to select from and claim
OptimizerDb .[#8D8D8D,thickness=2].> @toopToop : <color:#8D8D8D>open work items to claim
OptimizerDb .[#8D8D8D,thickness=2].> @toopDownstream : <color:#8D8D8D>accepted topologies for review
OptimizerDb .[#8D8D8D,thickness=2].> @toopClient : <color:#8D8D8D>[...]
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDb : <color:#8D8D8D>[...]
@toopToop .[#8D8D8D,thickness=2].> OptimizerDb : <color:#8D8D8D>[...]
@toopClient .[#8D8D8D,thickness=2].> OptimizerDb : <color:#8D8D8D>[...]
@enduml
`;case`postgresCommands`:return`@startuml
title "Postgres instead of Kafka"
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

skinparam rectangle<<OptimizerDbResultTablesStrategy>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<@toopToopDcOptimizer>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<OptimizerDbCommandTablesActiveWorker>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam person<<@toopClient>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesTopology>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesTopologyWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<AcFastFail>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<@toopToopAcValidator>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<OptimizerDbResultTablesStageTopologyResult>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesOptimizationJob>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageExecutionHistory>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<@toopDownstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "Optimizer database" <<OptimizerDb>> as OptimizerDb {
  skinparam RectangleBorderColor<<OptimizerDb>> #428a4f
  skinparam RectangleFontColor<<OptimizerDb>> #428a4f
  skinparam RectangleBorderStyle<<OptimizerDb>> dashed

  rectangle "Result tables" <<OptimizerDbResultTables>> as OptimizerDbResultTables {
    skinparam RectangleBorderColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbResultTables>> dashed

    rectangle "==strategy\\n\\nGroup of topologies for multi-timestep runs, not supported yet." <<OptimizerDbResultTablesStrategy>> as OptimizerDbResultTablesStrategy
    rectangle "==topology\\n\\nThe immutable definition of a topology: actions, disconnections, PST setpoints, hash." <<OptimizerDbResultTablesTopology>> as OptimizerDbResultTablesTopology
    rectangle "==topology_work_item\\n\\nOpen work: one row per topology and stage that still has to be evaluated, with the claiming worker and an attempt counter." <<OptimizerDbResultTablesTopologyWorkItem>> as OptimizerDbResultTablesTopologyWorkItem
    rectangle "==stage_topology_result\\n\\nArchive: one row per evaluated topology and stage, with outcome, fitness, metrics and loadflow reference." <<OptimizerDbResultTablesStageTopologyResult>> as OptimizerDbResultTablesStageTopologyResult
  }
  rectangle "Command tables" <<OptimizerDbCommandTables>> as OptimizerDbCommandTables {
    skinparam RectangleBorderColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbCommandTables>> dashed

    rectangle "==active_worker\\n\\nOne row per worker: last heartbeat, the job it is on, visible GPUs and CPUs." <<OptimizerDbCommandTablesActiveWorker>> as OptimizerDbCommandTablesActiveWorker
    rectangle "==optimization_job\\n\\nThe immutable request: grid files, DC and AC parameters, priority, created_at." <<OptimizerDbCommandTablesOptimizationJob>> as OptimizerDbCommandTablesOptimizationJob
    rectangle "==stage_work_item\\n\\nOne row per job and stage, polled by the workers: status, attempt counter, lease expiry." <<OptimizerDbCommandTablesStageWorkItem>> as OptimizerDbCommandTablesStageWorkItem
    rectangle "==stage_execution_history\\n\\nOne row per attempt: worker, timestamps, KPIs, error message." <<OptimizerDbCommandTablesStageExecutionHistory>> as OptimizerDbCommandTablesStageExecutionHistory
  }
}
rectangle "ToOp Engine" <<@toopToop>> as @toopToop {
  skinparam RectangleBorderColor<<@toopToop>> #64748b
  skinparam RectangleFontColor<<@toopToop>> #64748b
  skinparam RectangleBorderStyle<<@toopToop>> dashed

  rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<@toopToopDcOptimizer>> as @toopToopDcOptimizer
  rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<@toopToopAcValidator>> as @toopToopAcValidator
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<@toopClient>> as @toopClient
rectangle "==AC fast-failing stage\\n<size:10>[Kafka worker]</size>\\n\\nNew stage, OptimizerType.AC_FAST_FAILING: computes only the worst-k N-1\\ncases the DC stage reported, to reject topologies cheaply before the\\nfull AC run." <<AcFastFail>> as AcFastFail
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<@toopDownstream>> as @toopDownstream

@toopClient .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#8D8D8D>start_optimization: insert the request
@toopClient .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>one TRIGGERED row per stage; cancel_optimization sets CANCELLED
OptimizerDbCommandTablesStageWorkItem .[#8D8D8D,thickness=2].> @toopClient : <color:#8D8D8D>job status inferred from the work items
OptimizerDbCommandTablesActiveWorker .[#8D8D8D,thickness=2].> @toopClient : <color:#8D8D8D>which workers are alive
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>claims the DC stage with SKIP LOCKED, refreshes the lease every epoch
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageExecutionHistory : <color:#8D8D8D>attempt row, KPIs per epoch
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopology : <color:#8D8D8D>new elites, immutable
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>work item for AC_FAST_FAILING
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>DC fitness, metrics, worst-k cases
OptimizerDbResultTablesTopologyWorkItem .[#8D8D8D,thickness=2].> AcFastFail : <color:#8D8D8D>open work items to select from and claim
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>reads the shared state of the stage
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageExecutionHistory : <color:#8D8D8D>attempt row, KPIs
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds, the job it is on
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>worst-k AC result
OptimizerDbResultTablesTopologyWorkItem .[#8D8D8D,thickness=2].> @toopToopAcValidator : <color:#8D8D8D>open work items to claim
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>reads the shared state of the stage
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageExecutionHistory : <color:#8D8D8D>attempt row, KPIs
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds, the job it is on
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>claims items; deletes its item with the result
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>full AC result with loadflow reference
OptimizerDbResultTablesStageTopologyResult .[#8D8D8D,thickness=2].> @toopDownstream : <color:#8D8D8D>accepted topologies for review
@enduml
`;case`postgresCommandsTables`:return`@startuml
title "Tables of the optimizer database"
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

skinparam rectangle<<OptimizerDbResultTablesTopologyWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesStageTopologyResult>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesTopology>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesStrategy>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageExecutionHistory>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesActiveWorker>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesOptimizationJob>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "Optimizer database" <<OptimizerDb>> as OptimizerDb {
  skinparam RectangleBorderColor<<OptimizerDb>> #428a4f
  skinparam RectangleFontColor<<OptimizerDb>> #428a4f
  skinparam RectangleBorderStyle<<OptimizerDb>> dashed

  rectangle "Result tables" <<OptimizerDbResultTables>> as OptimizerDbResultTables {
    skinparam RectangleBorderColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbResultTables>> dashed

    rectangle "==topology_work_item\\n\\nOpen work: one row per topology and stage that still has to be evaluated, with the claiming worker and an attempt counter." <<OptimizerDbResultTablesTopologyWorkItem>> as OptimizerDbResultTablesTopologyWorkItem
    rectangle "==stage_topology_result\\n\\nArchive: one row per evaluated topology and stage, with outcome, fitness, metrics and loadflow reference." <<OptimizerDbResultTablesStageTopologyResult>> as OptimizerDbResultTablesStageTopologyResult
    rectangle "==topology\\n\\nThe immutable definition of a topology: actions, disconnections, PST setpoints, hash." <<OptimizerDbResultTablesTopology>> as OptimizerDbResultTablesTopology
    rectangle "==strategy\\n\\nGroup of topologies for multi-timestep runs, not supported yet." <<OptimizerDbResultTablesStrategy>> as OptimizerDbResultTablesStrategy
  }
  rectangle "Command tables" <<OptimizerDbCommandTables>> as OptimizerDbCommandTables {
    skinparam RectangleBorderColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbCommandTables>> dashed

    rectangle "==stage_execution_history\\n\\nOne row per attempt: worker, timestamps, KPIs, error message." <<OptimizerDbCommandTablesStageExecutionHistory>> as OptimizerDbCommandTablesStageExecutionHistory
    rectangle "==stage_work_item\\n\\nOne row per job and stage, polled by the workers: status, attempt counter, lease expiry." <<OptimizerDbCommandTablesStageWorkItem>> as OptimizerDbCommandTablesStageWorkItem
    rectangle "==active_worker\\n\\nOne row per worker: last heartbeat, the job it is on, visible GPUs and CPUs." <<OptimizerDbCommandTablesActiveWorker>> as OptimizerDbCommandTablesActiveWorker
    rectangle "==optimization_job\\n\\nThe immutable request: grid files, DC and AC parameters, priority, created_at." <<OptimizerDbCommandTablesOptimizationJob>> as OptimizerDbCommandTablesOptimizationJob
  }
}

OptimizerDbCommandTablesStageWorkItem .[#64748b,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#64748b>job, one row per stage
OptimizerDbCommandTablesActiveWorker .[#64748b,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#64748b>job it is on
OptimizerDbCommandTablesStageExecutionHistory .[#64748b,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#64748b>one row per attempt
OptimizerDbCommandTablesStageExecutionHistory .[#64748b,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#64748b>worker of the attempt
OptimizerDbResultTablesTopology .[#64748b,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#64748b>job, unique with the hash
OptimizerDbResultTablesTopologyWorkItem .[#64748b,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#64748b>claimed by, open again once its heartbeat is older than the timeout
OptimizerDbResultTablesTopology .[#64748b,thickness=2].> OptimizerDbResultTablesStrategy : <color:#64748b>grouped into
OptimizerDbResultTablesTopologyWorkItem .[#64748b,thickness=2].> OptimizerDbResultTablesTopology : <color:#64748b>topology to evaluate, per stage
OptimizerDbResultTablesStageTopologyResult .[#64748b,thickness=2].> OptimizerDbResultTablesTopology : <color:#64748b>result of, per stage
@enduml
`;case`postgresCommandsLiveness`:return`@startuml
title "Worker liveness and claims"
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

skinparam rectangle<<AcFastFail>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<@toopToopDcOptimizer>>{
  BackgroundColor #6366f1
  FontColor #eef2ff
  BorderColor #4f46e5
}
skinparam rectangle<<@toopToopAcValidator>>{
  BackgroundColor #AC4D39
  FontColor #FBD3CB
  BorderColor #853A2D
}
skinparam rectangle<<OptimizerDbResultTablesTopologyWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesActiveWorker>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesOptimizationJob>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
rectangle "ToOp Engine" <<@toopToop>> as @toopToop {
  skinparam RectangleBorderColor<<@toopToop>> #3b82f6
  skinparam RectangleFontColor<<@toopToop>> #3b82f6
  skinparam RectangleBorderStyle<<@toopToop>> dashed

  rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nClaims one DC stage at a time and holds it with a lease." <<@toopToopDcOptimizer>> as @toopToopDcOptimizer
  rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nShares the AC stages of a job with other AC workers, claims topology work items." <<@toopToopAcValidator>> as @toopToopAcValidator
}
rectangle "==AC fast-failing stage\\n<size:10>[Python]</size>\\n\\nShares the AC stages of a job with other AC workers, claims topology work items." <<AcFastFail>> as AcFastFail
rectangle "Optimizer database" <<OptimizerDb>> as OptimizerDb {
  skinparam RectangleBorderColor<<OptimizerDb>> #428a4f
  skinparam RectangleFontColor<<OptimizerDb>> #428a4f
  skinparam RectangleBorderStyle<<OptimizerDb>> dashed

  rectangle "Result tables" <<OptimizerDbResultTables>> as OptimizerDbResultTables {
    skinparam RectangleBorderColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbResultTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbResultTables>> dashed

    rectangle "==topology_work_item\\n\\nOpen work: one row per topology and stage that still has to be evaluated, with the claiming worker and an attempt counter." <<OptimizerDbResultTablesTopologyWorkItem>> as OptimizerDbResultTablesTopologyWorkItem
  }
  rectangle "Command tables" <<OptimizerDbCommandTables>> as OptimizerDbCommandTables {
    skinparam RectangleBorderColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleFontColor<<OptimizerDbCommandTables>> #428a4f
    skinparam RectangleBorderStyle<<OptimizerDbCommandTables>> dashed

    rectangle "==stage_work_item\\n\\nOne row per job and stage, polled by the workers: status, attempt counter, lease expiry." <<OptimizerDbCommandTablesStageWorkItem>> as OptimizerDbCommandTablesStageWorkItem
    rectangle "==active_worker\\n\\nOne row per worker: last heartbeat, the job it is on, visible GPUs and CPUs." <<OptimizerDbCommandTablesActiveWorker>> as OptimizerDbCommandTablesActiveWorker
    rectangle "==optimization_job\\n\\nThe immutable request: grid files, DC and AC parameters, priority, created_at." <<OptimizerDbCommandTablesOptimizationJob>> as OptimizerDbCommandTablesOptimizationJob
  }
}

OptimizerDbCommandTablesStageWorkItem .[#64748b,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#64748b>job, one row per stage
OptimizerDbCommandTablesActiveWorker .[#64748b,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#64748b>job it is on
OptimizerDbResultTablesTopologyWorkItem .[#64748b,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#64748b>claimed by, open again once its heartbeat is older than the timeout
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>claims the DC stage with SKIP LOCKED, refreshes the lease every epoch
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>reads the shared state of the stage
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds, the job it is on
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>claims items; with the result deletes its item and inserts the item for AC if ACCEPTED or WARN
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>reads the shared state of the stage
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesActiveWorker : <color:#8D8D8D>heartbeat every few seconds, the job it is on
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>claims items; deletes its item with the result
@enduml
`;case`postgresCommandsLifecycle`:return`@startuml
title "One job on the database"
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

skinparam person<<@toopClient>>{
  BackgroundColor #0284c7
  FontColor #f0f9ff
  BorderColor #0369a1
}
skinparam rectangle<<OptimizerDbCommandTablesOptimizationJob>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<@toopToopDcOptimizer>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<AcFastFail>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<@toopToopAcValidator>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<OptimizerDbCommandTablesStageWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbCommandTablesStageExecutionHistory>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesTopology>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesStageTopologyResult>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam rectangle<<OptimizerDbResultTablesTopologyWorkItem>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
skinparam database<<@toopLoadflowStore>>{
  BackgroundColor #428a4f
  FontColor #f8fafc
  BorderColor #2d5d39
}
skinparam rectangle<<@toopDownstream>>{
  BackgroundColor #64748b
  FontColor #f8fafc
  BorderColor #475569
}
person "==Operator / orchestration client\\n\\nDrives the engine either directly from Python or by producing Kafka\\ncommands. ToOp ships no GUI or system integration of its own.\\nIn operational use the whole run must finish inside a 15 minute window,\\nbecause topology measures have to be fixed before redispatch planning." <<@toopClient>> as @toopClient
rectangle "==optimization_job\\n\\nThe immutable request: grid files, DC and AC parameters, priority, created_at." <<OptimizerDbCommandTablesOptimizationJob>> as OptimizerDbCommandTablesOptimizationJob
rectangle "==DC-Optimizer\\n<size:10>[Python, JAX / XLA]</size>\\n\\nQuality-diversity search over the action set. The whole loop is\\nGPU-resident, so no host transfer happens per iteration; results leave\\nonly once per epoch. JAX JIT costs about 13s on the first epoch." <<@toopToopDcOptimizer>> as @toopToopDcOptimizer
rectangle "==AC fast-failing stage\\n<size:10>[Kafka worker]</size>\\n\\nNew stage, OptimizerType.AC_FAST_FAILING: computes only the worst-k N-1\\ncases the DC stage reported, to reject topologies cheaply before the\\nfull AC run." <<AcFastFail>> as AcFastFail
rectangle "==AC-Validator\\n<size:10>[Python, PyPowSyBl, polars, SQLite]</size>\\n\\nProposes no topologies of its own -- it is the quality gate in front of\\nthe operator. What it does produce is the AC loadflow results: every\\ncandidate it evaluates gets a full result folder written to the\\nloadflow store and referenced by handle.\\n\\nRuns concurrently with the DC-Optimizer and outlives it, so the last DC\\ntopologies still get validated. Far more candidates arrive than can be\\nAC-evaluated in the time budget, so much of this component is about\\nchoosing which ones are worth spending a loadflow on." <<@toopToopAcValidator>> as @toopToopAcValidator
rectangle "==stage_work_item\\n\\nOne row per job and stage, polled by the workers: status, attempt counter, lease expiry." <<OptimizerDbCommandTablesStageWorkItem>> as OptimizerDbCommandTablesStageWorkItem
rectangle "==stage_execution_history\\n\\nOne row per attempt: worker, timestamps, KPIs, error message." <<OptimizerDbCommandTablesStageExecutionHistory>> as OptimizerDbCommandTablesStageExecutionHistory
rectangle "==topology\\n\\nThe immutable definition of a topology: actions, disconnections, PST setpoints, hash." <<OptimizerDbResultTablesTopology>> as OptimizerDbResultTablesTopology
rectangle "==stage_topology_result\\n\\nArchive: one row per evaluated topology and stage, with outcome, fitness, metrics and loadflow reference." <<OptimizerDbResultTablesStageTopologyResult>> as OptimizerDbResultTablesStageTopologyResult
rectangle "==topology_work_item\\n\\nOpen work: one row per topology and stage that still has to be evaluated, with the claiming worker and an attempt counter." <<OptimizerDbResultTablesTopologyWorkItem>> as OptimizerDbResultTablesTopologyWorkItem
database "==Loadflow result store\\n<size:10>[fsspec, polars, Parquet]</size>\\n\\nLoadflow tables addressed by a StoredLoadflowReference passed in messages,\\nso the tables themselves stay out of Kafka.\\n\\nThe AC-Validator is the main producer: every topology it evaluates gets\\nits loadflow results written here and referenced by handle. The importer\\ncontributes one folder, the AC N-1 of the import, run once on the\\nimported grid and stored with its permanent and artificial branch limits,\\nwhich the validator reads back as its baseline -- and computes and stores\\nitself if no reference was passed in.\\n\\nA reference is a folder, not a file. Inside it, one Parquet file per\\nresult family plus a small metadata sidecar. Polars writes them with\\nsink_parquet and reads them back with scan_parquet, so a consumer can\\npredicate-pushdown into a single table without materializing the rest.\\n\\nEvery table is indexed by (timestep, contingency, ...) -- that pair is the\\njoin key across the whole folder." <<@toopLoadflowStore>> as @toopLoadflowStore
rectangle "==Frontend / downstream systems\\n\\nWhere an operator reviews the proposed actions and exports the accepted\\nones. Not part of this repository." <<@toopDownstream>> as @toopDownstream

@toopClient .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesOptimizationJob : <color:#8D8D8D>start_optimization: the request, write-once
@toopClient .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>one work item per stage, TRIGGERED
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>claims DC: RUNNING, attempt 1, lease
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>claims AC_FAST_FAILING
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>claims AC
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageExecutionHistory : <color:#8D8D8D>attempt row with worker_id and claimed_at
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>every epoch: refresh lease, check for CANCELLED
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageExecutionHistory : <color:#8D8D8D>every epoch: KPIs
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopology : <color:#8D8D8D>new elites as immutable rows
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>DC fitness and worst-k cases
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>work item for AC_FAST_FAILING
OptimizerDbResultTablesTopologyWorkItem .[#8D8D8D,thickness=2].> AcFastFail : <color:#8D8D8D>selects among the open items and claims them
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>worst-k AC result
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>deletes its item; ACCEPTED or WARN: item for AC
OptimizerDbResultTablesTopologyWorkItem .[#8D8D8D,thickness=2].> @toopToopAcValidator : <color:#8D8D8D>claims an open item
@toopToopAcValidator .[#8D8D8D,thickness=2].> @toopLoadflowStore : <color:#8D8D8D>full AC N-1 per topology
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbResultTablesStageTopologyResult : <color:#8D8D8D>fitness, metrics, loadflow reference
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbResultTablesTopologyWorkItem : <color:#8D8D8D>deletes its item
@toopToopDcOptimizer .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>COMPLETED, lease cleared
AcFastFail .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>COMPLETED
@toopToopAcValidator .[#8D8D8D,thickness=2].> OptimizerDbCommandTablesStageWorkItem : <color:#8D8D8D>COMPLETED
OptimizerDbCommandTablesStageWorkItem .[#8D8D8D,thickness=2].> @toopClient : <color:#8D8D8D>all stages COMPLETED: job is done
OptimizerDbResultTablesStageTopologyResult .[#8D8D8D,thickness=2].> @toopDownstream : <color:#8D8D8D>accepted topologies for review
@enduml
`;default:throw Error(`Unknown viewId: `+e)}};export{e as pumlSource};