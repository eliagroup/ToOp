var e=e=>{switch(e){case`index`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=index,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2,
        style=""
    ];
    optimizerdb [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Optimizer database</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">PostgreSQL, SQLModel</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Replaces the commands, results and heartbeat<BR/>topics. Request payload,<BR/>per-stage work item and execution history are<BR/>separate tables so that<BR/>each can evolve on its own. Workers poll with</FONT></TD></TR></TABLE>>,
        likec4_id=optimizerDb,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    acfastfail [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC fast-failing stage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Kafka worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">New stage, OptimizerType.AC_FAST_FAILING:<BR/>computes only the worst-k N-1<BR/>cases the DC stage reported, to reject<BR/>topologies cheaply before the<BR/>full AC run.</FONT></TD></TR></TABLE>>,
        likec4_id=acFastFail,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    optimizerdb -> acfastfail [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">open work items to select from and claim</FONT></TD></TR></TABLE>>,
        likec4_id="3z5fdz",
        style=dashed];
    toop [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">ToOp Engine</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Topology optimization engine: Importer,<BR/>DC-Optimizer, AC-Validator.</FONT></TD></TR></TABLE>>,
        likec4_id="@toop.toop",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    optimizerdb -> toop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">open work items to claim</FONT></TD></TR></TABLE>>,
        likec4_id="1xcf889",
        style=dashed];
    downstream [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>,
        likec4_id="@toop.downstream",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    optimizerdb -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">accepted topologies for review</FONT></TD></TR></TABLE>>,
        likec4_id="16rtd43",
        minlen=1,
        style=dashed];
    client [color="#0369a1",
        fillcolor="#0284c7",
        fontcolor="#f0f9ff",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>,
        likec4_id="@toop.client",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    optimizerdb -> client [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="29zufo",
        style=dashed];
    acfastfail -> optimizerdb [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1cpbc93",
        style=dashed];
    toop -> optimizerdb [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=o8klop,
        style=dashed];
    client -> optimizerdb [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=b6quus,
        style=dashed];
}`;case`postgresCommands`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=postgresCommands,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=LR,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2,
        style=""
    ];
    subgraph cluster_optimizerdb {
        graph [color="#1c3021",
            fillcolor="#29472f",
            label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>OPTIMIZER DATABASE</B></FONT>>,
            likec4_depth=2,
            likec4_id=optimizerDb,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_resulttables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>RESULT TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.resultTables",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            strategy [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">strategy</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Group of topologies for multi-timestep runs,<BR/>not supported yet.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.strategy",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            topology [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable definition of a topology:<BR/>actions, disconnections, PST setpoints, hash.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.topology",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            topologyworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Open work: one row per topology and stage<BR/>that still has to be evaluated, with the<BR/>claiming worker and an attempt counter.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.topologyWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stagetopologyresult [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_topology_result</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Archive: one row per evaluated topology and<BR/>stage, with outcome, fitness, metrics and<BR/>loadflow reference.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.stageTopologyResult",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_commandtables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>COMMAND TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.commandTables",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            activeworker [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">active_worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per worker: last heartbeat, the job<BR/>it is on, visible GPUs and CPUs.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.activeWorker",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            optimizationjob [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimization_job</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable request: grid files, DC and AC<BR/>parameters, priority, created_at.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.optimizationJob",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stageworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per job and stage, polled by the<BR/>workers: status, attempt counter, lease<BR/>expiry.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.stageWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stageexecutionhistory [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_execution_history</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per attempt: worker, timestamps,<BR/>KPIs, error message.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.stageExecutionHistory",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
    }
    subgraph cluster_toop {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=1,
            likec4_id="@toop.toop",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        dcoptimizer [color="#4f46e5",
            fillcolor="#6366f1",
            fontcolor="#eef2ff",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c7d2fe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>,
            likec4_id="@toop.toop.dcOptimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acvalidator [color="#853A2D",
            fillcolor="#AC4D39",
            fontcolor="#FBD3CB",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>,
            likec4_id="@toop.toop.acValidator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    dcoptimizer -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds</FONT></TD></TR></TABLE>>,
        likec4_id="1bstfej",
        style=dashed];
    dcoptimizer -> topology [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">new elites, immutable</FONT></TD></TR></TABLE>>,
        likec4_id="1h4te9i",
        minlen=1,
        style=dashed];
    dcoptimizer -> topologyworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">work item for AC_FAST_FAILING</FONT></TD></TR></TABLE>>,
        likec4_id=wzj26a,
        style=dashed];
    dcoptimizer -> stagetopologyresult [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">DC fitness, metrics, worst-k cases</FONT></TD></TR></TABLE>>,
        likec4_id="1hos6wr",
        style=dashed];
    dcoptimizer -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims the DC stage with SKIP LOCKED,<BR/>refreshes the lease every epoch</FONT></TD></TR></TABLE>>,
        likec4_id=ps04j5,
        style=dashed];
    dcoptimizer -> stageexecutionhistory [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">attempt row, KPIs per epoch</FONT></TD></TR></TABLE>>,
        likec4_id=vxr3xx,
        style=dashed];
    client [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>,
        likec4_id="@toop.client",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    activeworker -> client [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">which workers are alive</FONT></TD></TR></TABLE>>,
        likec4_id="18onapw",
        style=dashed,
        weight=2];
    client -> optimizationjob [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">start_optimization: insert the request</FONT></TD></TR></TABLE>>,
        likec4_id="1x8l1q0",
        minlen=1,
        style=dashed];
    client -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">one TRIGGERED row per stage;<BR/>cancel_optimization sets CANCELLED</FONT></TD></TR></TABLE>>,
        likec4_id="12t1r1q",
        style=dashed];
    acfastfail [color="#853A2D",
        fillcolor="#AC4D39",
        fontcolor="#FBD3CB",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC fast-failing stage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Kafka worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">New stage, OptimizerType.AC_FAST_FAILING:<BR/>computes only the worst-k N-1<BR/>cases the DC stage reported, to reject<BR/>topologies cheaply before the<BR/>full AC run.</FONT></TD></TR></TABLE>>,
        likec4_id=acFastFail,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    topologyworkitem -> acfastfail [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">open work items to select from and claim</FONT></TD></TR></TABLE>>,
        likec4_id="4c5wu",
        style=dashed];
    topologyworkitem -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">open work items to claim</FONT></TD></TR></TABLE>>,
        likec4_id="1ulbwqa",
        style=dashed];
    acfastfail -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds, the job it<BR/>is on</FONT></TD></TR></TABLE>>,
        likec4_id="15wn6xj",
        style=dashed,
        weight=3];
    acfastfail -> topologyworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims items; with the result deletes<BR/>its item and inserts the item for AC if<BR/>ACCEPTED or WARN</FONT></TD></TR></TABLE>>,
        likec4_id=y8a12m,
        style=dashed];
    acfastfail -> stagetopologyresult [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">worst-k AC result</FONT></TD></TR></TABLE>>,
        likec4_id=xgb2vr,
        style=dashed,
        weight=3];
    acfastfail -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reads the shared state of the stage</FONT></TD></TR></TABLE>>,
        likec4_id=n2pzx,
        style=dashed,
        weight=3];
    acfastfail -> stageexecutionhistory [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">attempt row, KPIs</FONT></TD></TR></TABLE>>,
        likec4_id="23fgwp",
        style=dashed,
        weight=3];
    acvalidator -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds, the job it<BR/>is on</FONT></TD></TR></TABLE>>,
        likec4_id="1byabbf",
        style=dashed];
    acvalidator -> topologyworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims items; deletes its item with the<BR/>result</FONT></TD></TR></TABLE>>,
        likec4_id=cimnoy,
        style=dashed];
    acvalidator -> stagetopologyresult [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">full AC result with loadflow reference</FONT></TD></TR></TABLE>>,
        likec4_id=ey5w7f,
        style=dashed];
    acvalidator -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reads the shared state of the stage</FONT></TD></TR></TABLE>>,
        likec4_id="1cpe0e9",
        style=dashed];
    acvalidator -> stageexecutionhistory [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">attempt row, KPIs</FONT></TD></TR></TABLE>>,
        likec4_id="1sxtxad",
        style=dashed];
    downstream [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>,
        likec4_id="@toop.downstream",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    stagetopologyresult -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">accepted topologies for review</FONT></TD></TR></TABLE>>,
        likec4_id=zf2kc3,
        minlen=1,
        style=dashed,
        weight=2];
    stageworkitem -> client [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job status inferred from the work items</FONT></TD></TR></TABLE>>,
        likec4_id="3z6uoe",
        style=dashed];
}`;case`postgresCommandsTables`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=postgresCommandsTables,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2,
        style=""
    ];
    subgraph cluster_optimizerdb {
        graph [color="#1c3021",
            fillcolor="#29472f",
            label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>OPTIMIZER DATABASE</B></FONT>>,
            likec4_depth=2,
            likec4_id=optimizerDb,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_resulttables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>RESULT TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.resultTables",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            topologyworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.resultTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Open work: one row per topology and stage<BR/>that still has to be evaluated, with the<BR/>claiming worker and an attempt counter.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.topologyWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stagetopologyresult [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.resultTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_topology_result</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Archive: one row per evaluated topology and<BR/>stage, with outcome, fitness, metrics and<BR/>loadflow reference.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.stageTopologyResult",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            topology [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.resultTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable definition of a topology:<BR/>actions, disconnections, PST setpoints, hash.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.topology",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            strategy [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.resultTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">strategy</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Group of topologies for multi-timestep runs,<BR/>not supported yet.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.strategy",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_commandtables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>COMMAND TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.commandTables",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            stageexecutionhistory [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_execution_history</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per attempt: worker, timestamps,<BR/>KPIs, error message.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.stageExecutionHistory",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stageworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per job and stage, polled by the<BR/>workers: status, attempt counter, lease<BR/>expiry.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.stageWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            activeworker [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">active_worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per worker: last heartbeat, the job<BR/>it is on, visible GPUs and CPUs.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.activeWorker",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            optimizationjob [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimization_job</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable request: grid files, DC and AC<BR/>parameters, priority, created_at.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.optimizationJob",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
    }
    topologyworkitem -> topology [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">topology to evaluate, per stage</FONT></TD></TR></TABLE>>,
        likec4_id="1a1ht32",
        style=solid,
        weight=3];
    topologyworkitem -> activeworker [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claimed by, open again once its<BR/>heartbeat is older than the timeout</FONT></TD></TR></TABLE>>,
        likec4_id=m63lf7,
        style=solid];
    stagetopologyresult -> topology [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">result of, per stage</FONT></TD></TR></TABLE>>,
        likec4_id=gbjcnb,
        minlen=1,
        style=solid,
        weight=3];
    topology -> strategy [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">grouped into</FONT></TD></TR></TABLE>>,
        likec4_id="1gxoq08",
        minlen=1,
        style=solid,
        weight=3];
    topology -> optimizationjob [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job, unique with the hash</FONT></TD></TR></TABLE>>,
        likec4_id="1jqvwbf",
        style=solid];
    stageexecutionhistory -> stageworkitem [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">one row per attempt</FONT></TD></TR></TABLE>>,
        likec4_id="1l8fxhq",
        style=solid];
    stageexecutionhistory -> activeworker [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">worker of the attempt</FONT></TD></TR></TABLE>>,
        likec4_id="1a4g2w4",
        style=solid,
        weight=3];
    stageworkitem -> optimizationjob [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job, one row per stage</FONT></TD></TR></TABLE>>,
        likec4_id="17ayp3w",
        style=solid,
        weight=3];
    activeworker -> optimizationjob [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job it is on</FONT></TD></TR></TABLE>>,
        likec4_id="13g315i",
        style=solid,
        weight=3];
}`;case`postgresCommandsLiveness`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=postgresCommandsLiveness,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2,
        style=""
    ];
    subgraph cluster_toop {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=1,
            likec4_id="@toop.toop",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        dcoptimizer [color="#4f46e5",
            fillcolor="#6366f1",
            fontcolor="#eef2ff",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c7d2fe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Claims one DC stage at a time and holds it<BR/>with a lease.</FONT></TD></TR></TABLE>>,
            likec4_id="@toop.toop.dcOptimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acvalidator [color="#853A2D",
            fillcolor="#AC4D39",
            fontcolor="#FBD3CB",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Shares the AC stages of a job with other AC<BR/>workers, claims topology work items.</FONT></TD></TR></TABLE>>,
            likec4_id="@toop.toop.acValidator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_optimizerdb {
        graph [color="#1c3021",
            fillcolor="#29472f",
            label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>OPTIMIZER DATABASE</B></FONT>>,
            likec4_depth=2,
            likec4_id=optimizerDb,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_resulttables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>RESULT TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.resultTables",
                likec4_level=1,
                margin=32,
                style=filled
            ];
            topologyworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Open work: one row per topology and stage<BR/>that still has to be evaluated, with the<BR/>claiming worker and an attempt counter.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.resultTables.topologyWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_commandtables {
            graph [color="#1e3524",
                fillcolor="#2c4e32",
                label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>COMMAND TABLES</B></FONT>>,
                likec4_depth=1,
                likec4_id="optimizerDb.commandTables",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            stageworkitem [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per job and stage, polled by the<BR/>workers: status, attempt counter, lease<BR/>expiry.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.stageWorkItem",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            activeworker [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">active_worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per worker: last heartbeat, the job<BR/>it is on, visible GPUs and CPUs.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.activeWorker",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            optimizationjob [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group="optimizerDb.commandTables",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimization_job</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable request: grid files, DC and AC<BR/>parameters, priority, created_at.</FONT></TD></TR></TABLE>>,
                likec4_id="optimizerDb.commandTables.optimizationJob",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
    }
    acfastfail [color="#853A2D",
        fillcolor="#AC4D39",
        fontcolor="#FBD3CB",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC fast-failing stage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Python</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Shares the AC stages of a job with other AC<BR/>workers, claims topology work items.</FONT></TD></TR></TABLE>>,
        likec4_id=acFastFail,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    acfastfail -> topologyworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims items; with the result deletes<BR/>its item and inserts the item for AC if<BR/>ACCEPTED or WARN</FONT></TD></TR></TABLE>>,
        likec4_id=y8a12m,
        style=dashed,
        weight=3];
    acfastfail -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reads the shared state of the stage</FONT></TD></TR></TABLE>>,
        likec4_id=n2pzx,
        style=dashed,
        weight=3];
    acfastfail -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds, the job it<BR/>is on</FONT></TD></TR></TABLE>>,
        likec4_id="15wn6xj",
        style=dashed,
        weight=3];
    dcoptimizer -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims the DC stage with SKIP LOCKED,<BR/>refreshes the lease every epoch</FONT></TD></TR></TABLE>>,
        likec4_id=ps04j5,
        style=dashed];
    dcoptimizer -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds</FONT></TD></TR></TABLE>>,
        likec4_id="1bstfej",
        style=dashed];
    acvalidator -> topologyworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claims items; deletes its item with the<BR/>result</FONT></TD></TR></TABLE>>,
        likec4_id=cimnoy,
        style=dashed];
    acvalidator -> stageworkitem [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reads the shared state of the stage</FONT></TD></TR></TABLE>>,
        likec4_id="1cpe0e9",
        style=dashed];
    acvalidator -> activeworker [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">heartbeat every few seconds, the job it<BR/>is on</FONT></TD></TR></TABLE>>,
        likec4_id="1byabbf",
        style=dashed];
    topologyworkitem -> activeworker [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">claimed by, open again once its<BR/>heartbeat is older than the timeout</FONT></TD></TR></TABLE>>,
        likec4_id=m63lf7,
        style=solid,
        weight=3];
    stageworkitem -> optimizationjob [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job, one row per stage</FONT></TD></TR></TABLE>>,
        likec4_id="17ayp3w",
        style=solid,
        weight=5];
    activeworker -> optimizationjob [arrowhead=normal,
        color="#64748b",
        fontcolor="#cbd5e1",
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">job it is on</FONT></TD></TR></TABLE>>,
        likec4_id="13g315i",
        style=solid,
        weight=5];
}`;case`postgresCommandsLifecycle`:return`digraph {
  likec4_viewId = "postgresCommandsLifecycle";
  bgcolor = "transparent";
  layout = "dot";
  compound = true;
  rankdir = "LR";
  splines = "spline";
  outputorder = "nodesfirst";
  nodesep = 1.528;
  ranksep = 1.667;
  pad = 0.209;
  fontname = "Arial";
  ordering = "in";
  graph [
    fontsize = 20;
    labeljust = "l";
    labelloc = "t";
  ];
  edge [
    arrowsize = 0.75;
    fontname = "Arial";
    fontsize = 14;
    penwidth = 2;
    color = "#8D8D8D";
    fontcolor = "#C9C9C9";
    style = "dashed";
  ];
  node [
    fontname = "Arial";
    shape = "rect";
    fillcolor = "#3b82f6";
    fontcolor = "#eff6ff";
    color = "#2563eb";
    style = "filled";
    penwidth = 0;
  ];
  "client" [
    likec4_id = "@toop.client";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#0284c7";
    fontcolor = "#f0f9ff";
    color = "#0369a1";
  ];
  "optimizationjob" [
    likec4_id = "optimizerDb.commandTables.optimizationJob";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimization_job</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable request: grid files, DC and AC<BR/>parameters, priority, created_at.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "dcoptimizer" [
    likec4_id = "@toop.toop.dcOptimizer";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "acfastfail" [
    likec4_id = "acFastFail";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC fast-failing stage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Kafka worker</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">New stage, OptimizerType.AC_FAST_FAILING:<BR/>computes only the worst-k N-1<BR/>cases the DC stage reported, to reject<BR/>topologies cheaply before the<BR/>full AC run.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "acvalidator" [
    likec4_id = "@toop.toop.acValidator";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "stageworkitem" [
    likec4_id = "optimizerDb.commandTables.stageWorkItem";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per job and stage, polled by the<BR/>workers: status, attempt counter, lease<BR/>expiry.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "stageexecutionhistory" [
    likec4_id = "optimizerDb.commandTables.stageExecutionHistory";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_execution_history</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One row per attempt: worker, timestamps,<BR/>KPIs, error message.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "topology" [
    likec4_id = "optimizerDb.resultTables.topology";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The immutable definition of a topology:<BR/>actions, disconnections, PST setpoints, hash.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "stagetopologyresult" [
    likec4_id = "optimizerDb.resultTables.stageTopologyResult";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">stage_topology_result</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Archive: one row per evaluated topology and<BR/>stage, with outcome, fitness, metrics and<BR/>loadflow reference.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "topologyworkitem" [
    likec4_id = "optimizerDb.resultTables.topologyWorkItem";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">topology_work_item</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Open work: one row per topology and stage<BR/>that still has to be evaluated, with the<BR/>claiming worker and an attempt counter.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "loadflowstore" [
    likec4_id = "@toop.loadflowStore";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>;
    margin = "0.223,0";
    width = 4.445;
    height = 2.5;
    fillcolor = "#428a4f";
    fontcolor = "#f8fafc";
    color = "#2d5d39";
    penwidth = 2;
    shape = "cylinder";
  ];
  "downstream" [
    likec4_id = "@toop.downstream";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "client" -> "optimizationjob" [
    likec4_id = "step-01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>0</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">start_optimization: the request,<BR/>write-once</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "client" -> "stageworkitem" [
    likec4_id = "step-02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>1</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">one work item per stage, TRIGGERED</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stageworkitem" [
    likec4_id = "step-03:par.01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>2</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">claims DC: RUNNING, attempt 1, lease</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acfastfail" -> "stageworkitem" [
    likec4_id = "step-03:par.02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>3</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">claims AC_FAST_FAILING</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "stageworkitem" [
    likec4_id = "step-03:par.03";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>4</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">claims AC</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stageexecutionhistory" [
    likec4_id = "step-04";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>5</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">attempt row with worker_id and<BR/>claimed_at</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stageworkitem" [
    likec4_id = "step-05";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>6</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">every epoch: refresh lease, check for<BR/>CANCELLED</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stageexecutionhistory" [
    likec4_id = "step-06";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>7</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">every epoch: KPIs</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "topology" [
    likec4_id = "step-07";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>8</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">new elites as immutable rows</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stagetopologyresult" [
    likec4_id = "step-08";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>9</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">DC fitness and worst-k cases</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "topologyworkitem" [
    likec4_id = "step-09";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>10</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">work item for AC_FAST_FAILING</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acfastfail" -> "topologyworkitem" [
    likec4_id = "step-10";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>11</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">selects among the open items and claims<BR/>them</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "acfastfail" -> "stagetopologyresult" [
    likec4_id = "step-11";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>12</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">worst-k AC result</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acfastfail" -> "topologyworkitem" [
    likec4_id = "step-12";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>13</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">deletes its item; ACCEPTED or WARN: item<BR/>for AC</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "topologyworkitem" [
    likec4_id = "step-13";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>14</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">claims an open item</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "acvalidator" -> "loadflowstore" [
    likec4_id = "step-14";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>15</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">full AC N-1 per topology</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "stagetopologyresult" [
    likec4_id = "step-15";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>16</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">fitness, metrics, loadflow reference</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "topologyworkitem" [
    likec4_id = "step-16";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>17</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">deletes its item</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "stageworkitem" [
    likec4_id = "step-17:par.01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>18</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">COMPLETED, lease cleared</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acfastfail" -> "stageworkitem" [
    likec4_id = "step-17:par.02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>19</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">COMPLETED</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "stageworkitem" [
    likec4_id = "step-17:par.03";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>20</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">COMPLETED</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "client" -> "stageworkitem" [
    likec4_id = "step-18";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>21</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">all stages COMPLETED: job is done</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "stagetopologyresult" -> "downstream" [
    likec4_id = "step-19";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>22</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">accepted topologies for review</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
}`;default:throw Error(`Unknown viewId: `+e)}},t=e=>{switch(e){case`index`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1724pt" height="553pt"
 viewBox="0.00 0.00 1724.00 553.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 538.05)">
<!-- optimizerdb -->
<g id="node1" class="node">
<title>optimizerdb</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M1118.11,-504.8C1118.11,-514.84 1038.08,-523 939.56,-523 841.03,-523 761,-514.84 761,-504.8 761,-504.8 761,-341 761,-341 761,-330.96 841.03,-322.8 939.56,-322.8 1038.08,-322.8 1118.11,-330.96 1118.11,-341 1118.11,-341 1118.11,-504.8 1118.11,-504.8"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M1118.11,-504.8C1118.11,-494.76 1038.08,-486.6 939.56,-486.6 841.03,-486.6 761,-494.76 761,-504.8"/>
<text xml:space="preserve" text-anchor="start" x="853.56" y="-491.7" font-family="Arial" font-size="20.00" fill="#f8fafc">Optimizer database</text>
<text xml:space="preserve" text-anchor="start" x="870.56" y="-463.7" font-family="Arial" font-size="13.00" fill="#c2f0c2">PostgreSQL, SQLModel</text>
<text xml:space="preserve" text-anchor="start" x="781.06" y="-444.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">Replaces the commands, results and heartbeat</text>
<text xml:space="preserve" text-anchor="start" x="856.56" y="-426.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">topics. Request payload,</text>
<text xml:space="preserve" text-anchor="start" x="786.06" y="-408.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">per&#45;stage work item and execution history are</text>
<text xml:space="preserve" text-anchor="start" x="862.56" y="-390.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">separate tables so that</text>
<text xml:space="preserve" text-anchor="start" x="786.56" y="-372.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">each can evolve on its own. Workers poll with</text>
</g>
<!-- acfastfail -->
<g id="node2" class="node">
<title>acfastfail</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="359.11,-180 0,-180 0,0 359.11,0 359.11,-180"/>
<text xml:space="preserve" text-anchor="start" x="91.06" y="-158.8" font-family="Arial" font-size="20.00" fill="#eff6ff">AC fast&#45;failing stage</text>
<text xml:space="preserve" text-anchor="start" x="142.06" y="-130.8" font-family="Arial" font-size="13.00" fill="#bfdbfe">Kafka worker</text>
<text xml:space="preserve" text-anchor="start" x="20.06" y="-111.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">New stage, OptimizerType.AC_FAST_FAILING:</text>
<text xml:space="preserve" text-anchor="start" x="77.06" y="-93.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">computes only the worst&#45;k N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="51.56" y="-75.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">cases the DC stage reported, to reject</text>
<text xml:space="preserve" text-anchor="start" x="80.56" y="-57.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">topologies cheaply before the</text>
<text xml:space="preserve" text-anchor="start" x="142.56" y="-39.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">full AC run.</text>
</g>
<!-- toop -->
<g id="node3" class="node">
<title>toop</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="810.58,-180 490.54,-180 490.54,0 810.58,0 810.58,-180"/>
<text xml:space="preserve" text-anchor="start" x="593.56" y="-122" font-family="Arial" font-size="20.00" fill="#eff6ff">ToOp Engine</text>
<text xml:space="preserve" text-anchor="start" x="518.56" y="-94" font-family="Arial" font-size="15.00" fill="#bfdbfe">Topology optimization engine: Importer,</text>
<text xml:space="preserve" text-anchor="start" x="557.56" y="-76" font-family="Arial" font-size="15.00" fill="#bfdbfe">DC&#45;Optimizer, AC&#45;Validator.</text>
</g>
<!-- downstream -->
<g id="node4" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1242.61,-180 920.5,-180 920.5,0 1242.61,0 1242.61,-180"/>
<text xml:space="preserve" text-anchor="start" x="940.56" y="-131" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="944.06" y="-103" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="970.06" y="-85" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="974.06" y="-67" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- client -->
<g id="node5" class="node">
<title>client</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="1693.61,-180 1355.5,-180 1355.5,0 1693.61,0 1693.61,-180"/>
<text xml:space="preserve" text-anchor="start" x="1393.56" y="-149" font-family="Arial" font-size="20.00" fill="#f0f9ff">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="1375.56" y="-121" font-family="Arial" font-size="15.00" fill="#b6ecf7">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="1451.06" y="-103" font-family="Arial" font-size="15.00" fill="#b6ecf7">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="1384.06" y="-85" font-family="Arial" font-size="15.00" fill="#b6ecf7">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="1452.56" y="-67" font-family="Arial" font-size="15.00" fill="#b6ecf7">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="1378.56" y="-49" font-family="Arial" font-size="15.00" fill="#b6ecf7">In operational use the whole run must finish</text>
</g>
<!-- optimizerdb&#45;&gt;acfastfail -->
<g id="edge1" class="edge">
<title>optimizerdb&#45;&gt;acfastfail</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M760.4,-412.29C547.53,-396.69 211.33,-357.42 133.56,-262.8 116.52,-242.07 116.35,-215.62 123.56,-189.72"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="126.01,-190.66 125.75,-182.72 121,-189.09 126.01,-190.66"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="133.56,-240 133.56,-262.8 389.56,-262.8 389.56,-240 133.56,-240"/>
<text xml:space="preserve" text-anchor="start" x="136.56" y="-259.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">open work items to select from and claim</text>
</g>
<!-- optimizerdb&#45;&gt;toop -->
<g id="edge2" class="edge">
<title>optimizerdb&#45;&gt;toop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M760.23,-363.69C714.59,-339.86 671.02,-307.14 644.56,-262.8 631.65,-241.18 627.9,-215.15 628.68,-189.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="631.29,-190.27 629.08,-182.63 626.05,-189.98 631.29,-190.27"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="644.56,-240 644.56,-262.8 802.56,-262.8 802.56,-240 644.56,-240"/>
<text xml:space="preserve" text-anchor="start" x="647.56" y="-259.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">open work items to claim</text>
</g>
<!-- optimizerdb&#45;&gt;downstream -->
<g id="edge3" class="edge">
<title>optimizerdb&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M982.35,-322.17C1000.38,-280.15 1021.31,-231.38 1039.29,-189.5"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1041.69,-190.55 1042.24,-182.63 1036.87,-188.48 1041.69,-190.55"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1014.81,-240 1014.81,-262.8 1206.81,-262.8 1206.81,-240 1014.81,-240"/>
<text xml:space="preserve" text-anchor="start" x="1017.81" y="-259.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topologies for review</text>
</g>
<!-- optimizerdb&#45;&gt;client -->
<g id="edge4" class="edge">
<title>optimizerdb&#45;&gt;client</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1112.71,-329.59C1152.56,-307.96 1194.66,-284.8 1233.56,-262.8 1277.27,-238.08 1324.29,-210.68 1367.51,-185.16"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1368.66,-187.53 1373.78,-181.45 1365.99,-183.01 1368.66,-187.53"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1273.47,-240 1273.47,-262.8 1298.47,-262.8 1298.47,-240 1273.47,-240"/>
<text xml:space="preserve" text-anchor="start" x="1276.47" y="-247.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- acfastfail&#45;&gt;optimizerdb -->
<g id="edge5" class="edge">
<title>acfastfail&#45;&gt;optimizerdb</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M358.82,-169.05C476.88,-220.46 631.22,-287.65 751.11,-339.85"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="749.71,-342.1 757.63,-342.69 751.8,-337.29 749.71,-342.1"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="559.56,-240 559.56,-262.8 584.56,-262.8 584.56,-240 559.56,-240"/>
<text xml:space="preserve" text-anchor="start" x="562.56" y="-247.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- toop&#45;&gt;optimizerdb -->
<g id="edge6" class="edge">
<title>toop&#45;&gt;optimizerdb</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M764.99,-179.61C786.24,-198.45 807.42,-219.02 825.56,-240 844.85,-262.32 863.11,-288.24 879.13,-313.38"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="876.88,-314.73 883.09,-319.68 881.32,-311.93 876.88,-314.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="843.91,-240 843.91,-262.8 868.91,-262.8 868.91,-240 843.91,-240"/>
<text xml:space="preserve" text-anchor="start" x="846.91" y="-247.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- client&#45;&gt;optimizerdb -->
<g id="edge7" class="edge">
<title>client&#45;&gt;optimizerdb</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1434.25,-179.76C1401.74,-208.69 1363.65,-239.36 1325.56,-262.8 1264.51,-300.36 1193.03,-332.64 1128.29,-358.12"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1127.64,-355.56 1121.6,-360.73 1129.54,-360.45 1127.64,-355.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1355.31,-240 1355.31,-262.8 1380.31,-262.8 1380.31,-240 1355.31,-240"/>
<text xml:space="preserve" text-anchor="start" x="1358.31" y="-247.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
</g>
</svg>`;case`postgresCommands`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1918pt" height="2760pt"
 viewBox="0.00 0.00 1918.00 2760.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 2745.31)">
<g id="clust1" class="cluster">
<title>cluster_optimizerdb</title>
<polygon fill="#29472f" stroke="#1c3021" points="732.11,-8 732.11,-2451 1238.22,-2451 1238.22,-8 732.11,-8"/>
<text xml:space="preserve" text-anchor="start" x="740.11" y="-2437.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">OPTIMIZER DATABASE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_resulttables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="772.11,-1239 772.11,-2390 1198.22,-2390 1198.22,-1239 772.11,-1239"/>
<text xml:space="preserve" text-anchor="start" x="780.11" y="-2376.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">RESULT TABLES</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_commandtables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="772.61,-48 772.61,-1199 1197.72,-1199 1197.72,-48 772.61,-48"/>
<text xml:space="preserve" text-anchor="start" x="780.61" y="-1185.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">COMMAND TABLES</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_toop</title>
<polygon fill="#3e4651" stroke="#2d333d" points="8,-1719 8,-2290 443.11,-2290 443.11,-1719 8,-1719"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-2276.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<!-- strategy -->
<g id="node1" class="node">
<title>strategy</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1150.72,-1459 819.61,-1459 819.61,-1279 1150.72,-1279 1150.72,-1459"/>
<text xml:space="preserve" text-anchor="start" x="949.67" y="-1401" font-family="Arial" font-size="20.00" fill="#f8fafc">strategy</text>
<text xml:space="preserve" text-anchor="start" x="839.67" y="-1373" font-family="Arial" font-size="15.00" fill="#cbd5e1">Group of topologies for multi&#45;timestep runs,</text>
<text xml:space="preserve" text-anchor="start" x="924.67" y="-1355" font-family="Arial" font-size="15.00" fill="#cbd5e1">not supported yet.</text>
</g>
<!-- topology -->
<g id="node2" class="node">
<title>topology</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1158.22,-2329 812.11,-2329 812.11,-2149 1158.22,-2149 1158.22,-2329"/>
<text xml:space="preserve" text-anchor="start" x="947.67" y="-2271" font-family="Arial" font-size="20.00" fill="#f8fafc">topology</text>
<text xml:space="preserve" text-anchor="start" x="855.17" y="-2243" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable definition of a topology:</text>
<text xml:space="preserve" text-anchor="start" x="832.17" y="-2225" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions, disconnections, PST setpoints, hash.</text>
</g>
<!-- topologyworkitem -->
<g id="node3" class="node">
<title>topologyworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1152.22,-2039 818.11,-2039 818.11,-1859 1152.22,-1859 1152.22,-2039"/>
<text xml:space="preserve" text-anchor="start" x="897.17" y="-1990" font-family="Arial" font-size="20.00" fill="#f8fafc">topology_work_item</text>
<text xml:space="preserve" text-anchor="start" x="838.17" y="-1962" font-family="Arial" font-size="15.00" fill="#cbd5e1">Open work: one row per topology and stage</text>
<text xml:space="preserve" text-anchor="start" x="860.17" y="-1944" font-family="Arial" font-size="15.00" fill="#cbd5e1">that still has to be evaluated, with the</text>
<text xml:space="preserve" text-anchor="start" x="850.67" y="-1926" font-family="Arial" font-size="15.00" fill="#cbd5e1">claiming worker and an attempt counter.</text>
</g>
<!-- stagetopologyresult -->
<g id="node4" class="node">
<title>stagetopologyresult</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1155.22,-1749 815.11,-1749 815.11,-1569 1155.22,-1569 1155.22,-1749"/>
<text xml:space="preserve" text-anchor="start" x="888.17" y="-1700" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_topology_result</text>
<text xml:space="preserve" text-anchor="start" x="835.17" y="-1672" font-family="Arial" font-size="15.00" fill="#cbd5e1">Archive: one row per evaluated topology and</text>
<text xml:space="preserve" text-anchor="start" x="847.67" y="-1654" font-family="Arial" font-size="15.00" fill="#cbd5e1">stage, with outcome, fitness, metrics and</text>
<text xml:space="preserve" text-anchor="start" x="921.17" y="-1636" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow reference.</text>
</g>
<!-- activeworker -->
<g id="node5" class="node">
<title>activeworker</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1149.22,-1138 821.11,-1138 821.11,-958 1149.22,-958 1149.22,-1138"/>
<text xml:space="preserve" text-anchor="start" x="923.67" y="-1080" font-family="Arial" font-size="20.00" fill="#f8fafc">active_worker</text>
<text xml:space="preserve" text-anchor="start" x="841.17" y="-1052" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per worker: last heartbeat, the job</text>
<text xml:space="preserve" text-anchor="start" x="876.67" y="-1034" font-family="Arial" font-size="15.00" fill="#cbd5e1">it is on, visible GPUs and CPUs.</text>
</g>
<!-- optimizationjob -->
<g id="node6" class="node">
<title>optimizationjob</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1157.72,-848 812.61,-848 812.61,-668 1157.72,-668 1157.72,-848"/>
<text xml:space="preserve" text-anchor="start" x="913.67" y="-790" font-family="Arial" font-size="20.00" fill="#f8fafc">optimization_job</text>
<text xml:space="preserve" text-anchor="start" x="832.67" y="-762" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable request: grid files, DC and AC</text>
<text xml:space="preserve" text-anchor="start" x="878.17" y="-744" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters, priority, created_at.</text>
</g>
<!-- stageworkitem -->
<g id="node7" class="node">
<title>stageworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1145.19,-558 825.15,-558 825.15,-378 1145.19,-378 1145.19,-558"/>
<text xml:space="preserve" text-anchor="start" x="910.17" y="-509" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_work_item</text>
<text xml:space="preserve" text-anchor="start" x="848.67" y="-481" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per job and stage, polled by the</text>
<text xml:space="preserve" text-anchor="start" x="854.67" y="-463" font-family="Arial" font-size="15.00" fill="#cbd5e1">workers: status, attempt counter, lease</text>
<text xml:space="preserve" text-anchor="start" x="963.67" y="-445" font-family="Arial" font-size="15.00" fill="#cbd5e1">expiry.</text>
</g>
<!-- stageexecutionhistory -->
<g id="node8" class="node">
<title>stageexecutionhistory</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1146.22,-268 824.11,-268 824.11,-88 1146.22,-88 1146.22,-268"/>
<text xml:space="preserve" text-anchor="start" x="878.17" y="-210" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_execution_history</text>
<text xml:space="preserve" text-anchor="start" x="844.17" y="-182" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per attempt: worker, timestamps,</text>
<text xml:space="preserve" text-anchor="start" x="914.67" y="-164" font-family="Arial" font-size="15.00" fill="#cbd5e1">KPIs, error message.</text>
</g>
<!-- dcoptimizer -->
<g id="node9" class="node">
<title>dcoptimizer</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="403.11,-2229 48,-2229 48,-2049 403.11,-2049 403.11,-2229"/>
<text xml:space="preserve" text-anchor="start" x="166.06" y="-2207.8" font-family="Arial" font-size="20.00" fill="#eef2ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="172.56" y="-2179.8" font-family="Arial" font-size="13.00" fill="#c7d2fe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="80.06" y="-2160.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="166.56" y="-2142.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="68.06" y="-2124.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="151.56" y="-2106.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="71.06" y="-2088.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acvalidator -->
<g id="node10" class="node">
<title>acvalidator</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="394.61,-1939 56.5,-1939 56.5,-1759 394.61,-1759 394.61,-1939"/>
<text xml:space="preserve" text-anchor="start" x="170.06" y="-1917.8" font-family="Arial" font-size="20.00" fill="#fbd3cb">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="124.06" y="-1889.8" font-family="Arial" font-size="13.00" fill="#f5b2a3">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="91.56" y="-1870.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="140.06" y="-1852.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="76.56" y="-1834.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="150.56" y="-1816.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="96.56" y="-1798.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">candidate it evaluates gets a full result</text>
</g>
<!-- client -->
<g id="node11" class="node">
<title>client</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1888.34,-851 1550.22,-851 1550.22,-671 1888.34,-671 1888.34,-851"/>
<text xml:space="preserve" text-anchor="start" x="1588.28" y="-820" font-family="Arial" font-size="20.00" fill="#f8fafc">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="1570.28" y="-792" font-family="Arial" font-size="15.00" fill="#cbd5e1">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="1645.78" y="-774" font-family="Arial" font-size="15.00" fill="#cbd5e1">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="1578.78" y="-756" font-family="Arial" font-size="15.00" fill="#cbd5e1">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="1647.28" y="-738" font-family="Arial" font-size="15.00" fill="#cbd5e1">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="1573.28" y="-720" font-family="Arial" font-size="15.00" fill="#cbd5e1">In operational use the whole run must finish</text>
</g>
<!-- acfastfail -->
<g id="node12" class="node">
<title>acfastfail</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="405.11,-696 46,-696 46,-516 405.11,-516 405.11,-696"/>
<text xml:space="preserve" text-anchor="start" x="137.06" y="-674.8" font-family="Arial" font-size="20.00" fill="#fbd3cb">AC fast&#45;failing stage</text>
<text xml:space="preserve" text-anchor="start" x="188.06" y="-646.8" font-family="Arial" font-size="13.00" fill="#f5b2a3">Kafka worker</text>
<text xml:space="preserve" text-anchor="start" x="66.06" y="-627.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">New stage, OptimizerType.AC_FAST_FAILING:</text>
<text xml:space="preserve" text-anchor="start" x="123.06" y="-609.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">computes only the worst&#45;k N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="97.56" y="-591.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">cases the DC stage reported, to reject</text>
<text xml:space="preserve" text-anchor="start" x="126.56" y="-573.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">topologies cheaply before the</text>
<text xml:space="preserve" text-anchor="start" x="188.56" y="-555.2" font-family="Arial" font-size="15.00" fill="#f5b2a3">full AC run.</text>
</g>
<!-- downstream -->
<g id="node13" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1880.34,-1749 1558.22,-1749 1558.22,-1569 1880.34,-1569 1880.34,-1749"/>
<text xml:space="preserve" text-anchor="start" x="1578.28" y="-1700" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="1581.78" y="-1672" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="1607.78" y="-1654" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="1611.78" y="-1636" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- topologyworkitem&#45;&gt;acvalidator -->
<g id="edge11" class="edge">
<title>topologyworkitem&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M818.25,-1927.1C695.82,-1910.94 529.45,-1888.98 404.3,-1872.46"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="405,-1869.91 397.23,-1871.53 404.32,-1875.11 405,-1869.91"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="515.61,-1913.8 515.61,-1936.6 673.61,-1936.6 673.61,-1913.8 515.61,-1913.8"/>
<text xml:space="preserve" text-anchor="start" x="518.61" y="-1933.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">open work items to claim</text>
</g>
<!-- topologyworkitem&#45;&gt;acfastfail -->
<g id="edge10" class="edge">
<title>topologyworkitem&#45;&gt;acfastfail</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M818.21,-1886.26C783.98,-1865.4 752.26,-1838.42 732.11,-1804 721.63,-1786.1 735.9,-1074.27 724.11,-1057.2 653.87,-955.52 561.79,-1032.99 465.11,-956 380.4,-888.54 314.21,-782.87 273.26,-705.28"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="275.59,-704.08 269.79,-698.64 270.94,-706.51 275.59,-704.08"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="466.61,-1057.2 466.61,-1080 722.61,-1080 722.61,-1057.2 466.61,-1057.2"/>
<text xml:space="preserve" text-anchor="start" x="469.61" y="-1077" font-family="Arial" font-size="14.00" fill="#c9c9c9">open work items to select from and claim</text>
</g>
<!-- stagetopologyresult&#45;&gt;downstream -->
<g id="edge22" class="edge">
<title>stagetopologyresult&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1155.05,-1659C1273.02,-1659 1429.93,-1659 1548.17,-1659"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1547.88,-1661.63 1555.38,-1659 1547.88,-1656.38 1547.88,-1661.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1272.22,-1659 1272.22,-1681.8 1464.22,-1681.8 1464.22,-1659 1272.22,-1659"/>
<text xml:space="preserve" text-anchor="start" x="1275.22" y="-1678.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topologies for review</text>
</g>
<!-- activeworker&#45;&gt;client -->
<g id="edge7" class="edge">
<title>activeworker&#45;&gt;client</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1149.1,-984.13C1265.3,-938.58 1421.46,-877.36 1540.81,-830.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1541.58,-833.09 1547.61,-827.91 1539.67,-828.2 1541.58,-833.09"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1293.22,-938.65 1293.22,-961.45 1443.22,-961.45 1443.22,-938.65 1293.22,-938.65"/>
<text xml:space="preserve" text-anchor="start" x="1296.22" y="-958.45" font-family="Arial" font-size="14.00" fill="#c9c9c9">which workers are alive</text>
</g>
<!-- stageworkitem&#45;&gt;client -->
<g id="edge23" class="edge">
<title>stageworkitem&#45;&gt;client</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1145.1,-461.33C1248.33,-463.16 1383.11,-477.52 1490.22,-530.2 1552.23,-560.7 1607.18,-614.96 1647.59,-663.21"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1645.48,-664.77 1652.28,-668.88 1649.52,-661.43 1645.48,-664.77"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1248.22,-530.2 1248.22,-553 1488.22,-553 1488.22,-530.2 1248.22,-530.2"/>
<text xml:space="preserve" text-anchor="start" x="1251.22" y="-550" font-family="Arial" font-size="14.00" fill="#c9c9c9">job status inferred from the work items</text>
</g>
<!-- dcoptimizer&#45;&gt;topology -->
<g id="edge2" class="edge">
<title>dcoptimizer&#45;&gt;topology</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M237.92,-2229C258.76,-2351.6 315.7,-2566.86 465.11,-2663 561.91,-2725.29 623.67,-2719.23 724.11,-2663 847.11,-2594.15 918.99,-2440.71 954.93,-2338.55"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="957.33,-2339.64 957.3,-2331.69 952.37,-2337.92 957.33,-2339.64"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="525.11,-2707.46 525.11,-2730.26 664.11,-2730.26 664.11,-2707.46 525.11,-2707.46"/>
<text xml:space="preserve" text-anchor="start" x="528.11" y="-2727.26" font-family="Arial" font-size="14.00" fill="#c9c9c9">new elites, immutable</text>
</g>
<!-- dcoptimizer&#45;&gt;topologyworkitem -->
<g id="edge3" class="edge">
<title>dcoptimizer&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M253.27,-2228.77C286.3,-2321.79 352.44,-2462.18 465.11,-2525 565.65,-2581.06 641.96,-2605.64 724.11,-2525 741.2,-2508.22 719.93,-2114.62 732.11,-2094 750.66,-2062.61 778.79,-2037.4 809.64,-2017.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="810.89,-2019.68 815.84,-2013.47 808.09,-2015.24 810.89,-2019.68"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="489.11,-2576.66 489.11,-2599.46 700.11,-2599.46 700.11,-2576.66 489.11,-2576.66"/>
<text xml:space="preserve" text-anchor="start" x="492.11" y="-2596.46" font-family="Arial" font-size="14.00" fill="#c9c9c9">work item for AC_FAST_FAILING</text>
</g>
<!-- dcoptimizer&#45;&gt;stagetopologyresult -->
<g id="edge4" class="edge">
<title>dcoptimizer&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M284.23,-2228.85C326.55,-2285.92 389.67,-2354.89 465.11,-2387 571.03,-2432.08 642.16,-2467.84 724.11,-2387 735.64,-2375.63 723.91,-1817.96 732.11,-1804 750.05,-1773.46 777.09,-1748.78 806.89,-1729.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="807.93,-1731.48 812.82,-1725.22 805.09,-1727.06 807.93,-1731.48"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="487.61,-2435.12 487.61,-2457.92 701.61,-2457.92 701.61,-2435.12 487.61,-2435.12"/>
<text xml:space="preserve" text-anchor="start" x="490.61" y="-2454.92" font-family="Arial" font-size="14.00" fill="#c9c9c9">DC fitness, metrics, worst&#45;k cases</text>
</g>
<!-- dcoptimizer&#45;&gt;activeworker -->
<g id="edge1" class="edge">
<title>dcoptimizer&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M365.06,-2228.94C476.83,-2289.17 630.39,-2342 724.11,-2249 734.34,-2238.85 725.2,-1224.63 732.11,-1212 751,-1177.49 780.62,-1149.08 813,-1126.18"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="814.11,-1128.61 818.8,-1122.19 811.13,-1124.28 814.11,-1128.61"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="503.61,-2299.43 503.61,-2322.23 685.61,-2322.23 685.61,-2299.43 503.61,-2299.43"/>
<text xml:space="preserve" text-anchor="start" x="506.61" y="-2319.23" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds</text>
</g>
<!-- dcoptimizer&#45;&gt;stageworkitem -->
<g id="edge5" class="edge">
<title>dcoptimizer&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M403.07,-2176.27C511.86,-2187.97 645.19,-2179.91 724.11,-2094 738.03,-2078.85 721.77,-630.78 732.11,-613 751.89,-579.01 782.98,-552.28 816.65,-531.54"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="817.57,-534.05 822.66,-527.95 814.88,-529.54 817.57,-534.05"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="465.11,-2180.48 465.11,-2220.08 724.11,-2220.08 724.11,-2180.48 465.11,-2180.48"/>
<text xml:space="preserve" text-anchor="start" x="468.11" y="-2217.08" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims the DC stage with SKIP LOCKED,</text>
<text xml:space="preserve" text-anchor="start" x="468.11" y="-2200.28" font-family="Arial" font-size="14.00" fill="#c9c9c9">refreshes the lease every epoch</text>
</g>
<!-- dcoptimizer&#45;&gt;stageexecutionhistory -->
<g id="edge6" class="edge">
<title>dcoptimizer&#45;&gt;stageexecutionhistory</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M403,-2076.72C544.05,-2026.54 719.08,-1963.17 724.11,-1956 737.14,-1937.43 720.71,-342.61 732.11,-323 751.63,-289.45 782.15,-262.97 815.32,-242.35"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="816.6,-244.64 821.67,-238.52 813.89,-240.14 816.6,-244.64"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="504.61,-2052.02 504.61,-2074.82 684.61,-2074.82 684.61,-2052.02 504.61,-2052.02"/>
<text xml:space="preserve" text-anchor="start" x="507.61" y="-2071.82" font-family="Arial" font-size="14.00" fill="#c9c9c9">attempt row, KPIs per epoch</text>
</g>
<!-- acvalidator&#45;&gt;topologyworkitem -->
<g id="edge18" class="edge">
<title>acvalidator&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M394.57,-1812.04C492.03,-1796.83 616.42,-1788.18 724.11,-1814.4 760.27,-1823.2 797.02,-1837.98 831.06,-1854.59"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="829.52,-1856.76 837.41,-1857.74 831.85,-1852.06 829.52,-1856.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="477.61,-1814.4 477.61,-1854 711.61,-1854 711.61,-1814.4 477.61,-1814.4"/>
<text xml:space="preserve" text-anchor="start" x="480.61" y="-1851" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims items; deletes its item with the</text>
<text xml:space="preserve" text-anchor="start" x="480.61" y="-1834.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">result</text>
</g>
<!-- acvalidator&#45;&gt;stagetopologyresult -->
<g id="edge19" class="edge">
<title>acvalidator&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M394.55,-1772.39C417.95,-1763.38 441.92,-1754.99 465.11,-1748.2 576.03,-1715.7 703.97,-1693.55 805.07,-1679.45"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="805.32,-1682.07 812.39,-1678.44 804.6,-1676.87 805.32,-1682.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="480.11,-1748.2 480.11,-1771 709.11,-1771 709.11,-1748.2 480.11,-1748.2"/>
<text xml:space="preserve" text-anchor="start" x="483.11" y="-1768" font-family="Arial" font-size="14.00" fill="#c9c9c9">full AC result with loadflow reference</text>
</g>
<!-- acvalidator&#45;&gt;activeworker -->
<g id="edge17" class="edge">
<title>acvalidator&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M321.72,-1759.07C464.55,-1623.85 718.4,-1382.53 724.11,-1370 738.69,-1338.01 714.57,-1242.47 732.11,-1212 751.48,-1178.36 780.86,-1150.42 812.8,-1127.72"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="814.25,-1129.9 818.93,-1123.48 811.27,-1125.58 814.25,-1129.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="473.61,-1616.05 473.61,-1655.65 715.61,-1655.65 715.61,-1616.05 473.61,-1616.05"/>
<text xml:space="preserve" text-anchor="start" x="476.61" y="-1652.65" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds, the job it</text>
<text xml:space="preserve" text-anchor="start" x="476.61" y="-1635.85" font-family="Arial" font-size="14.00" fill="#c9c9c9">is on</text>
</g>
<!-- acvalidator&#45;&gt;stageworkitem -->
<g id="edge20" class="edge">
<title>acvalidator&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M246.07,-1759.37C275.19,-1645.73 340.45,-1450.51 465.11,-1333.2 555.11,-1248.5 653.76,-1333.61 724.11,-1232 743.69,-1203.72 714.7,-642.66 732.11,-613 752.02,-579.09 783.15,-552.39 816.83,-531.65"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="817.75,-534.16 822.85,-528.06 815.06,-529.65 817.75,-534.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="485.11,-1333.2 485.11,-1356 704.11,-1356 704.11,-1333.2 485.11,-1333.2"/>
<text xml:space="preserve" text-anchor="start" x="488.11" y="-1353" font-family="Arial" font-size="14.00" fill="#c9c9c9">reads the shared state of the stage</text>
</g>
<!-- acvalidator&#45;&gt;stageexecutionhistory -->
<g id="edge21" class="edge">
<title>acvalidator&#45;&gt;stageexecutionhistory</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M235.86,-1759.25C254.83,-1620.56 310.22,-1353.63 465.11,-1195.2 551.51,-1106.83 653.89,-1195.7 724.11,-1094 736.28,-1076.38 721.29,-341.48 732.11,-323 751.72,-289.5 782.28,-263.05 815.45,-242.43"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="816.74,-244.72 821.81,-238.6 814.03,-240.22 816.74,-244.72"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="537.61,-1195.2 537.61,-1218 651.61,-1218 651.61,-1195.2 537.61,-1195.2"/>
<text xml:space="preserve" text-anchor="start" x="540.61" y="-1215" font-family="Arial" font-size="14.00" fill="#c9c9c9">attempt row, KPIs</text>
</g>
<!-- client&#45;&gt;optimizationjob -->
<g id="edge8" class="edge">
<title>client&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1550.34,-760.31C1436.07,-759.84 1284.94,-759.22 1167.86,-758.74"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1167.89,-756.12 1160.38,-758.71 1167.87,-761.37 1167.89,-756.12"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1254.22,-760.03 1254.22,-782.83 1482.22,-782.83 1482.22,-760.03 1254.22,-760.03"/>
<text xml:space="preserve" text-anchor="start" x="1257.22" y="-779.83" font-family="Arial" font-size="14.00" fill="#c9c9c9">start_optimization: insert the request</text>
</g>
<!-- client&#45;&gt;stageworkitem -->
<g id="edge9" class="edge">
<title>client&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1550.51,-693.86C1431.85,-646.37 1273.39,-582.95 1154.59,-535.4"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1155.77,-533.05 1147.84,-532.7 1153.82,-537.93 1155.77,-533.05"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1246.22,-666.23 1246.22,-705.83 1490.22,-705.83 1490.22,-666.23 1246.22,-666.23"/>
<text xml:space="preserve" text-anchor="start" x="1249.22" y="-702.83" font-family="Arial" font-size="14.00" fill="#c9c9c9">one TRIGGERED row per stage;</text>
<text xml:space="preserve" text-anchor="start" x="1249.22" y="-686.03" font-family="Arial" font-size="14.00" fill="#c9c9c9">cancel_optimization sets CANCELLED</text>
</g>
<!-- acfastfail&#45;&gt;topologyworkitem -->
<g id="edge13" class="edge">
<title>acfastfail&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M331.25,-695.9C371.26,-727.13 418.5,-760.44 465.11,-785 572.62,-841.65 652.85,-774.17 724.11,-872.6 739.28,-893.56 719.06,-1781.66 732.11,-1804 750.5,-1835.48 778.57,-1860.73 809.39,-1880.78"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="807.85,-1882.9 815.59,-1884.68 810.65,-1878.46 807.85,-1882.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="481.11,-872.6 481.11,-929 708.11,-929 708.11,-872.6 481.11,-872.6"/>
<text xml:space="preserve" text-anchor="start" x="484.11" y="-926" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims items; with the result deletes</text>
<text xml:space="preserve" text-anchor="start" x="484.11" y="-909.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">its item and inserts the item for AC if</text>
<text xml:space="preserve" text-anchor="start" x="484.11" y="-892.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">ACCEPTED or WARN</text>
</g>
<!-- acfastfail&#45;&gt;stagetopologyresult -->
<g id="edge14" class="edge">
<title>acfastfail&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M405.04,-646.05C539.43,-678 703.87,-721.65 724.11,-748.2 737.01,-765.12 721.36,-1495.64 732.11,-1514 750,-1544.56 777.02,-1569.27 806.81,-1589.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="805.01,-1590.99 812.74,-1592.83 807.85,-1586.57 805.01,-1590.99"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="539.61,-748.2 539.61,-771 649.61,-771 649.61,-748.2 539.61,-748.2"/>
<text xml:space="preserve" text-anchor="start" x="542.61" y="-768" font-family="Arial" font-size="14.00" fill="#c9c9c9">worst&#45;k AC result</text>
</g>
<!-- acfastfail&#45;&gt;activeworker -->
<g id="edge12" class="edge">
<title>acfastfail&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M404.96,-539.92C511.69,-513.01 642.09,-504.41 724.11,-584.4 736.79,-596.76 723.06,-887.79 732.11,-903 751.31,-935.26 780.62,-961.01 812.55,-981.31"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="810.83,-983.33 818.59,-985.03 813.59,-978.86 810.83,-983.33"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="473.61,-584.4 473.61,-624 715.61,-624 715.61,-584.4 473.61,-584.4"/>
<text xml:space="preserve" text-anchor="start" x="476.61" y="-621" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds, the job it</text>
<text xml:space="preserve" text-anchor="start" x="476.61" y="-604.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">is on</text>
</g>
<!-- acfastfail&#45;&gt;stageworkitem -->
<g id="edge15" class="edge">
<title>acfastfail&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M340,-516.06C377.66,-491.37 421.23,-467.91 465.11,-455.2 578.65,-422.32 712.07,-426.49 815.21,-438.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="814.64,-440.95 822.4,-439.22 815.26,-435.73 814.64,-440.95"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="485.11,-455.2 485.11,-478 704.11,-478 704.11,-455.2 485.11,-455.2"/>
<text xml:space="preserve" text-anchor="start" x="488.11" y="-475" font-family="Arial" font-size="14.00" fill="#c9c9c9">reads the shared state of the stage</text>
</g>
<!-- acfastfail&#45;&gt;stageexecutionhistory -->
<g id="edge16" class="edge">
<title>acfastfail&#45;&gt;stageexecutionhistory</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M305.64,-516.05C349.17,-470.68 406.22,-417.51 465.11,-380.2 573.55,-311.5 709.07,-259.69 814.45,-225.71"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="815,-228.29 821.34,-223.51 813.4,-223.29 815,-228.29"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="537.61,-380.2 537.61,-403 651.61,-403 651.61,-380.2 537.61,-380.2"/>
<text xml:space="preserve" text-anchor="start" x="540.61" y="-400" font-family="Arial" font-size="14.00" fill="#c9c9c9">attempt row, KPIs</text>
</g>
</g>
</svg>`;case`postgresCommandsTables`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1868pt" height="1091pt"
 viewBox="0.00 0.00 1868.00 1091.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1075.85)">
<g id="clust1" class="cluster">
<title>cluster_optimizerdb</title>
<polygon fill="#29472f" stroke="#1c3021" points="8,-8 8,-1052.8 1830,-1052.8 1830,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1039.35" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">OPTIMIZER DATABASE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_resulttables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="48,-48 48,-991.6 912,-991.6 912,-48 48,-48"/>
<text xml:space="preserve" text-anchor="start" x="56" y="-978.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">RESULT TABLES</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_commandtables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="952,-48 952,-991.6 1790,-991.6 1790,-48 952,-48"/>
<text xml:space="preserve" text-anchor="start" x="960" y="-978.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">COMMAND TABLES</text>
</g>
<!-- topologyworkitem -->
<g id="node1" class="node">
<title>topologyworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="872.06,-930.4 537.94,-930.4 537.94,-750.4 872.06,-750.4 872.06,-930.4"/>
<text xml:space="preserve" text-anchor="start" x="617" y="-881.4" font-family="Arial" font-size="20.00" fill="#f8fafc">topology_work_item</text>
<text xml:space="preserve" text-anchor="start" x="558" y="-853.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Open work: one row per topology and stage</text>
<text xml:space="preserve" text-anchor="start" x="580" y="-835.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">that still has to be evaluated, with the</text>
<text xml:space="preserve" text-anchor="start" x="570.5" y="-817.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">claiming worker and an attempt counter.</text>
</g>
<!-- stagetopologyresult -->
<g id="node2" class="node">
<title>stagetopologyresult</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="428.06,-930.4 87.94,-930.4 87.94,-750.4 428.06,-750.4 428.06,-930.4"/>
<text xml:space="preserve" text-anchor="start" x="161" y="-881.4" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_topology_result</text>
<text xml:space="preserve" text-anchor="start" x="108" y="-853.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Archive: one row per evaluated topology and</text>
<text xml:space="preserve" text-anchor="start" x="120.5" y="-835.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">stage, with outcome, fitness, metrics and</text>
<text xml:space="preserve" text-anchor="start" x="194" y="-817.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow reference.</text>
</g>
<!-- topology -->
<g id="node3" class="node">
<title>topology</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="689.06,-590.8 342.94,-590.8 342.94,-410.8 689.06,-410.8 689.06,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="478.5" y="-532.8" font-family="Arial" font-size="20.00" fill="#f8fafc">topology</text>
<text xml:space="preserve" text-anchor="start" x="386" y="-504.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable definition of a topology:</text>
<text xml:space="preserve" text-anchor="start" x="363" y="-486.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions, disconnections, PST setpoints, hash.</text>
</g>
<!-- strategy -->
<g id="node4" class="node">
<title>strategy</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="681.56,-268 350.44,-268 350.44,-88 681.56,-88 681.56,-268"/>
<text xml:space="preserve" text-anchor="start" x="480.5" y="-210" font-family="Arial" font-size="20.00" fill="#f8fafc">strategy</text>
<text xml:space="preserve" text-anchor="start" x="370.5" y="-182" font-family="Arial" font-size="15.00" fill="#cbd5e1">Group of topologies for multi&#45;timestep runs,</text>
<text xml:space="preserve" text-anchor="start" x="455.5" y="-164" font-family="Arial" font-size="15.00" fill="#cbd5e1">not supported yet.</text>
</g>
<!-- stageexecutionhistory -->
<g id="node5" class="node">
<title>stageexecutionhistory</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1317.06,-930.4 994.94,-930.4 994.94,-750.4 1317.06,-750.4 1317.06,-930.4"/>
<text xml:space="preserve" text-anchor="start" x="1049" y="-872.4" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_execution_history</text>
<text xml:space="preserve" text-anchor="start" x="1015" y="-844.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per attempt: worker, timestamps,</text>
<text xml:space="preserve" text-anchor="start" x="1085.5" y="-826.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">KPIs, error message.</text>
</g>
<!-- stageworkitem -->
<g id="node6" class="node">
<title>stageworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1750.02,-590.8 1429.98,-590.8 1429.98,-410.8 1750.02,-410.8 1750.02,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="1515" y="-541.8" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_work_item</text>
<text xml:space="preserve" text-anchor="start" x="1453.5" y="-513.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per job and stage, polled by the</text>
<text xml:space="preserve" text-anchor="start" x="1459.5" y="-495.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">workers: status, attempt counter, lease</text>
<text xml:space="preserve" text-anchor="start" x="1568.5" y="-477.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">expiry.</text>
</g>
<!-- activeworker -->
<g id="node7" class="node">
<title>activeworker</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1320.06,-590.8 991.94,-590.8 991.94,-410.8 1320.06,-410.8 1320.06,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="1094.5" y="-532.8" font-family="Arial" font-size="20.00" fill="#f8fafc">active_worker</text>
<text xml:space="preserve" text-anchor="start" x="1012" y="-504.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per worker: last heartbeat, the job</text>
<text xml:space="preserve" text-anchor="start" x="1047.5" y="-486.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">it is on, visible GPUs and CPUs.</text>
</g>
<!-- optimizationjob -->
<g id="node8" class="node">
<title>optimizationjob</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1337.56,-268 992.44,-268 992.44,-88 1337.56,-88 1337.56,-268"/>
<text xml:space="preserve" text-anchor="start" x="1093.5" y="-210" font-family="Arial" font-size="20.00" fill="#f8fafc">optimization_job</text>
<text xml:space="preserve" text-anchor="start" x="1012.5" y="-182" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable request: grid files, DC and AC</text>
<text xml:space="preserve" text-anchor="start" x="1058" y="-164" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters, priority, created_at.</text>
</g>
<!-- topologyworkitem&#45;&gt;topology -->
<g id="edge1" class="edge">
<title>topologyworkitem&#45;&gt;topology</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M542.04,-750.42C522.74,-733.07 505.76,-713.1 494,-690.4 479.9,-663.18 478.95,-630.85 483.45,-600.72"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="485.99,-601.41 484.66,-593.58 480.82,-600.53 485.99,-601.41"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="494,-659.2 494,-682 689,-682 689,-659.2 494,-659.2"/>
<text xml:space="preserve" text-anchor="start" x="497" y="-679" font-family="Arial" font-size="14.00" fill="#cbd5e1">topology to evaluate, per stage</text>
</g>
<!-- topologyworkitem&#45;&gt;activeworker -->
<g id="edge2" class="edge">
<title>topologyworkitem&#45;&gt;activeworker</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M796.3,-750.41C831.81,-717.59 873.62,-681.06 914,-650.8 938.94,-632.11 966.31,-613.58 993.38,-596.26"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="994.68,-598.54 999.6,-592.3 991.87,-594.11 994.68,-598.54"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="914,-650.8 914,-690.4 1129,-690.4 1129,-650.8 914,-650.8"/>
<text xml:space="preserve" text-anchor="start" x="917" y="-687.4" font-family="Arial" font-size="14.00" fill="#cbd5e1">claimed by, open again once its</text>
<text xml:space="preserve" text-anchor="start" x="917" y="-670.6" font-family="Arial" font-size="14.00" fill="#cbd5e1">heartbeat is older than the timeout</text>
</g>
<!-- stagetopologyresult&#45;&gt;topology -->
<g id="edge3" class="edge">
<title>stagetopologyresult&#45;&gt;topology</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M283.95,-750.69C296.23,-717.45 313.08,-680.56 335,-650.8 348.94,-631.87 365.8,-613.91 383.56,-597.42"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="385.2,-599.48 388.98,-592.49 381.66,-595.6 385.2,-599.48"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="335,-659.2 335,-682 455,-682 455,-659.2 335,-659.2"/>
<text xml:space="preserve" text-anchor="start" x="338" y="-679" font-family="Arial" font-size="14.00" fill="#cbd5e1">result of, per stage</text>
</g>
<!-- topology&#45;&gt;strategy -->
<g id="edge4" class="edge">
<title>topology&#45;&gt;strategy</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M516,-410.87C516,-369.67 516,-320.56 516,-278.17"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="518.63,-278.36 516,-270.86 513.38,-278.36 518.63,-278.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="516,-328 516,-350.8 598,-350.8 598,-328 516,-328"/>
<text xml:space="preserve" text-anchor="start" x="519" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">grouped into</text>
</g>
<!-- topology&#45;&gt;optimizationjob -->
<g id="edge5" class="edge">
<title>topology&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M688.64,-414.46C779.49,-369.56 890.85,-314.51 983.05,-268.94"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="984.2,-271.3 989.76,-265.62 981.88,-266.59 984.2,-271.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="859.96,-328 859.96,-350.8 1018.96,-350.8 1018.96,-328 859.96,-328"/>
<text xml:space="preserve" text-anchor="start" x="862.96" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">job, unique with the hash</text>
</g>
<!-- stageexecutionhistory&#45;&gt;stageworkitem -->
<g id="edge6" class="edge">
<title>stageexecutionhistory&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1270.31,-750.48C1331,-703.27 1405.62,-645.23 1467.49,-597.1"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1469.05,-599.21 1473.36,-592.53 1465.83,-595.07 1469.05,-599.21"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1386.01,-659.2 1386.01,-682 1514.01,-682 1514.01,-659.2 1386.01,-659.2"/>
<text xml:space="preserve" text-anchor="start" x="1389.01" y="-679" font-family="Arial" font-size="14.00" fill="#cbd5e1">one row per attempt</text>
</g>
<!-- stageexecutionhistory&#45;&gt;activeworker -->
<g id="edge7" class="edge">
<title>stageexecutionhistory&#45;&gt;activeworker</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1156,-750.7C1156,-704.74 1156,-648.47 1156,-601.07"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1158.63,-601.31 1156,-593.81 1153.38,-601.31 1158.63,-601.31"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1156,-659.2 1156,-682 1291,-682 1291,-659.2 1156,-659.2"/>
<text xml:space="preserve" text-anchor="start" x="1159" y="-679" font-family="Arial" font-size="14.00" fill="#cbd5e1">worker of the attempt</text>
</g>
<!-- stageworkitem&#45;&gt;optimizationjob -->
<g id="edge8" class="edge">
<title>stageworkitem&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1472.18,-410.87C1415.87,-368.36 1348.41,-317.44 1291.07,-274.16"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1292.67,-272.08 1285.1,-269.65 1289.5,-276.27 1292.67,-272.08"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1390.24,-328 1390.24,-350.8 1532.24,-350.8 1532.24,-328 1390.24,-328"/>
<text xml:space="preserve" text-anchor="start" x="1393.24" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">job, one row per stage</text>
</g>
<!-- activeworker&#45;&gt;optimizationjob -->
<g id="edge9" class="edge">
<title>activeworker&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1158.5,-410.87C1159.65,-369.67 1161.03,-320.56 1162.22,-278.17"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1164.84,-278.43 1162.42,-270.86 1159.59,-278.28 1164.84,-278.43"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1160.77,-328 1160.77,-350.8 1228.77,-350.8 1228.77,-328 1160.77,-328"/>
<text xml:space="preserve" text-anchor="start" x="1163.77" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">job it is on</text>
</g>
</g>
</svg>`;case`postgresCommandsLiveness`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="2778pt" height="1458pt"
 viewBox="0.00 0.00 2778.00 1458.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1442.65)">
<g id="clust1" class="cluster">
<title>cluster_toop</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="255.6,-1138.4 255.6,-1419.6 1109.6,-1419.6 1109.6,-1138.4 255.6,-1138.4"/>
<text xml:space="preserve" text-anchor="start" x="263.6" y="-1406.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_optimizerdb</title>
<polygon fill="#29472f" stroke="#1c3021" points="1156.6,-8 1156.6,-1054 2074.6,-1054 2074.6,-8 1156.6,-8"/>
<text xml:space="preserve" text-anchor="start" x="1164.6" y="-1040.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">OPTIMIZER DATABASE</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_resulttables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="1620.6,-727.6 1620.6,-992.8 2018.6,-992.8 2018.6,-727.6 1620.6,-727.6"/>
<text xml:space="preserve" text-anchor="start" x="1628.6" y="-979.35" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">RESULT TABLES</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_commandtables</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="1196.6,-48 1196.6,-652 2034.6,-652 2034.6,-48 1196.6,-48"/>
<text xml:space="preserve" text-anchor="start" x="1204.6" y="-638.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">COMMAND TABLES</text>
</g>
<!-- dcoptimizer -->
<g id="node1" class="node">
<title>dcoptimizer</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="623.65,-1358.4 295.54,-1358.4 295.54,-1178.4 623.65,-1178.4 623.65,-1358.4"/>
<text xml:space="preserve" text-anchor="start" x="400.1" y="-1310.2" font-family="Arial" font-size="20.00" fill="#eef2ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="406.6" y="-1282.2" font-family="Arial" font-size="13.00" fill="#c7d2fe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="315.6" y="-1262.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">Claims one DC stage at a time and holds it</text>
<text xml:space="preserve" text-anchor="start" x="417.6" y="-1244.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">with a lease.</text>
</g>
<!-- acvalidator -->
<g id="node2" class="node">
<title>acvalidator</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1069.65,-1358.4 733.54,-1358.4 733.54,-1178.4 1069.65,-1178.4 1069.65,-1358.4"/>
<text xml:space="preserve" text-anchor="start" x="846.1" y="-1310.2" font-family="Arial" font-size="20.00" fill="#fbd3cb">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="800.1" y="-1282.2" font-family="Arial" font-size="13.00" fill="#f5b2a3">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="753.6" y="-1262.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">Shares the AC stages of a job with other AC</text>
<text xml:space="preserve" text-anchor="start" x="778.1" y="-1244.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">workers, claims topology work items.</text>
</g>
<!-- topologyworkitem -->
<g id="node3" class="node">
<title>topologyworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1986.65,-939.6 1652.54,-939.6 1652.54,-759.6 1986.65,-759.6 1986.65,-939.6"/>
<text xml:space="preserve" text-anchor="start" x="1731.6" y="-890.6" font-family="Arial" font-size="20.00" fill="#f8fafc">topology_work_item</text>
<text xml:space="preserve" text-anchor="start" x="1672.6" y="-862.6" font-family="Arial" font-size="15.00" fill="#cbd5e1">Open work: one row per topology and stage</text>
<text xml:space="preserve" text-anchor="start" x="1694.6" y="-844.6" font-family="Arial" font-size="15.00" fill="#cbd5e1">that still has to be evaluated, with the</text>
<text xml:space="preserve" text-anchor="start" x="1685.1" y="-826.6" font-family="Arial" font-size="15.00" fill="#cbd5e1">claiming worker and an attempt counter.</text>
</g>
<!-- stageworkitem -->
<g id="node4" class="node">
<title>stageworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1556.62,-590.8 1236.58,-590.8 1236.58,-410.8 1556.62,-410.8 1556.62,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="1321.6" y="-541.8" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_work_item</text>
<text xml:space="preserve" text-anchor="start" x="1260.1" y="-513.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per job and stage, polled by the</text>
<text xml:space="preserve" text-anchor="start" x="1266.1" y="-495.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">workers: status, attempt counter, lease</text>
<text xml:space="preserve" text-anchor="start" x="1375.1" y="-477.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">expiry.</text>
</g>
<!-- activeworker -->
<g id="node5" class="node">
<title>activeworker</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1994.65,-590.8 1666.54,-590.8 1666.54,-410.8 1994.65,-410.8 1994.65,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="1769.1" y="-532.8" font-family="Arial" font-size="20.00" fill="#f8fafc">active_worker</text>
<text xml:space="preserve" text-anchor="start" x="1686.6" y="-504.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per worker: last heartbeat, the job</text>
<text xml:space="preserve" text-anchor="start" x="1722.1" y="-486.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">it is on, visible GPUs and CPUs.</text>
</g>
<!-- optimizationjob -->
<g id="node6" class="node">
<title>optimizationjob</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1748.15,-268 1403.04,-268 1403.04,-88 1748.15,-88 1748.15,-268"/>
<text xml:space="preserve" text-anchor="start" x="1504.1" y="-210" font-family="Arial" font-size="20.00" fill="#f8fafc">optimization_job</text>
<text xml:space="preserve" text-anchor="start" x="1423.1" y="-182" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable request: grid files, DC and AC</text>
<text xml:space="preserve" text-anchor="start" x="1468.6" y="-164" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters, priority, created_at.</text>
</g>
<!-- acfastfail -->
<g id="node7" class="node">
<title>acfastfail</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="2374.65,-1358.4 2038.54,-1358.4 2038.54,-1178.4 2374.65,-1178.4 2374.65,-1358.4"/>
<text xml:space="preserve" text-anchor="start" x="2118.1" y="-1310.2" font-family="Arial" font-size="20.00" fill="#fbd3cb">AC fast&#45;failing stage</text>
<text xml:space="preserve" text-anchor="start" x="2186.6" y="-1282.2" font-family="Arial" font-size="13.00" fill="#f5b2a3">Python</text>
<text xml:space="preserve" text-anchor="start" x="2058.6" y="-1262.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">Shares the AC stages of a job with other AC</text>
<text xml:space="preserve" text-anchor="start" x="2083.1" y="-1244.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">workers, claims topology work items.</text>
</g>
<!-- dcoptimizer&#45;&gt;stageworkitem -->
<g id="edge4" class="edge">
<title>dcoptimizer&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M295.73,-1223.47C210.57,-1191.4 113.99,-1138.4 63.6,-1054 -10.77,-929.45 -30.02,-838.41 63.6,-727.6 210.59,-553.61 900.85,-513.65 1226.17,-504.5"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1226.18,-507.13 1233.61,-504.3 1226.04,-501.88 1226.18,-507.13"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="63.6,-829.8 63.6,-869.4 322.6,-869.4 322.6,-829.8 63.6,-829.8"/>
<text xml:space="preserve" text-anchor="start" x="66.6" y="-866.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims the DC stage with SKIP LOCKED,</text>
<text xml:space="preserve" text-anchor="start" x="66.6" y="-849.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">refreshes the lease every epoch</text>
</g>
<!-- dcoptimizer&#45;&gt;activeworker -->
<g id="edge5" class="edge">
<title>dcoptimizer&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M465.15,-1178.55C478.03,-1055.5 520.78,-836.15 662.6,-727.6 787.75,-631.81 1211.32,-670.11 1368.6,-660 1395.56,-658.27 1585.93,-660.42 1611.6,-652 1649,-639.73 1685.8,-618.94 1718.26,-596.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1719.72,-598.86 1724.37,-592.41 1716.72,-594.55 1719.72,-598.86"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="662.6,-838.2 662.6,-861 844.6,-861 844.6,-838.2 662.6,-838.2"/>
<text xml:space="preserve" text-anchor="start" x="665.6" y="-858" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds</text>
</g>
<!-- acvalidator&#45;&gt;topologyworkitem -->
<g id="edge6" class="edge">
<title>acvalidator&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1069.33,-1233.39C1209.14,-1200.56 1410.45,-1142.63 1569.6,-1054 1622.51,-1024.53 1675.38,-983.61 1718.78,-946.2"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1720.26,-948.39 1724.2,-941.49 1716.82,-944.42 1720.26,-948.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1553.12,-1070.4 1553.12,-1110 1787.12,-1110 1787.12,-1070.4 1553.12,-1070.4"/>
<text xml:space="preserve" text-anchor="start" x="1556.12" y="-1107" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims items; deletes its item with the</text>
<text xml:space="preserve" text-anchor="start" x="1556.12" y="-1090.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">result</text>
</g>
<!-- acvalidator&#45;&gt;stageworkitem -->
<g id="edge7" class="edge">
<title>acvalidator&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M883.85,-1178.47C866.12,-1063.85 853.11,-863.21 944.6,-727.6 1009.4,-631.54 1127.68,-574.49 1227,-541.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1227.64,-544.32 1233.97,-539.52 1226.03,-539.33 1227.64,-544.32"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="944.6,-838.2 944.6,-861 1163.6,-861 1163.6,-838.2 944.6,-838.2"/>
<text xml:space="preserve" text-anchor="start" x="947.6" y="-858" font-family="Arial" font-size="14.00" fill="#c9c9c9">reads the shared state of the stage</text>
</g>
<!-- acvalidator&#45;&gt;activeworker -->
<g id="edge8" class="edge">
<title>acvalidator&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1046.58,-1178.48C1095.21,-1143.85 1147.07,-1101.12 1186.6,-1054 1291.07,-929.47 1226.53,-831.43 1351.6,-727.6 1444.19,-650.73 1502.82,-703.48 1611.6,-652 1644.07,-636.63 1677.06,-616.7 1707.26,-596.41"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1708.46,-598.77 1713.19,-592.38 1705.51,-594.42 1708.46,-598.77"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1351.6,-829.8 1351.6,-869.4 1593.6,-869.4 1593.6,-829.8 1351.6,-829.8"/>
<text xml:space="preserve" text-anchor="start" x="1354.6" y="-866.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds, the job it</text>
<text xml:space="preserve" text-anchor="start" x="1354.6" y="-849.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">is on</text>
</g>
<!-- topologyworkitem&#45;&gt;activeworker -->
<g id="edge9" class="edge">
<title>topologyworkitem&#45;&gt;activeworker</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1822.42,-759.74C1823.95,-711.31 1825.86,-651.23 1827.44,-601.24"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1830.07,-601.33 1827.68,-593.75 1824.82,-601.17 1830.07,-601.33"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1825.43,-660 1825.43,-699.6 2040.43,-699.6 2040.43,-660 1825.43,-660"/>
<text xml:space="preserve" text-anchor="start" x="1828.43" y="-696.6" font-family="Arial" font-size="14.00" fill="#cbd5e1">claimed by, open again once its</text>
<text xml:space="preserve" text-anchor="start" x="1828.43" y="-679.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">heartbeat is older than the timeout</text>
</g>
<!-- stageworkitem&#45;&gt;optimizationjob -->
<g id="edge10" class="edge">
<title>stageworkitem&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1446.22,-410.87C1469.45,-369.23 1497.19,-319.52 1521.01,-276.83"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1523.19,-278.3 1524.56,-270.47 1518.61,-275.74 1523.19,-278.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1491.46,-328 1491.46,-350.8 1633.46,-350.8 1633.46,-328 1491.46,-328"/>
<text xml:space="preserve" text-anchor="start" x="1494.46" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">job, one row per stage</text>
</g>
<!-- activeworker&#45;&gt;optimizationjob -->
<g id="edge11" class="edge">
<title>activeworker&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#64748b" stroke-width="2" d="M1759.9,-410.87C1726.6,-368.97 1686.8,-318.9 1652.72,-276.03"/>
<polygon fill="#64748b" stroke="#64748b" stroke-width="2" points="1654.81,-274.44 1648.09,-270.2 1650.7,-277.7 1654.81,-274.44"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1710.74,-328 1710.74,-350.8 1778.74,-350.8 1778.74,-328 1710.74,-328"/>
<text xml:space="preserve" text-anchor="start" x="1713.74" y="-347.8" font-family="Arial" font-size="14.00" fill="#cbd5e1">job it is on</text>
</g>
<!-- acfastfail&#45;&gt;topologyworkitem -->
<g id="edge1" class="edge">
<title>acfastfail&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2038.84,-1217.22C1986.98,-1194.43 1933.78,-1162.42 1896.6,-1118.4 1856.78,-1071.26 1837.52,-1004.6 1828.23,-949.59"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1830.86,-949.44 1827.08,-942.45 1825.68,-950.27 1830.86,-949.44"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1896.6,-1062 1896.6,-1118.4 2123.6,-1118.4 2123.6,-1062 1896.6,-1062"/>
<text xml:space="preserve" text-anchor="start" x="1899.6" y="-1115.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims items; with the result deletes</text>
<text xml:space="preserve" text-anchor="start" x="1899.6" y="-1098.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">its item and inserts the item for AC if</text>
<text xml:space="preserve" text-anchor="start" x="1899.6" y="-1081.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">ACCEPTED or WARN</text>
</g>
<!-- acfastfail&#45;&gt;stageworkitem -->
<g id="edge2" class="edge">
<title>acfastfail&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2199.42,-1178.51C2185.06,-1025.98 2147.14,-724.54 2067.6,-660 2047.92,-644.03 1635.7,-659.82 1611.6,-652 1574.42,-639.94 1538.01,-619.09 1506.02,-596.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1507.65,-594.62 1500.02,-592.41 1504.6,-598.9 1507.65,-594.62"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2185.24,-838.2 2185.24,-861 2404.24,-861 2404.24,-838.2 2185.24,-838.2"/>
<text xml:space="preserve" text-anchor="start" x="2188.24" y="-858" font-family="Arial" font-size="14.00" fill="#c9c9c9">reads the shared state of the stage</text>
</g>
<!-- acfastfail&#45;&gt;activeworker -->
<g id="edge3" class="edge">
<title>acfastfail&#45;&gt;activeworker</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2349.33,-1178.6C2390.46,-1145.07 2430.24,-1102.99 2452.6,-1054 2512.82,-922.03 2533.17,-848.24 2452.6,-727.6 2355.06,-581.56 2152.23,-529.42 2004.63,-511.11"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2005.31,-508.55 1997.55,-510.26 2004.69,-513.76 2005.31,-508.55"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2505.67,-829.8 2505.67,-869.4 2747.67,-869.4 2747.67,-829.8 2505.67,-829.8"/>
<text xml:space="preserve" text-anchor="start" x="2508.67" y="-866.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">heartbeat every few seconds, the job it</text>
<text xml:space="preserve" text-anchor="start" x="2508.67" y="-849.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">is on</text>
</g>
</g>
</svg>`;case`postgresCommandsLifecycle`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1862pt" height="3482pt"
 viewBox="0.00 0.00 1862.00 3482.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3467.15)">
<!-- client -->
<g id="node1" class="node">
<title>client</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="348.61,-3298.1 10.5,-3298.1 10.5,-3118.1 348.61,-3118.1 348.61,-3298.1"/>
<text xml:space="preserve" text-anchor="start" x="48.56" y="-3267.1" font-family="Arial" font-size="20.00" fill="#f0f9ff">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="30.56" y="-3239.1" font-family="Arial" font-size="15.00" fill="#b6ecf7">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="106.06" y="-3221.1" font-family="Arial" font-size="15.00" fill="#b6ecf7">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="39.06" y="-3203.1" font-family="Arial" font-size="15.00" fill="#b6ecf7">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="107.56" y="-3185.1" font-family="Arial" font-size="15.00" fill="#b6ecf7">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="33.56" y="-3167.1" font-family="Arial" font-size="15.00" fill="#b6ecf7">In operational use the whole run must finish</text>
</g>
<!-- optimizationjob -->
<g id="node2" class="node">
<title>optimizationjob</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1149.72,-3452.1 804.61,-3452.1 804.61,-3272.1 1149.72,-3272.1 1149.72,-3452.1"/>
<text xml:space="preserve" text-anchor="start" x="905.67" y="-3394.1" font-family="Arial" font-size="20.00" fill="#f8fafc">optimization_job</text>
<text xml:space="preserve" text-anchor="start" x="824.67" y="-3366.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable request: grid files, DC and AC</text>
<text xml:space="preserve" text-anchor="start" x="870.17" y="-3348.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters, priority, created_at.</text>
</g>
<!-- dcoptimizer -->
<g id="node3" class="node">
<title>dcoptimizer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="357.11,-1821.1 2,-1821.1 2,-1641.1 357.11,-1641.1 357.11,-1821.1"/>
<text xml:space="preserve" text-anchor="start" x="120.06" y="-1799.9" font-family="Arial" font-size="20.00" fill="#eff6ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="126.56" y="-1771.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="34.06" y="-1752.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="120.56" y="-1734.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="22.06" y="-1716.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="105.56" y="-1698.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="25.06" y="-1680.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acfastfail -->
<g id="node4" class="node">
<title>acfastfail</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="359.11,-1128.1 0,-1128.1 0,-948.1 359.11,-948.1 359.11,-1128.1"/>
<text xml:space="preserve" text-anchor="start" x="91.06" y="-1106.9" font-family="Arial" font-size="20.00" fill="#eff6ff">AC fast&#45;failing stage</text>
<text xml:space="preserve" text-anchor="start" x="142.06" y="-1078.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">Kafka worker</text>
<text xml:space="preserve" text-anchor="start" x="20.06" y="-1059.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">New stage, OptimizerType.AC_FAST_FAILING:</text>
<text xml:space="preserve" text-anchor="start" x="77.06" y="-1041.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">computes only the worst&#45;k N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="51.56" y="-1023.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">cases the DC stage reported, to reject</text>
<text xml:space="preserve" text-anchor="start" x="80.56" y="-1005.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">topologies cheaply before the</text>
<text xml:space="preserve" text-anchor="start" x="142.56" y="-987.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">full AC run.</text>
</g>
<!-- acvalidator -->
<g id="node5" class="node">
<title>acvalidator</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="348.61,-668.1 10.5,-668.1 10.5,-488.1 348.61,-488.1 348.61,-668.1"/>
<text xml:space="preserve" text-anchor="start" x="124.06" y="-646.9" font-family="Arial" font-size="20.00" fill="#eff6ff">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="78.06" y="-618.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="45.56" y="-599.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="94.06" y="-581.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="30.56" y="-563.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="104.56" y="-545.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="50.56" y="-527.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">candidate it evaluates gets a full result</text>
</g>
<!-- stageworkitem -->
<g id="node6" class="node">
<title>stageworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1137.19,-2675.1 817.15,-2675.1 817.15,-2495.1 1137.19,-2495.1 1137.19,-2675.1"/>
<text xml:space="preserve" text-anchor="start" x="902.17" y="-2626.1" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_work_item</text>
<text xml:space="preserve" text-anchor="start" x="840.67" y="-2598.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per job and stage, polled by the</text>
<text xml:space="preserve" text-anchor="start" x="846.67" y="-2580.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">workers: status, attempt counter, lease</text>
<text xml:space="preserve" text-anchor="start" x="955.67" y="-2562.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">expiry.</text>
</g>
<!-- stageexecutionhistory -->
<g id="node7" class="node">
<title>stageexecutionhistory</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1138.22,-1889.1 816.11,-1889.1 816.11,-1709.1 1138.22,-1709.1 1138.22,-1889.1"/>
<text xml:space="preserve" text-anchor="start" x="870.17" y="-1831.1" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_execution_history</text>
<text xml:space="preserve" text-anchor="start" x="836.17" y="-1803.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">One row per attempt: worker, timestamps,</text>
<text xml:space="preserve" text-anchor="start" x="906.67" y="-1785.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">KPIs, error message.</text>
</g>
<!-- topology -->
<g id="node8" class="node">
<title>topology</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1150.22,-1590.1 804.11,-1590.1 804.11,-1410.1 1150.22,-1410.1 1150.22,-1590.1"/>
<text xml:space="preserve" text-anchor="start" x="939.67" y="-1532.1" font-family="Arial" font-size="20.00" fill="#f8fafc">topology</text>
<text xml:space="preserve" text-anchor="start" x="847.17" y="-1504.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">The immutable definition of a topology:</text>
<text xml:space="preserve" text-anchor="start" x="824.17" y="-1486.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions, disconnections, PST setpoints, hash.</text>
</g>
<!-- stagetopologyresult -->
<g id="node9" class="node">
<title>stagetopologyresult</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1147.22,-1128.1 807.11,-1128.1 807.11,-948.1 1147.22,-948.1 1147.22,-1128.1"/>
<text xml:space="preserve" text-anchor="start" x="880.17" y="-1079.1" font-family="Arial" font-size="20.00" fill="#f8fafc">stage_topology_result</text>
<text xml:space="preserve" text-anchor="start" x="827.17" y="-1051.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">Archive: one row per evaluated topology and</text>
<text xml:space="preserve" text-anchor="start" x="839.67" y="-1033.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">stage, with outcome, fitness, metrics and</text>
<text xml:space="preserve" text-anchor="start" x="913.17" y="-1015.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow reference.</text>
</g>
<!-- topologyworkitem -->
<g id="node10" class="node">
<title>topologyworkitem</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1144.22,-652.1 810.11,-652.1 810.11,-472.1 1144.22,-472.1 1144.22,-652.1"/>
<text xml:space="preserve" text-anchor="start" x="889.17" y="-603.1" font-family="Arial" font-size="20.00" fill="#f8fafc">topology_work_item</text>
<text xml:space="preserve" text-anchor="start" x="830.17" y="-575.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">Open work: one row per topology and stage</text>
<text xml:space="preserve" text-anchor="start" x="852.17" y="-557.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">that still has to be evaluated, with the</text>
<text xml:space="preserve" text-anchor="start" x="842.67" y="-539.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">claiming worker and an attempt counter.</text>
</g>
<!-- loadflowstore -->
<g id="node11" class="node">
<title>loadflowstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M1158.22,-182C1158.22,-192.04 1077.07,-200.2 977.17,-200.2 877.26,-200.2 796.11,-192.04 796.11,-182 796.11,-182 796.11,-18.2 796.11,-18.2 796.11,-8.16 877.26,0 977.17,0 1077.07,0 1158.22,-8.16 1158.22,-18.2 1158.22,-18.2 1158.22,-182 1158.22,-182"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M1158.22,-182C1158.22,-171.96 1077.07,-163.8 977.17,-163.8 877.26,-163.8 796.11,-171.96 796.11,-182"/>
<text xml:space="preserve" text-anchor="start" x="886.17" y="-168.9" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="911.17" y="-140.9" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="871.17" y="-121.3" font-family="Arial" font-size="15.00" fill="#c2f0c2">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="816.17" y="-103.3" font-family="Arial" font-size="15.00" fill="#c2f0c2">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="832.67" y="-85.3" font-family="Arial" font-size="15.00" fill="#c2f0c2">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="826.67" y="-67.3" font-family="Arial" font-size="15.00" fill="#c2f0c2">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="891.67" y="-49.3" font-family="Arial" font-size="15.00" fill="#c2f0c2">topology it evaluates gets</text>
</g>
<!-- downstream -->
<g id="node12" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1832.34,-1128.1 1510.22,-1128.1 1510.22,-948.1 1832.34,-948.1 1832.34,-1128.1"/>
<text xml:space="preserve" text-anchor="start" x="1530.28" y="-1079.1" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="1533.78" y="-1051.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="1559.78" y="-1033.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="1563.78" y="-1015.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- client&#45;&gt;optimizationjob -->
<g id="edge1" class="edge">
<title>client&#45;&gt;optimizationjob</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M348.31,-3240.57C479.19,-3265.9 660.68,-3301.03 794.88,-3327.01"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="794.03,-3329.52 801.9,-3328.37 795.03,-3324.37 794.03,-3329.52"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="466.61,-3315.29 466.61,-3354.89 490.61,-3354.89 490.61,-3315.29 466.61,-3315.29"/>
<text xml:space="preserve" text-anchor="start" x="475.11" y="-3331.19" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">0</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="493.61,-3315.29 493.61,-3354.89 688.61,-3354.89 688.61,-3315.29 493.61,-3315.29"/>
<text xml:space="preserve" text-anchor="start" x="496.61" y="-3351.89" font-family="Arial" font-size="14.00" fill="#c9c9c9">start_optimization: the request,</text>
<text xml:space="preserve" text-anchor="start" x="559.61" y="-3335.09" font-family="Arial" font-size="14.00" fill="#c9c9c9">write&#45;once</text>
</g>
<!-- client&#45;&gt;stageworkitem -->
<g id="edge2" class="edge">
<title>client&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M348.54,-3190.28C489.41,-3172.38 676.8,-3140.73 736.11,-3094.1 866.35,-2991.71 929.37,-2801.66 957.09,-2684.99"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="959.59,-2685.83 958.73,-2677.93 954.47,-2684.64 959.59,-2685.83"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="442.61,-3183.31 442.61,-3216.11 466.61,-3216.11 466.61,-3183.31 442.61,-3183.31"/>
<text xml:space="preserve" text-anchor="start" x="451.11" y="-3195.81" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">1</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="469.61,-3183.31 469.61,-3216.11 712.61,-3216.11 712.61,-3183.31 469.61,-3183.31"/>
<text xml:space="preserve" text-anchor="start" x="472.61" y="-3208.11" font-family="Arial" font-size="14.00" fill="#c9c9c9">one work item per stage, TRIGGERED</text>
</g>
<!-- client&#45;&gt;stageworkitem -->
<g id="edge22" class="edge">
<title>client&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M183.72,-3108.11C194.64,-2836.46 240.54,-2095.25 419.11,-1947.3 527.6,-1857.41 620.26,-1867.12 736.11,-1947.3 916.7,-2072.28 961.61,-2351.35 972.66,-2495.34"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="181.11,-3107.73 183.44,-3115.33 186.36,-3107.94 181.11,-3107.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="446.11,-1950.3 446.11,-1983.1 477.11,-1983.1 477.11,-1950.3 446.11,-1950.3"/>
<text xml:space="preserve" text-anchor="start" x="454.11" y="-1962.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">21</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="480.11,-1950.3 480.11,-1983.1 709.11,-1983.1 709.11,-1950.3 480.11,-1950.3"/>
<text xml:space="preserve" text-anchor="start" x="483.11" y="-1975.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">all stages COMPLETED: job is done</text>
</g>
<!-- dcoptimizer&#45;&gt;stageworkitem -->
<g id="edge3" class="edge">
<title>dcoptimizer&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M184.11,-1820.82C196.28,-2076.15 245.68,-2797.54 419.11,-2940.1 527.95,-3029.56 611.68,-3006.17 736.11,-2940.1 836.98,-2886.54 903.19,-2769.59 940.04,-2684.51"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="942.35,-2685.79 942.87,-2677.87 937.52,-2683.73 942.35,-2685.79"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="440.11,-3001.75 440.11,-3034.55 464.11,-3034.55 464.11,-3001.75 440.11,-3001.75"/>
<text xml:space="preserve" text-anchor="start" x="448.61" y="-3014.25" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">2</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="467.11,-3001.75 467.11,-3034.55 715.11,-3034.55 715.11,-3001.75 467.11,-3001.75"/>
<text xml:space="preserve" text-anchor="start" x="470.11" y="-3026.55" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims DC: RUNNING, attempt 1, lease</text>
</g>
<!-- dcoptimizer&#45;&gt;stageworkitem -->
<g id="edge7" class="edge">
<title>dcoptimizer&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M181.94,-1821.03C189.51,-1977.81 228.82,-2300.67 419.11,-2471.1 475.57,-2521.67 664.74,-2552.59 807.09,-2569.24"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="806.5,-2571.81 814.25,-2570.07 807.1,-2566.6 806.5,-2571.81"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="448.11,-2562.71 448.11,-2602.31 472.11,-2602.31 472.11,-2562.71 448.11,-2562.71"/>
<text xml:space="preserve" text-anchor="start" x="456.61" y="-2578.61" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">6</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="475.11,-2562.71 475.11,-2602.31 707.11,-2602.31 707.11,-2562.71 475.11,-2562.71"/>
<text xml:space="preserve" text-anchor="start" x="478.11" y="-2599.31" font-family="Arial" font-size="14.00" fill="#c9c9c9">every epoch: refresh lease, check for</text>
<text xml:space="preserve" text-anchor="start" x="549.61" y="-2582.51" font-family="Arial" font-size="14.00" fill="#c9c9c9">CANCELLED</text>
</g>
<!-- dcoptimizer&#45;&gt;stageworkitem -->
<g id="edge19" class="edge">
<title>dcoptimizer&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M192.16,-1820.87C213.6,-1949.63 271.59,-2185.67 419.11,-2317.1 528.3,-2414.38 604.89,-2340.76 736.11,-2405.3 781.37,-2427.56 826.75,-2458.72 865.67,-2488.85"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="863.97,-2490.85 871.49,-2493.39 867.2,-2486.71 863.97,-2490.85"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="469.61,-2408.3 469.61,-2441.1 500.61,-2441.1 500.61,-2408.3 469.61,-2408.3"/>
<text xml:space="preserve" text-anchor="start" x="477.61" y="-2420.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">18</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="503.61,-2408.3 503.61,-2441.1 685.61,-2441.1 685.61,-2408.3 503.61,-2408.3"/>
<text xml:space="preserve" text-anchor="start" x="506.61" y="-2433.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">COMPLETED, lease cleared</text>
</g>
<!-- dcoptimizer&#45;&gt;stageexecutionhistory -->
<g id="edge6" class="edge">
<title>dcoptimizer&#45;&gt;stageexecutionhistory</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M356.63,-1746.15C490.57,-1757.6 673.78,-1773.26 806.19,-1784.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="805.72,-1787.17 813.42,-1785.19 806.17,-1781.94 805.72,-1787.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="465.61,-1780.1 465.61,-1819.7 489.61,-1819.7 489.61,-1780.1 465.61,-1780.1"/>
<text xml:space="preserve" text-anchor="start" x="474.11" y="-1796" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">5</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="492.61,-1780.1 492.61,-1819.7 689.61,-1819.7 689.61,-1780.1 492.61,-1780.1"/>
<text xml:space="preserve" text-anchor="start" x="495.61" y="-1816.7" font-family="Arial" font-size="14.00" fill="#c9c9c9">attempt row with worker_id and</text>
<text xml:space="preserve" text-anchor="start" x="558.11" y="-1799.9" font-family="Arial" font-size="14.00" fill="#c9c9c9">claimed_at</text>
</g>
<!-- dcoptimizer&#45;&gt;stageexecutionhistory -->
<g id="edge8" class="edge">
<title>dcoptimizer&#45;&gt;stageexecutionhistory</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M356.93,-1690.9C467.07,-1672.24 611.19,-1659.64 736.11,-1687.3 759.59,-1692.5 783.49,-1700.24 806.66,-1709.26"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="805.44,-1711.6 813.38,-1711.94 807.39,-1706.73 805.44,-1711.6"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="505.11,-1690.3 505.11,-1723.1 529.11,-1723.1 529.11,-1690.3 505.11,-1690.3"/>
<text xml:space="preserve" text-anchor="start" x="513.61" y="-1702.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">7</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="532.11,-1690.3 532.11,-1723.1 650.11,-1723.1 650.11,-1690.3 532.11,-1690.3"/>
<text xml:space="preserve" text-anchor="start" x="535.11" y="-1715.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">every epoch: KPIs</text>
</g>
<!-- dcoptimizer&#45;&gt;topology -->
<g id="edge9" class="edge">
<title>dcoptimizer&#45;&gt;topology</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M335.07,-1641.15C362.45,-1627.68 391.15,-1615.06 419.11,-1605.3 540.64,-1562.86 683.4,-1536.33 793.97,-1520.52"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="794.17,-1523.15 801.23,-1519.5 793.43,-1517.95 794.17,-1523.15"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="471.11,-1608.3 471.11,-1641.1 495.11,-1641.1 495.11,-1608.3 471.11,-1608.3"/>
<text xml:space="preserve" text-anchor="start" x="479.61" y="-1620.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">8</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="498.11,-1608.3 498.11,-1641.1 684.11,-1641.1 684.11,-1608.3 498.11,-1608.3"/>
<text xml:space="preserve" text-anchor="start" x="501.11" y="-1633.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">new elites as immutable rows</text>
</g>
<!-- dcoptimizer&#45;&gt;stagetopologyresult -->
<g id="edge10" class="edge">
<title>dcoptimizer&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M252.78,-1641.2C298,-1587.31 358.94,-1518.88 419.11,-1464.3 548.47,-1346.97 597.71,-1338.62 736.11,-1232.1 776.72,-1200.85 820.48,-1165.88 859.38,-1134.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="860.58,-1136.73 864.74,-1129.97 857.27,-1132.66 860.58,-1136.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="471.61,-1467.3 471.61,-1500.1 495.61,-1500.1 495.61,-1467.3 471.61,-1467.3"/>
<text xml:space="preserve" text-anchor="start" x="480.11" y="-1479.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">9</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="498.61,-1467.3 498.61,-1500.1 683.61,-1500.1 683.61,-1467.3 498.61,-1467.3"/>
<text xml:space="preserve" text-anchor="start" x="501.61" y="-1492.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">DC fitness and worst&#45;k cases</text>
</g>
<!-- dcoptimizer&#45;&gt;topologyworkitem -->
<g id="edge11" class="edge">
<title>dcoptimizer&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M193.77,-1641.12C216.91,-1516.23 276.63,-1291.59 419.11,-1166.3 528.93,-1069.73 630.54,-1179.3 736.11,-1078.1 798.51,-1018.28 761.98,-972.51 796.11,-893.1 830.77,-812.45 878.57,-725.4 916.21,-660.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="918.4,-662.28 919.92,-654.48 913.87,-659.63 918.4,-662.28"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="455.11,-1169.3 455.11,-1202.1 486.11,-1202.1 486.11,-1169.3 455.11,-1169.3"/>
<text xml:space="preserve" text-anchor="start" x="463.11" y="-1181.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">10</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="489.11,-1169.3 489.11,-1202.1 700.11,-1202.1 700.11,-1169.3 489.11,-1169.3"/>
<text xml:space="preserve" text-anchor="start" x="492.11" y="-1194.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">work item for AC_FAST_FAILING</text>
</g>
<!-- acfastfail&#45;&gt;stageworkitem -->
<g id="edge4" class="edge">
<title>acfastfail&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M217.96,-1127.81C260.64,-1234.8 328.47,-1420.05 359.11,-1586.1 383.34,-1717.38 322.39,-2694.09 419.11,-2786.1 521.19,-2883.2 603.69,-2834.22 736.11,-2786.1 794.43,-2764.91 848.47,-2722.65 890.45,-2682.33"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="892.17,-2684.32 895.71,-2677.21 888.5,-2680.56 892.17,-2684.32"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="478.61,-2845.03 478.61,-2877.83 502.61,-2877.83 502.61,-2845.03 478.61,-2845.03"/>
<text xml:space="preserve" text-anchor="start" x="487.11" y="-2857.53" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">3</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="505.61,-2845.03 505.61,-2877.83 676.61,-2877.83 676.61,-2845.03 505.61,-2845.03"/>
<text xml:space="preserve" text-anchor="start" x="508.61" y="-2869.83" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims AC_FAST_FAILING</text>
</g>
<!-- acfastfail&#45;&gt;stageworkitem -->
<g id="edge20" class="edge">
<title>acfastfail&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M216.23,-1127.64C257.41,-1234.85 323.98,-1420.69 359.11,-1586.1 385.9,-1712.2 335.91,-2064.63 419.11,-2163.1 513.49,-2274.81 616.19,-2167.6 736.11,-2251.3 821.61,-2310.97 887.69,-2411.22 928.68,-2486.31"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="926.24,-2487.32 932.12,-2492.67 930.86,-2484.83 926.24,-2487.32"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="514.61,-2254.3 514.61,-2287.1 545.61,-2287.1 545.61,-2254.3 514.61,-2254.3"/>
<text xml:space="preserve" text-anchor="start" x="522.61" y="-2266.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">19</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="548.61,-2254.3 548.61,-2287.1 640.61,-2287.1 640.61,-2254.3 548.61,-2254.3"/>
<text xml:space="preserve" text-anchor="start" x="551.61" y="-2279.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">COMPLETED</text>
</g>
<!-- acfastfail&#45;&gt;stagetopologyresult -->
<g id="edge13" class="edge">
<title>acfastfail&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M358.9,-1020.01C379.19,-1018.49 399.59,-1017.19 419.11,-1016.3 559.85,-1009.88 595.37,-1009.92 736.11,-1016.3 755.78,-1017.19 776.33,-1018.49 796.77,-1020.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="796.42,-1022.63 804.1,-1020.58 796.82,-1017.39 796.42,-1022.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="505.61,-1019.3 505.61,-1052.1 536.61,-1052.1 536.61,-1019.3 505.61,-1019.3"/>
<text xml:space="preserve" text-anchor="start" x="513.61" y="-1031.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">12</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="539.61,-1019.3 539.61,-1052.1 649.61,-1052.1 649.61,-1019.3 539.61,-1019.3"/>
<text xml:space="preserve" text-anchor="start" x="542.61" y="-1044.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">worst&#45;k AC result</text>
</g>
<!-- acfastfail&#45;&gt;topologyworkitem -->
<g id="edge12" class="edge">
<title>acfastfail&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M225.2,-939.16C265.16,-861.82 331.03,-757.65 419.11,-697.5 451.73,-675.23 658.11,-628.51 810.2,-596.23"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="222.9,-937.88 221.83,-945.76 227.58,-940.27 222.9,-937.88"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="431.11,-700.5 431.11,-740.1 462.11,-740.1 462.11,-700.5 431.11,-700.5"/>
<text xml:space="preserve" text-anchor="start" x="439.11" y="-716.4" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">11</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="465.11,-700.5 465.11,-740.1 724.11,-740.1 724.11,-700.5 465.11,-700.5"/>
<text xml:space="preserve" text-anchor="start" x="468.11" y="-737.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">selects among the open items and claims</text>
<text xml:space="preserve" text-anchor="start" x="579.61" y="-720.3" font-family="Arial" font-size="14.00" fill="#c9c9c9">them</text>
</g>
<!-- acfastfail&#45;&gt;topologyworkitem -->
<g id="edge14" class="edge">
<title>acfastfail&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M237.84,-948.24C275.58,-886.53 324.26,-801.86 359.11,-723.1 393.68,-644.98 349.92,-590.6 419.11,-540.5 527.67,-461.9 681.79,-474.66 800.41,-502.93"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="799.48,-505.41 807.39,-504.64 800.72,-500.31 799.48,-505.41"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="422.11,-543.5 422.11,-583.1 453.11,-583.1 453.11,-543.5 422.11,-543.5"/>
<text xml:space="preserve" text-anchor="start" x="430.11" y="-559.4" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">13</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="456.11,-543.5 456.11,-583.1 733.11,-583.1 733.11,-543.5 456.11,-543.5"/>
<text xml:space="preserve" text-anchor="start" x="459.11" y="-580.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">deletes its item; ACCEPTED or WARN: item</text>
<text xml:space="preserve" text-anchor="start" x="575.61" y="-563.3" font-family="Arial" font-size="14.00" fill="#c9c9c9">for AC</text>
</g>
<!-- acvalidator&#45;&gt;stageworkitem -->
<g id="edge5" class="edge">
<title>acvalidator&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M247.07,-667.95C287.67,-728.29 336.06,-811.43 359.11,-893.1 385.37,-986.13 349.59,-2564.93 419.11,-2632.1 519.71,-2729.29 683.59,-2702.15 807.62,-2659.87"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="808.14,-2662.47 814.37,-2657.53 806.42,-2657.51 808.14,-2662.47"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="530.61,-2701.28 530.61,-2734.08 554.61,-2734.08 554.61,-2701.28 530.61,-2701.28"/>
<text xml:space="preserve" text-anchor="start" x="539.11" y="-2713.78" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">4</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="557.61,-2701.28 557.61,-2734.08 624.61,-2734.08 624.61,-2701.28 557.61,-2701.28"/>
<text xml:space="preserve" text-anchor="start" x="560.61" y="-2726.08" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims AC</text>
</g>
<!-- acvalidator&#45;&gt;stageworkitem -->
<g id="edge21" class="edge">
<title>acvalidator&#45;&gt;stageworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M246.32,-667.77C286.75,-728.2 335.23,-811.5 359.11,-893.1 393.99,-1012.28 341.36,-1912.27 419.11,-2009.1 510.68,-2123.13 623.55,-2003.94 736.11,-2097.3 857.41,-2197.91 922.6,-2374.77 953.27,-2485.63"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="950.65,-2486.02 955.16,-2492.56 955.72,-2484.64 950.65,-2486.02"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="514.61,-2100.3 514.61,-2133.1 545.61,-2133.1 545.61,-2100.3 514.61,-2100.3"/>
<text xml:space="preserve" text-anchor="start" x="522.61" y="-2112.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">20</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="548.61,-2100.3 548.61,-2133.1 640.61,-2133.1 640.61,-2100.3 548.61,-2100.3"/>
<text xml:space="preserve" text-anchor="start" x="551.61" y="-2125.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">COMPLETED</text>
</g>
<!-- acvalidator&#45;&gt;stagetopologyresult -->
<g id="edge17" class="edge">
<title>acvalidator&#45;&gt;stagetopologyresult</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M281.49,-667.87C322.6,-702.11 371.57,-740.09 419.11,-770.1 539.94,-846.36 685.75,-915.79 797.88,-964.86"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="796.56,-967.15 804.48,-967.74 798.66,-962.33 796.56,-967.15"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="451.11,-937.37 451.11,-970.17 482.11,-970.17 482.11,-937.37 451.11,-937.37"/>
<text xml:space="preserve" text-anchor="start" x="459.11" y="-949.87" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">16</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="485.11,-937.37 485.11,-970.17 704.11,-970.17 704.11,-937.37 485.11,-937.37"/>
<text xml:space="preserve" text-anchor="start" x="488.11" y="-962.17" font-family="Arial" font-size="14.00" fill="#c9c9c9">fitness, metrics, loadflow reference</text>
</g>
<!-- acvalidator&#45;&gt;topologyworkitem -->
<g id="edge15" class="edge">
<title>acvalidator&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M270.86,-481.2C312.04,-443.96 363.9,-405.74 419.11,-386.3 552,-339.5 602.2,-342.51 736.11,-386.3 789.75,-403.84 841.33,-438.43 882.83,-472.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="269.29,-479.08 265.53,-486.09 272.83,-482.95 269.29,-479.08"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="496.11,-389.3 496.11,-422.1 527.11,-422.1 527.11,-389.3 496.11,-389.3"/>
<text xml:space="preserve" text-anchor="start" x="504.11" y="-401.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">14</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="530.11,-389.3 530.11,-422.1 659.11,-422.1 659.11,-389.3 530.11,-389.3"/>
<text xml:space="preserve" text-anchor="start" x="533.11" y="-414.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">claims an open item</text>
</g>
<!-- acvalidator&#45;&gt;topologyworkitem -->
<g id="edge18" class="edge">
<title>acvalidator&#45;&gt;topologyworkitem</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M225.38,-488.27C265.55,-418.83 331.61,-327.79 419.11,-286.3 546.42,-225.94 607.89,-227.91 736.11,-286.3 816.01,-322.68 879.63,-400.24 921.26,-463.87"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="918.82,-464.93 925.09,-469.81 923.23,-462.08 918.82,-464.93"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="511.61,-289.3 511.61,-322.1 542.61,-322.1 542.61,-289.3 511.61,-289.3"/>
<text xml:space="preserve" text-anchor="start" x="519.61" y="-301.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">17</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="545.61,-289.3 545.61,-322.1 643.61,-322.1 643.61,-289.3 545.61,-289.3"/>
<text xml:space="preserve" text-anchor="start" x="548.61" y="-314.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">deletes its item</text>
</g>
<!-- acvalidator&#45;&gt;loadflowstore -->
<g id="edge16" class="edge">
<title>acvalidator&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M207.88,-488.21C241.51,-394.04 308.23,-249.82 419.11,-176.3 526.12,-105.35 670.22,-87.19 785.05,-86.31"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="784.85,-88.93 792.34,-86.28 784.83,-83.68 784.85,-88.93"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="484.61,-179.3 484.61,-212.1 515.61,-212.1 515.61,-179.3 484.61,-179.3"/>
<text xml:space="preserve" text-anchor="start" x="492.61" y="-191.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">15</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="518.61,-179.3 518.61,-212.1 670.61,-212.1 670.61,-179.3 518.61,-179.3"/>
<text xml:space="preserve" text-anchor="start" x="521.61" y="-204.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">full AC N&#45;1 per topology</text>
</g>
<!-- stagetopologyresult&#45;&gt;downstream -->
<g id="edge23" class="edge">
<title>stagetopologyresult&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1147.03,-1038.1C1254.22,-1038.1 1392.47,-1038.1 1499.94,-1038.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1499.71,-1040.73 1507.21,-1038.1 1499.71,-1035.48 1499.71,-1040.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1221.22,-1041.1 1221.22,-1073.9 1252.22,-1073.9 1252.22,-1041.1 1221.22,-1041.1"/>
<text xml:space="preserve" text-anchor="start" x="1229.22" y="-1053.6" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">22</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1255.22,-1041.1 1255.22,-1073.9 1447.22,-1073.9 1447.22,-1041.1 1255.22,-1041.1"/>
<text xml:space="preserve" text-anchor="start" x="1258.22" y="-1065.9" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topologies for review</text>
</g>
</g>
</svg>`;default:throw Error(`Unknown viewId: `+e)}};export{e as dotSource,t as svgSource};