var e=e=>{switch(e){case`dataFlow`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=dataFlow,
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
    subgraph cluster_processedgrid {
        graph [color="#1e3524",
            fillcolor="#2c4e32",
            label=<<FONT POINT-SIZE="11" COLOR="#c2f0c2b3"><B>PROCESSED GRID FOLDER</B></FONT>>,
            likec4_depth=1,
            likec4_id=processedGrid,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        gridsnapshot [color="#2d5d39",
            fillcolor="#428a4f",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">grid.xiidm / grid.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">The normalized backend grid, written by the<BR/>importer.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.gridSnapshot",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        staticinfo [color="#2d5d39",
            fillcolor="#428a4f",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">static_information.hdf5</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">The critical asset: everything the GPU needs,<BR/>and nothing it does not.<BR/>One serialized StaticInformation -- a<BR/>SolverConfig, which is static and<BR/>part of the JIT signature, plus a</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.staticInfo",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        actionset [color="#2d5d39",
            fillcolor="#428a4f",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">action_set.json + action_set_diffs.hdf5</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">The same action space in physical terms:<BR/>station-local reconfigurations<BR/>A and disconnectable branches D, expressed as<BR/>switch positions against<BR/>the asset topology.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.actionSet",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        snapshots [color="#2d5d39",
            fillcolor="#428a4f",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimizer_snapshots/ac</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Repertoire, realized asset topologies, AC/DC<BR/>loadflow tables, SLDs, OpenRAO summaries.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.snapshots",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_kafka {
        graph [color="#462a17",
            fillcolor="#5a3620",
            label=<<FONT POINT-SIZE="11" COLOR="#f9b27cb3"><B>KAFKA</B></FONT>>,
            likec4_depth=1,
            likec4_id=kafka,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        importercommands [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">importer_commands</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">StartPreprocessingCommand, ShutdownCommand.<BR/>24 partitions.</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.importerCommands",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
        commands [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">commands</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">StartOptimizationCommand, ShutdownCommand. 4<BR/>partitions.</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.commands",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
        importerresults [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">importer_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">PreprocessingStartedResult,<BR/>PreprocessingSuccessResult, ErrorResult</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.importerResults",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
        importerheartbeat [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">importer_heartbeat</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">PreprocessHeartbeat carrying the current<BR/>PreprocessStage</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.importerHeartbeat",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
        results [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">The one shared topic. Both stages publish<BR/>topologies here and the<BR/>AC-Validator also consumes it to pick up DC<BR/>candidates.</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.results",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
        heartbeat [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">heartbeat</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Heartbeat tagged with OptimizerType.DC or<BR/>OptimizerType.AC</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.heartbeat",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
    }
    subgraph cluster_toop {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=1,
            likec4_id=toop,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        importer [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Importer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl, pandapower, JAX</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Normalizes a raw grid into a processed grid<BR/>folder and derives the<BR/>solver artifacts. Most of it depends only on<BR/>the initial grid topology,<BR/>so it can run before the forecast is</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcoptimizer [color="#4f46e5",
            fillcolor="#6366f1",
            fontcolor="#eef2ff",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c7d2fe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acvalidator [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        lfservice [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC loadflow service</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">A standalone N-1 service on its own<BR/>loadflow_commands / loadflow_results<BR/>/ loadflow_heartbeat topics. Present in the<BR/>codebase but off the main<BR/>path: dev-deployment does not create its</FONT></TD></TR></TABLE>>,
            likec4_id="toop.lfService",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    client [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>,
        likec4_id=client,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    client -> importercommands [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">StartPreprocessingCommand</FONT></TD></TR></TABLE>>,
        likec4_id="1lldnmt",
        style=dashed,
        weight=2];
    client -> commands [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">StartOptimizationCommand</FONT></TD></TR></TABLE>>,
        likec4_id=srcaa7,
        style=dashed,
        weight=2];
    unprocessedgridstore [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Unprocessed grid store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">fsspec AbstractFileSystem</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where the source grid files land before<BR/>anything touches them. The same<BR/>kind of thing as the loadflow result store --<BR/>an fsspec filesystem the<BR/>worker is handed, local disk or object</FONT></TD></TR></TABLE>>,
        likec4_id=unprocessedGridStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    unprocessedgridstore -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1tyg1gc",
        minlen=1,
        style=dashed,
        weight=2];
    gridsnapshot -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=uuc4de,
        style=dashed];
    gridsnapshot -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">base grid</FONT></TD></TR></TABLE>>,
        likec4_id=coz8k4,
        style=dashed];
    importercommands -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">consumes command</FONT></TD></TR></TABLE>>,
        likec4_id="50wkpo",
        style=dashed];
    commands -> dcoptimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">consumes command</FONT></TD></TR></TABLE>>,
        likec4_id=stkd4g,
        style=dashed];
    commands -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">consumes the same command</FONT></TD></TR></TABLE>>,
        likec4_id=coc48w,
        style=dashed];
    importer -> gridsnapshot [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1jd94bm",
        style=dashed];
    importer -> importerresults [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">PreprocessingSuccessResult</FONT></TD></TR></TABLE>>,
        likec4_id="174zseq",
        minlen=1,
        style=dashed];
    importer -> importerheartbeat [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">PreprocessHeartbeat per stage</FONT></TD></TR></TABLE>>,
        likec4_id="1tjga1c",
        minlen=1,
        style=dashed];
    importer -> staticinfo [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1p4wdy8",
        style=dashed];
    importer -> actionset [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">the same actions as physical switchings</FONT></TD></TR></TABLE>>,
        likec4_id=gcq0mi,
        style=dashed];
    loadflowstore [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>,
        likec4_id=loadflowStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    importer -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial AC N-1 results</FONT></TD></TR></TABLE>>,
        likec4_id=luns8x,
        style=dashed,
        weight=2];
    staticinfo -> dcoptimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">loaded onto the GPU at startup</FONT></TD></TR></TABLE>>,
        likec4_id="324nfq",
        style=dashed];
    actionset -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">to realize topologies</FONT></TD></TR></TABLE>>,
        likec4_id="1ebeh3w",
        style=dashed];
    loadflowstore -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial loadflow as baseline</FONT></TD></TR></TABLE>>,
        likec4_id="1v66hnb",
        style=dashed];
    dcoptimizer -> results [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">TopologyPushResult per epoch</FONT></TD></TR></TABLE>>,
        likec4_id="1vq8qm",
        style=dashed];
    dcoptimizer -> heartbeat [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">OptimizationStatsHeartbeat</FONT></TD></TR></TABLE>>,
        likec4_id="11ihrss",
        style=dashed];
    results -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">DC topologies</FONT></TD></TR></TABLE>>,
        likec4_id="1c9028e",
        style=dashed];
    downstream [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>,
        likec4_id=downstream,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    results -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">validated topologies for review</FONT></TD></TR></TABLE>>,
        likec4_id=nar1na,
        style=dashed,
        weight=2];
    acvalidator -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>,
        likec4_id="1ma18vr",
        style=dashed];
    acvalidator -> results [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC-validated Strategy</FONT></TD></TR></TABLE>>,
        likec4_id=w4m4we,
        style=dashed];
    acvalidator -> heartbeat [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">OptimizationStatsHeartbeat</FONT></TD></TR></TABLE>>,
        likec4_id=whfzik,
        style=dashed];
    acvalidator -> snapshots [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">summaries and diagrams</FONT></TD></TR></TABLE>>,
        likec4_id="4wli3j",
        style=dashed];
    snapshots -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">UCTE, DGS, OpenRAO summaries, single<BR/>line diagrams</FONT></TD></TR></TABLE>>,
        likec4_id=bh8p7r,
        style=dashed,
        weight=2];
}`;case`importerInternals`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=importerInternals,
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
    subgraph cluster_unprocessedgridstore {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>UNPROCESSED GRID STORE</B></FONT>>,
            likec4_depth=1,
            likec4_id=unprocessedGridStore,
            likec4_level=0,
            margin=32,
            style=filled
        ];
        inputnminus1 [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Input N-1 definition (optional)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">A JSON dump of Nminus1Definition with element<BR/>ids already resolved by<BR/>its producer, stored as<BR/>input_nminus1_definition.json in the data<BR/>folder or passed as</FONT></TD></TR></TABLE>>,
            likec4_id="unprocessedGridStore.inputNminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_processedgrid {
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>PROCESSED GRID FOLDER</B></FONT>>,
            likec4_depth=2,
            likec4_id=processedGrid,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_staticinfo {
            graph [color="#2d333d",
                fillcolor="#3e4651",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>STATIC_INFORMATION.HDF5</B></FONT>>,
                likec4_depth=1,
                likec4_id="processedGrid.staticInfo",
                likec4_level=1,
                margin=32,
                style=filled
            ];
            branchactionset [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">BranchActionSet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">What the DC-Optimizer actually samples from<BR/>-- a different asset from<BR/>action_set.json, in a different format and a<BR/>different file.<BR/>Padded boolean arrays (branch_actions,</FONT></TD></TR></TABLE>>,
                likec4_id="processedGrid.staticInfo.branchActionSet",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        assettopomaster [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">initial_topology/asset_topology_master_data.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">A serialized MasterAssetTopology, and the<BR/>only form of the asset<BR/>topology that gets a file of its own. Written<BR/>by the importer, read back<BR/>at the start of DC preprocessing. The runtime</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.assetTopoMaster",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        masks [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">masks/*.npy</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">~35 boolean and weight masks per asset class:<BR/>which branches count for<BR/>N-1, which are disconnectable, overload<BR/>weights, TSO/DSO borders,<BR/>blacklists.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.masks",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        gridsnapshot [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">grid.xiidm / grid.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The normalized backend grid, written by the<BR/>importer.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.gridSnapshot",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        nminus1 [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">nminus1_definition.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The grid-validated contingency set, written<BR/>by the importer from the<BR/>input N-1 definition (journey A) or the<BR/>network masks (journey B). It<BR/>is a separate file from the input N-1</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.nminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        actionset [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">action_set.json + action_set_diffs.hdf5</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The same action space in physical terms:<BR/>station-local reconfigurations<BR/>A and disconnectable branches D, expressed as<BR/>switch positions against<BR/>the asset topology.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.actionSet",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcnminus1 [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">dc_nminus1_definition.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The contingencies DC computes, in solver<BR/>order. DC preprocessing also<BR/>writes them as the N-1 definition if the<BR/>folder has none.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.dcNminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_importer {
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>IMPORTER</B></FONT>>,
            likec4_depth=2,
            likec4_id="toop.importer",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_importstage {
            graph [color="#1b3d88",
                fillcolor="#194b9e",
                label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>IMPORTER PREPROCESSING</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.importer.importStage",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            loadgrid [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Load and merge</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Parse UCTE/CGMES/PowerFactory. Dominates<BR/>importer runtime on CGMES.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.loadGrid",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            inputvalidation [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">load_nminus1_definition_for_network</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Journey A, stage 1. Validates the input N-1<BR/>definition against the<BR/>grid before its three-winding transformers<BR/>are converted, because<BR/>the input references them by their original</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.inputValidation",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            trafoconversion [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Convert 3W transformers</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Replaces every three-winding transformer &lt;id&gt;<BR/>in the network by the<BR/>two-winding legs &lt;id&gt;-Leg1/2/3 around a star<BR/>voltage level.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.trafoConversion",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            networkreduction [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">reduce_network_based_on_area_settings</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Optional. Keeps the voltage levels within a<BR/>range of the view,<BR/>control and N-1 area voltage levels and<BR/>removes the rest. In<BR/>journey A the voltage levels of all input N-1</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.networkReduction",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            whitelists [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Apply whitelists</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Apply the CB / black- and whitelists that<BR/>scope the switchable area.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.whitelists",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            convergingparams [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">find_converging_loadflow_params</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Sweep loadflow parameters and voltage init<BR/>methods until the base<BR/>case converges. Some grid files do not<BR/>converge on defaults.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.convergingParams",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            networkmasks [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_network_masks</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Build the per-asset masks from the area<BR/>settings: what is<BR/>monitored, outaged, switchable and<BR/>disconnectable. Computed in both<BR/>journeys; only journey B derives its N-1</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.networkMasks",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            nminus1definition [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">create_nminus1_definition</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Builds the N-1 definition saved next to the<BR/>processed grid, in the<BR/>same order for both journeys: source, area<BR/>settings, conversion.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.nminus1Definition",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            securityanalysis [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">import_grid_with_security_analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">The only AC N-1 of the import, with the<BR/>converging loadflow<BR/>parameters saved next to the grid. Threads<BR/>and batch size come from<BR/>n_contingency_worker and</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.securityAnalysis",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            artificiallimits [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">attach_artificial_limits_to_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Writes the artificial limits derived from the<BR/>worst currents onto<BR/>the network and attaches them to the AC N-1<BR/>results. Currents do not<BR/>depend on limits, so no second analysis is</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.artificialLimits",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            topologymodel [height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_master_asset_topology_artifact</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Extraction. Dispatches on the importer<BR/>data_type and hands off to<BR/>one of the readers below -- which is the<BR/>whole reason the rest of<BR/>the engine never has to know which framework</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.topologyModel",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_dcpreprocess {
            graph [color="#2a2490",
                fillcolor="#2225aa",
                label=<<FONT POINT-SIZE="11" COLOR="#c7d2feb3"><B>LOAD_GRID (DC PREPROCESSING)</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.importer.dcPreprocess",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            materialize [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_runtime_asset_topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Transition 1. Reads the master data back and<BR/>materializes it against<BR/>the loaded network: structure from the<BR/>importer artifact, switch and<BR/>busbar states from the grid file.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.materialize",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            bridges [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">compute_bridging_branches</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Tarjan bridge finding. A branch whose removal<BR/>islands the grid,<BR/>under N-0 or any contingency, cannot be<BR/>disconnected.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.bridges",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            relevantnodes [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">filter_relevant_nodes</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Drop substations that are not worth<BR/>switching: too few branches,<BR/>no assets, or double connections.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.relevantNodes",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            factors [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">compute PTDF / PSDF</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">The reference PTDF matrix, solved once. Every<BR/>topology the optimizer<BR/>later evaluates is a low-rank update of it<BR/>rather than a refactorization.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.factors",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            reduce [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">reduce node / branch dimension</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Collapse nodes that never change into a<BR/>single static column and<BR/>drop branches that are neither monitored,<BR/>outaged nor switched.<BR/>Directly shrinks the PTDF the GPU has to</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.reduce",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            nminus2filter [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">filter_disconnectable_branches_nminus2</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Exclude branches that island the grid in<BR/>combination with a contingency.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.nminus2Filter",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            simplify [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">simplify_asset_topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Transition 2. Projects each relevant station<BR/>onto one electrical<BR/>node at a time and runs<BR/>prepare_for_separation_set on the slice.<BR/>Stations that survive become the simplified</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.simplify",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            electricalactions [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">compute_electrical_actions</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Stage one of action set enumeration: every<BR/>electrically distinct<BR/>two-node split of a station, filtered for<BR/>islanding and<BR/>connectivity, clipped if a station exceeds</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.electricalActions",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            stationrealisations [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">enumerate_station_realisations</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Stage two: map each electrical split onto a<BR/>reachable node-breaker<BR/>realization and precompute its reassignment<BR/>distance. Splits with<BR/>no valid realization are discarded.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.stationRealisations",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            bboutage [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">preprocess_bb_outage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Optional busbar outage contingencies, used by<BR/>the do-not-make-it-worse criterion.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.dcPreprocess.bbOutage",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        initialloadflow [color="#853A2D",
            fillcolor="#AC4D39",
            fontcolor="#FBD3CB",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">store_initial_ca_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Stores the AC N-1 of the import, with<BR/>permanent and artificial branch<BR/>limits, as the reference every proposed<BR/>topology is later compared<BR/>against. Runs before the DC preprocessing and</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.initialLoadflow",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    assettopomaster -> materialize [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">canonical structure</FONT></TD></TR></TABLE>>,
        likec4_id=e0ty2w,
        minlen=1,
        style=dashed];
    inputnminus1 -> inputvalidation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: input N-1 definition</FONT></TD></TR></TABLE>>,
        likec4_id="16v0zir",
        minlen=1,
        style=dashed];
    loadgrid -> inputvalidation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: unconverted network</FONT></TD></TR></TABLE>>,
        likec4_id="3o2azt",
        style=dashed,
        weight=5];
    loadgrid -> trafoconversion [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">parsed network</FONT></TD></TR></TABLE>>,
        likec4_id="1ezvvz6",
        style=dashed];
    inputvalidation -> networkreduction [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: voltage levels to keep</FONT></TD></TR></TABLE>>,
        likec4_id=leeldc,
        style=dashed,
        weight=5];
    inputvalidation -> nminus1definition [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: grid-validated input<BR/>definition</FONT></TD></TR></TABLE>>,
        likec4_id="4a71dd",
        style=dashed,
        weight=5];
    trafoconversion -> networkreduction [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">converted network</FONT></TD></TR></TABLE>>,
        likec4_id="6hs6fv",
        style=dashed];
    networkreduction -> whitelists [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reduced network</FONT></TD></TR></TABLE>>,
        likec4_id=nhbijf,
        style=dashed];
    whitelists -> convergingparams [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">scoped network</FONT></TD></TR></TABLE>>,
        likec4_id=jtf58k,
        style=dashed];
    convergingparams -> networkmasks [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">converging parameters</FONT></TD></TR></TABLE>>,
        likec4_id="8c7q7f",
        style=dashed];
    networkmasks -> nminus1definition [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey B: masks</FONT></TD></TR></TABLE>>,
        likec4_id="1p0acvp",
        style=dashed,
        weight=5];
    networkmasks -> topologymodel [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">masks</FONT></TD></TR></TABLE>>,
        likec4_id=dat5vp,
        style=dashed,
        weight=5];
    nminus1definition -> securityanalysis [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">N-1 definition</FONT></TD></TR></TABLE>>,
        likec4_id="1yrfg7c",
        style=dashed,
        weight=5];
    nminus1definition -> nminus1 [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=gowz3u,
        minlen=1,
        style=dashed];
    securityanalysis -> artificiallimits [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC N-1 results</FONT></TD></TR></TABLE>>,
        likec4_id="99v0ku",
        style=dashed,
        weight=5];
    artificiallimits -> initialloadflow [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC N-1 results incl. artificial limits</FONT></TD></TR></TABLE>>,
        likec4_id="6x7ruf",
        style=dashed,
        weight=4];
    artificiallimits -> gridsnapshot [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">artificial limits</FONT></TD></TR></TABLE>>,
        likec4_id="1dtwt17",
        style=dashed];
    loadflowstore [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>,
        likec4_id=loadflowStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    initialloadflow -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial AC N-1 results</FONT></TD></TR></TABLE>>,
        likec4_id=sprnet,
        minlen=1,
        style=dashed];
    gridsnapshot -> topologymodel [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">normalized network</FONT></TD></TR></TABLE>>,
        likec4_id="1jt8oy5",
        style=dashed];
    gridsnapshot -> materialize [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">live switch, coupler and busbar state</FONT></TD></TR></TABLE>>,
        likec4_id=sejkvo,
        style=dashed];
    topologymodel -> masks [arrowhead=normal,
        likec4_id="53uecm",
        ltail=cluster_importstage,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">per-asset masks</FONT></TD></TR></TABLE>>];
    topologymodel -> materialize [arrowhead=normal,
        lhead=cluster_dcpreprocess,
        likec4_id="1xhrsbc",
        ltail=cluster_importstage,
        style=dashed,
        weight=5,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">ImportResult</FONT></TD></TR></TABLE>>];
    materialize -> bridges [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">runtime topology on NetworkData</FONT></TD></TR></TABLE>>,
        likec4_id="10cblsr",
        style=dashed,
        weight=5];
    bridges -> relevantnodes [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">bridge flags</FONT></TD></TR></TABLE>>,
        likec4_id="1b5i24m",
        style=dashed];
    relevantnodes -> factors [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">switchable subset</FONT></TD></TR></TABLE>>,
        likec4_id="1t6watw",
        style=dashed];
    factors -> reduce [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">PTDF / PSDF</FONT></TD></TR></TABLE>>,
        likec4_id=gosez8,
        style=dashed];
    reduce -> nminus2filter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reduced dimensions</FONT></TD></TR></TABLE>>,
        likec4_id="1cr209m",
        style=dashed];
    nminus2filter -> simplify [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">final branch and injection ordering</FONT></TD></TR></TABLE>>,
        likec4_id="192u47b",
        style=dashed];
    nminus2filter -> electricalactions [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">disconnectable set D</FONT></TD></TR></TABLE>>,
        likec4_id="94gv4p",
        style=dashed];
    simplify -> electricalactions [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reduced stations to enumerate in</FONT></TD></TR></TABLE>>,
        likec4_id=as6nsk,
        style=dashed];
    electricalactions -> stationrealisations [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">electrical splits</FONT></TD></TR></TABLE>>,
        likec4_id=rhzugt,
        style=dashed];
    stationrealisations -> bboutage [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">action set A</FONT></TD></TR></TABLE>>,
        likec4_id="1z14037",
        style=dashed];
    bboutage -> branchactionset [arrowhead=normal,
        likec4_id=wxl5z7,
        ltail=cluster_dcpreprocess,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">padded action arrays for the GPU</FONT></TD></TR></TABLE>>];
    bboutage -> actionset [arrowhead=normal,
        likec4_id="1tcfu8f",
        ltail=cluster_dcpreprocess,
        minlen=1,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">the same actions as physical switchings</FONT></TD></TR></TABLE>>];
    bboutage -> dcnminus1 [arrowhead=normal,
        likec4_id=rin2kn,
        ltail=cluster_dcpreprocess,
        minlen=1,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">contingencies DC computes</FONT></TD></TR></TABLE>>];
}`;case`nminus1Journeys`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=nminus1Journeys,
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
    subgraph cluster_unprocessedgridstore {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>UNPROCESSED GRID STORE</B></FONT>>,
            likec4_depth=1,
            likec4_id=unprocessedGridStore,
            likec4_level=0,
            margin=32,
            style=filled
        ];
        inputnminus1 [color="#0369a1",
            fillcolor="#0284c7",
            fontcolor="#f0f9ff",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Input N-1 definition (optional)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">A JSON dump of Nminus1Definition with element<BR/>ids already resolved by<BR/>its producer, stored as<BR/>input_nminus1_definition.json in the data<BR/>folder or passed as</FONT></TD></TR></TABLE>>,
            likec4_id="unprocessedGridStore.inputNminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_importstage {
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>IMPORTER PREPROCESSING</B></FONT>>,
            likec4_depth=2,
            likec4_id="toop.importer.importStage",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_nminus1definition {
            graph [color="#2d333d",
                fillcolor="#3e4651",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>CREATE_NMINUS1_DEFINITION</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.importer.importStage.nminus1Definition",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            maskdefinition [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                group="toop.importer.importStage.nminus1Definition",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">create_nminus1_definition_from_masks</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Journey B source. Area settings are applied<BR/>through the masks.<BR/>Three-winding transformers still carry their<BR/>original id here.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.nminus1Definition.maskDefinition",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            converttrafo3w [group="toop.importer.importStage.nminus1Definition",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">convert_three_winding_transformers_in_nminus1_definition</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Both journeys. Replaces each three-winding<BR/>transformer in<BR/>contingencies and monitored elements by its<BR/>three legs, and copies<BR/>SPPS conditions and actions on it once per</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.nminus1Definition.convertTrafo3w",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            finalcheck [color="#0369a1",
                fillcolor="#0284c7",
                fontcolor="#f0f9ff",
                group="toop.importer.importStage.nminus1Definition",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">filter_nminus1_definition_to_network</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Journey A only. Checks the converted input<BR/>definition against the<BR/>processed grid again, dropping elements later<BR/>preprocessing steps<BR/>removed.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.nminus1Definition.finalCheck",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        loadgrid [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Load and merge</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Parse UCTE/CGMES/PowerFactory. Dominates<BR/>importer runtime on CGMES.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.loadGrid",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        inputvalidation [color="#0369a1",
            fillcolor="#0284c7",
            fontcolor="#f0f9ff",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">load_nminus1_definition_for_network</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Journey A, stage 1. Validates the input N-1<BR/>definition against the<BR/>grid before its three-winding transformers<BR/>are converted, because<BR/>the input references them by their original</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.inputValidation",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        trafoconversion [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Convert 3W transformers</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Replaces every three-winding transformer &lt;id&gt;<BR/>in the network by the<BR/>two-winding legs &lt;id&gt;-Leg1/2/3 around a star<BR/>voltage level.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.trafoConversion",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        networkreduction [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">reduce_network_based_on_area_settings</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Optional. Keeps the voltage levels within a<BR/>range of the view,<BR/>control and N-1 area voltage levels and<BR/>removes the rest. In<BR/>journey A the voltage levels of all input N-1</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.networkReduction",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        whitelists [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Apply whitelists</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Apply the CB / black- and whitelists that<BR/>scope the switchable area.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.whitelists",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        convergingparams [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">find_converging_loadflow_params</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Sweep loadflow parameters and voltage init<BR/>methods until the base<BR/>case converges. Some grid files do not<BR/>converge on defaults.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.convergingParams",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        networkmasks [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_network_masks</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Build the per-asset masks from the area<BR/>settings: what is<BR/>monitored, outaged, switchable and<BR/>disconnectable. Computed in both<BR/>journeys; only journey B derives its N-1</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.networkMasks",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        securityanalysis [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">import_grid_with_security_analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The only AC N-1 of the import, with the<BR/>converging loadflow<BR/>parameters saved next to the grid. Threads<BR/>and batch size come from<BR/>n_contingency_worker and</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.securityAnalysis",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_processedgrid {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>PROCESSED GRID FOLDER</B></FONT>>,
            likec4_depth=1,
            likec4_id=processedGrid,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        nminus1 [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">nminus1_definition.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The grid-validated contingency set, written<BR/>by the importer from the<BR/>input N-1 definition (journey A) or the<BR/>network masks (journey B). It<BR/>is a separate file from the input N-1</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.nminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcnminus1 [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">dc_nminus1_definition.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The contingencies DC computes, in solver<BR/>order. DC preprocessing also<BR/>writes them as the N-1 definition if the<BR/>folder has none.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.dcNminus1",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    inputnminus1 -> inputvalidation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: input N-1 definition</FONT></TD></TR></TABLE>>,
        likec4_id="16v0zir",
        minlen=1,
        style=dashed];
    loadgrid -> inputvalidation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: unconverted network</FONT></TD></TR></TABLE>>,
        likec4_id="3o2azt",
        style=dashed,
        weight=5];
    loadgrid -> trafoconversion [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">parsed network</FONT></TD></TR></TABLE>>,
        likec4_id="1ezvvz6",
        style=dashed];
    dcpreprocess [color="#4f46e5",
        fillcolor="#6366f1",
        fontcolor="#eef2ff",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">load_grid (DC preprocessing)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c7d2fe">JAX, NumPy</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">A chain of NetworkData transforms that<BR/>shrinks the problem and<BR/>enumerates the action space. This is where<BR/>the linearized DC<BR/>representation is built.</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.dcPreprocess",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    dcpreprocess -> dcnminus1 [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">contingencies DC computes</FONT></TD></TR></TABLE>>,
        likec4_id=rin2kn,
        minlen=1,
        style=dashed];
    inputvalidation -> networkreduction [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: voltage levels to keep</FONT></TD></TR></TABLE>>,
        likec4_id=leeldc,
        style=dashed,
        weight=5];
    inputvalidation -> converttrafo3w [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: grid-validated input<BR/>definition</FONT></TD></TR></TABLE>>,
        likec4_id="1c2g6le",
        style=dashed,
        weight=5];
    trafoconversion -> networkreduction [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">converted network</FONT></TD></TR></TABLE>>,
        likec4_id="6hs6fv",
        style=dashed];
    networkreduction -> whitelists [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reduced network</FONT></TD></TR></TABLE>>,
        likec4_id=nhbijf,
        style=dashed];
    whitelists -> convergingparams [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">scoped network</FONT></TD></TR></TABLE>>,
        likec4_id=jtf58k,
        style=dashed];
    convergingparams -> networkmasks [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">converging parameters</FONT></TD></TR></TABLE>>,
        likec4_id="8c7q7f",
        style=dashed,
        weight=2];
    networkmasks -> maskdefinition [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey B: masks</FONT></TD></TR></TABLE>>,
        likec4_id=r5duuy,
        style=dashed];
    maskdefinition -> converttrafo3w [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey B: mask-derived definition</FONT></TD></TR></TABLE>>,
        likec4_id=i2yweu,
        style=dashed,
        weight=6];
    securityanalysis -> dcpreprocess [arrowhead=normal,
        likec4_id="1xhrsbc",
        ltail=cluster_importstage,
        style=dashed,
        weight=4,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">ImportResult</FONT></TD></TR></TABLE>>];
    converttrafo3w -> finalcheck [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: converted definition</FONT></TD></TR></TABLE>>,
        likec4_id="9zxf8d",
        style=dashed,
        weight=6];
    converttrafo3w -> nminus1 [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey B: converted definition</FONT></TD></TR></TABLE>>,
        likec4_id=c8ok6h,
        style=dashed];
    finalcheck -> securityanalysis [arrowhead=normal,
        likec4_id="1yrfg7c",
        ltail=cluster_nminus1definition,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">N-1 definition</FONT></TD></TR></TABLE>>];
    finalcheck -> nminus1 [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">journey A: grid-validated definition</FONT></TD></TR></TABLE>>,
        likec4_id="1is3x1a",
        style=dashed];
    initialloadflow [color="#853A2D",
        fillcolor="#AC4D39",
        fontcolor="#FBD3CB",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">store_initial_ca_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Stores the AC N-1 of the import, with<BR/>permanent and artificial branch<BR/>limits, as the reference every proposed<BR/>topology is later compared<BR/>against. Runs before the DC preprocessing and</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.initialLoadflow",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
}`;case`dcWorkerInternals`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=dcWorkerInternals,
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
    subgraph cluster_staticinfo {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>STATIC_INFORMATION.HDF5</B></FONT>>,
            likec4_depth=1,
            likec4_id="processedGrid.staticInfo",
            likec4_level=0,
            margin=32,
            style=filled
        ];
        branchactionset [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">BranchActionSet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">What the DC-Optimizer actually samples from<BR/>-- a different asset from<BR/>action_set.json, in a different format and a<BR/>different file.<BR/>Padded boolean arrays (branch_actions,</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.staticInfo.branchActionSet",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_dcoptimizer {
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>DC-OPTIMIZER</B></FONT>>,
            likec4_depth=2,
            likec4_id="toop.dcOptimizer",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_dcsolver {
            graph [color="#2a2490",
                fillcolor="#2225aa",
                label=<<FONT POINT-SIZE="11" COLOR="#c7d2feb3"><B>GPU DC LOADFLOW SOLVER</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.dcOptimizer.dcSolver",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            bsdfstage [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                group="toop.dcOptimizer.dcSolver",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">compute_bsdf_lodf_static_flows</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Everything that changes the PTDF, in one pass<BR/>per branch topology:<BR/>BSDF for each busbar split, MODF for the<BR/>remedial disconnections,<BR/>the LODF matrix, and the static flows.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.dcOptimizer.dcSolver.bsdfStage",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            n0stage [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                group="toop.dcOptimizer.dcSolver",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">N-0 flows</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Nodal injections against the already-updated<BR/>PTDF, plus the<BR/>cross-coupler flows across each split,<BR/>corrected for disconnections<BR/>and PST taps. Cheap, because the PTDF is</FONT></TD></TR></TABLE>>,
                likec4_id="toop.dcOptimizer.dcSolver.n0Stage",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            n1stage [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                group="toop.dcOptimizer.dcSolver",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Contingency analysis (N-1)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">The N-1 matrix from the LODF and multi-outage<BR/>factors, plus busbar<BR/>outage and injection outage cases. Runs over<BR/>the whole batch.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.dcOptimizer.dcSolver.n1Stage",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            resultextraction [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                group="toop.dcOptimizer.dcSolver",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Result aggregation and sparsification</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">The full N-1 matrix is far too large to keep<BR/>per topology, so only<BR/>the worst entries survive: a top-k over the<BR/>flattened matrix for<BR/>storage, and a per-case worst-k that tells</FONT></TD></TR></TABLE>>,
                likec4_id="toop.dcOptimizer.dcSolver.resultExtraction",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        scoring [group="toop.dcOptimizer",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Scoring</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Turns raw flows into the metric vector --<BR/>overload energy, critical<BR/>branch counts under N-0 and N-1, busbar<BR/>outage penalty -- and<BR/>aggregates it into the scalar fitness that is</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer.scoring",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        repertoire [color="#2d5d39",
            fillcolor="#428a4f",
            fontcolor="#f8fafc",
            group="toop.dcOptimizer",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Discrete MAP-Elites repertoire</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Cells indexed by the switching-distance<BR/>descriptors: disconnections,<BR/>split substations and reassignment distance.<BR/>Each cell keeps its own<BR/>elites (cell_depth), so a conservative</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer.repertoire",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        mutation [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            group="toop.dcOptimizer",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Mutation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Per genome, a Poisson-sampled number of<BR/>substation mutations followed<BR/>by one disconnection mutation, each drawing<BR/>ADD / CHANGE / REMOVE /<BR/>IDENTITY. Feasibility is enforced while</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer.mutation",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        crossover [color="#7E451D",
            fillcolor="#A35829",
            fontcolor="#FFE0C2",
            group="toop.dcOptimizer",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Crossover</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Builds an offspring by sampling actions and<BR/>disconnections from the<BR/>union of two parents, biased toward the first<BR/>parent.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer.crossover",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        pusher [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            group="toop.dcOptimizer",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Epoch result push</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">How topologies leave the DC stage. At the end<BR/>of each epoch the new<BR/>elites are pulled off the GPU, converted to<BR/>TopologyPushResult<BR/>messages and produced to the \`results\` topic</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer.pusher",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_kafka {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>KAFKA</B></FONT>>,
            likec4_depth=1,
            likec4_id=kafka,
            likec4_level=0,
            margin=32,
            style=filled
        ];
        results [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The one shared topic. Both stages publish<BR/>topologies here and the<BR/>AC-Validator also consumes it to pick up DC<BR/>candidates.</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.results",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
    }
    branchactionset -> scoring [arrowhead=normal,
        lhead=cluster_dcoptimizer,
        likec4_id="1329ke8",
        minlen=1,
        style=dashed,
        weight=4,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">sampling space -- indices into these<BR/>arrays</FONT></TD></TR></TABLE>>];
    scoring -> repertoire [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">fitness and descriptors, sorted insert</FONT></TD></TR></TABLE>>,
        likec4_id="1g9ky2a",
        style=dashed];
    repertoire -> mutation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">sampled elites</FONT></TD></TR></TABLE>>,
        likec4_id="2ye8q6",
        style=dashed];
    repertoire -> crossover [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">sampled pairs</FONT></TD></TR></TABLE>>,
        likec4_id="1k6dmmn",
        style=dashed];
    repertoire -> pusher [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">new elites</FONT></TD></TR></TABLE>>,
        likec4_id="1fjij92",
        style=dashed,
        weight=4];
    mutation -> bsdfstage [arrowhead=normal,
        lhead=cluster_dcsolver,
        likec4_id=yk80tp,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">candidate batch</FONT></TD></TR></TABLE>>];
    crossover -> bsdfstage [arrowhead=normal,
        lhead=cluster_dcsolver,
        likec4_id=khjb70,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">candidate batch</FONT></TD></TR></TABLE>>];
    pusher -> results [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">TopologyPushResult per epoch</FONT></TD></TR></TABLE>>,
        likec4_id="1vvbmvt",
        minlen=1,
        style=dashed];
    bsdfstage -> n0stage [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">updated PTDF, LODF, MODF, static flows</FONT></TD></TR></TABLE>>,
        likec4_id=nly0rr,
        style=dashed];
    n0stage -> n1stage [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">N-0 flows and nodal injections</FONT></TD></TR></TABLE>>,
        likec4_id="18lfffv",
        style=dashed];
    n1stage -> resultextraction [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">N-1 matrix</FONT></TD></TR></TABLE>>,
        likec4_id=o0q5b1,
        style=dashed];
    resultextraction -> scoring [arrowhead=normal,
        likec4_id="1r7jsq9",
        ltail=cluster_dcsolver,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">N-0 and N-1 flows</FONT></TD></TR></TABLE>>];
}`;case`acValidatorInternals`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=acValidatorInternals,
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
    subgraph cluster_kafka {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>KAFKA</B></FONT>>,
            likec4_depth=1,
            likec4_id=kafka,
            likec4_level=0,
            margin=32,
            style=filled
        ];
        results [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.389,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The one shared topic. Both stages publish<BR/>topologies here and the<BR/>AC-Validator also consumes it to pick up DC<BR/>candidates.</FONT></TD></TR></TABLE>>,
            likec4_id="kafka.results",
            likec4_level=1,
            margin="0.278,0.223",
            width=4.445];
    }
    subgraph cluster_processedgrid {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>PROCESSED GRID FOLDER</B></FONT>>,
            likec4_depth=1,
            likec4_id=processedGrid,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        gridsnapshot [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">grid.xiidm / grid.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The normalized backend grid, written by the<BR/>importer.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.gridSnapshot",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        actionset [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">action_set.json + action_set_diffs.hdf5</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The same action space in physical terms:<BR/>station-local reconfigurations<BR/>A and disconnectable branches D, expressed as<BR/>switch positions against<BR/>the asset topology.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.actionSet",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        snapshots [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">optimizer_snapshots/ac</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Repertoire, realized asset topologies, AC/DC<BR/>loadflow tables, SLDs, OpenRAO summaries.</FONT></TD></TR></TABLE>>,
            likec4_id="processedGrid.snapshots",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_acvalidator {
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>AC-VALIDATOR</B></FONT>>,
            likec4_depth=2,
            likec4_id="toop.acValidator",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_selectstrategy {
            graph [color="#462a17",
                fillcolor="#5a3620",
                label=<<FONT POINT-SIZE="11" COLOR="#f9b27cb3"><B>SELECT_STRATEGY</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.acValidator.selectStrategy",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            discriminator [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                group="toop.acValidator.selectStrategy",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Discriminator filter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Drop candidates too close to something<BR/>already validated.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acValidator.selectStrategy.discriminator",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            dominator [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                group="toop.acValidator.selectStrategy",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Dominator filter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Drop a candidate if another topology reaches<BR/>similar or better DC<BR/>fitness at a lower switching distance.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acValidator.selectStrategy.dominator",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            median [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                group="toop.acValidator.selectStrategy",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Median filter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Drop candidates whose fitness is below the<BR/>median of their descriptor cell.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acValidator.selectStrategy.median",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        resultlistener [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Result listener</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">SQLite (in-memory), SQLModel</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Spools the results topic into a local<BR/>database, at startup and<BR/>between epochs, so candidates are already<BR/>staged when a run begins.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator.resultListener",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        worstk [color="#853A2D",
            fillcolor="#AC4D39",
            fontcolor="#FBD3CB",
            group="toop.acValidator",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Worst-k epoch</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Reruns only the handful of contingencies the<BR/>DC stage flagged as worst<BR/>for this topology. A candidate that already<BR/>fails there, or converges<BR/>poorly, is rejected without the full</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator.worstK",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        remainingca [color="#853A2D",
            fillcolor="#AC4D39",
            fontcolor="#FBD3CB",
            group="toop.acValidator",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Remaining contingencies</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f5b2a3">PyPowSyBl security analysis, multiprocess</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">Full AC N-1 on the survivors, batched over<BR/>runner processes. Hundreds<BR/>of contingencies rather than a handful.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator.remainingCa",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acceptance [group="toop.acValidator",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Acceptance evaluation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">polars LazyFrame</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Detects constraint violations across the<BR/>loadflow tables and decides<BR/>whether a topology passes. Polars because the<BR/>result volume is the<BR/>bottleneck, not the check itself.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator.acceptance",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        summarywriter [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            group="toop.acValidator",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Summary writer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Realized asset topologies, loadflow tables,<BR/>SLDs and OpenRAO summaries.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator.summaryWriter",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    gridsnapshot -> resultlistener [arrowhead=normal,
        lhead=cluster_acvalidator,
        likec4_id=coz8k4,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">base grid</FONT></TD></TR></TABLE>>];
    actionset -> resultlistener [arrowhead=normal,
        lhead=cluster_acvalidator,
        likec4_id="1ebeh3w",
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">to realize topologies</FONT></TD></TR></TABLE>>];
    results -> resultlistener [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">DC topologies</FONT></TD></TR></TABLE>>,
        likec4_id="8u0lw7",
        minlen=1,
        style=dashed];
    resultlistener -> discriminator [arrowhead=normal,
        lhead=cluster_selectstrategy,
        likec4_id="1tshaby",
        style=dashed,
        weight=4,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">candidate pool</FONT></TD></TR></TABLE>>];
    discriminator -> dominator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">survivors</FONT></TD></TR></TABLE>>,
        likec4_id=iijqdj,
        style=dashed];
    worstk -> remainingca [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">survivors</FONT></TD></TR></TABLE>>,
        likec4_id=qevprm,
        style=dashed,
        weight=2];
    worstk -> acceptance [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">worst-k results</FONT></TD></TR></TABLE>>,
        likec4_id="42gnnp",
        style=dashed,
        weight=2];
    contingency [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Contingency analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">toop_engine_contingency_analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Runs an N-1 analysis against whichever<BR/>backend holds the grid, and<BR/>normalizes both into the same result object.<BR/>The two backends are not<BR/>at feature parity, so which one you import</FONT></TD></TR></TABLE>>,
        likec4_id="toop.contingency",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    worstk -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">worst-k contingencies</FONT></TD></TR></TABLE>>,
        likec4_id="1qhxvdl",
        style=dashed];
    dominator -> median [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">survivors</FONT></TD></TR></TABLE>>,
        likec4_id=ycl5xl,
        style=dashed];
    remainingca -> acceptance [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">full N-1 results</FONT></TD></TR></TABLE>>,
        likec4_id="1jdfjjx",
        style=dashed,
        weight=2];
    remainingca -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">full N-1</FONT></TD></TR></TABLE>>,
        likec4_id="1l8hsrl",
        style=dashed];
    median -> worstk [arrowhead=normal,
        likec4_id=o41a7,
        ltail=cluster_selectstrategy,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">selected batch</FONT></TD></TR></TABLE>>];
    acceptance -> summarywriter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">accepted topologies</FONT></TD></TR></TABLE>>,
        likec4_id="14hm69s",
        style=dashed,
        weight=4];
    interfaces [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Interfaces</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">toop_engine_interfaces</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The shared vocabulary. Everything here exists<BR/>so that two packages can<BR/>agree on a payload without depending on each<BR/>other.</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    summarywriter -> interfaces [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">accepted topology</FONT></TD></TR></TABLE>>,
        likec4_id="1rsuwgd",
        style=dashed,
        weight=3];
    loadflowstore [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>,
        likec4_id=loadflowStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    summarywriter -> loadflowstore [arrowhead=normal,
        likec4_id="1ma18vr",
        ltail=cluster_acvalidator,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>];
    summarywriter -> snapshots [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">summaries and diagrams</FONT></TD></TR></TABLE>>,
        likec4_id=tsvwcy,
        minlen=1,
        style=dashed];
    interfaces -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">persisted per job</FONT></TD></TR></TABLE>>,
        likec4_id=esngd,
        minlen=0,
        style=dashed];
    loadflowstore -> resultlistener [arrowhead=normal,
        lhead=cluster_acvalidator,
        likec4_id="1v66hnb",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial loadflow as baseline</FONT></TD></TR></TABLE>>];
}`;case`contingencyAnalysis`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=contingencyAnalysis,
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
        graph [color="#262b32",
            fillcolor="#353b43",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=3,
            likec4_id=toop,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_contingency {
            graph [color="#292f37",
                fillcolor="#3a404a",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>CONTINGENCY ANALYSIS</B></FONT>>,
                likec4_depth=2,
                likec4_id="toop.contingency",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            subgraph cluster_pwca {
                graph [color="#2a2490",
                    fillcolor="#2225aa",
                    label=<<FONT POINT-SIZE="11" COLOR="#c7d2feb3"><B>RUN_CONTINGENCY_ANALYSIS_POWSYBL</B></FONT>>,
                    likec4_depth=1,
                    likec4_id="toop.contingency.pwCa",
                    likec4_level=2,
                    margin=32,
                    style=filled
                ];
                pwlimitcache [color="#475569",
                    fillcolor="#64748b",
                    fontcolor="#f8fafc",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Branch limit cache</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Caches operational limits across runs on the<BR/>same network.</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.contingency.pwCa.pwLimitCache",
                    likec4_level=3,
                    margin="0.223,0.223",
                    width=4.445];
            }
            subgraph cluster_ppca {
                graph [color="#462a17",
                    fillcolor="#5a3620",
                    label=<<FONT POINT-SIZE="11" COLOR="#f9b27cb3"><B>RUN_CONTINGENCY_ANALYSIS_PANDAPOWER</B></FONT>>,
                    likec4_depth=1,
                    likec4_id="toop.contingency.ppCa",
                    likec4_level=2,
                    margin=40,
                    style=filled
                ];
                ppoutagegrouping [color="#475569",
                    fillcolor="#64748b",
                    fontcolor="#f8fafc",
                    group="toop.contingency.ppCa",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Outage grouping</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Expands a contingency into every element that<BR/>goes out with it.<BR/>Off by default; then each contingency is its<BR/>own group.</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.contingency.ppCa.ppOutageGrouping",
                    likec4_level=3,
                    margin="0.223,0.223",
                    width=4.445];
                ppslack [color="#475569",
                    fillcolor="#64748b",
                    fontcolor="#f8fafc",
                    group="toop.contingency.ppCa",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Slack allocation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Gives each surviving island its own slack<BR/>bus, above a minimum<BR/>island size. Without this an islanded<BR/>contingency simply fails.</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.contingency.ppCa.ppSlack",
                    likec4_level=3,
                    margin="0.223,0.223",
                    width=4.445];
                ppspps [color="#475569",
                    fillcolor="#64748b",
                    fontcolor="#f8fafc",
                    group="toop.contingency.ppCa",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">SpPS rule engine</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Special Protection Schemes as a<BR/>condition/action rule engine.<BR/>A scheme whose conditions pass applies its<BR/>actions to the network<BR/>and the loadflow re-runs, so the next</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.contingency.ppCa.ppSpps",
                    likec4_level=3,
                    margin="0.223,0.223",
                    width=4.445];
                ppcascade [color="#475569",
                    fillcolor="#64748b",
                    fontcolor="#f8fafc",
                    group="toop.contingency.ppCa",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Cascade simulation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Iterative follow-on outage simulation:<BR/>overload and distance<BR/>protection detection, outage grouping,<BR/>re-solve, repeat. Produces<BR/>an event log rather than a single</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.contingency.ppCa.ppCascade",
                    likec4_level=3,
                    margin="0.223,0.223",
                    width=4.445];
            }
            dispatcher [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group=toop,
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_ac_loadflow_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The single entry point. Dispatches on the<BR/>*type of the network<BR/>object* -- a pandapowerNet goes one way, a<BR/>PyPowSyBl Network the<BR/>other -- and raises if it is neither. Both</FONT></TD></TR></TABLE>>,
                likec4_id="toop.contingency.dispatcher",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_interfaces {
            graph [color="#292f37",
                fillcolor="#3a404a",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>INTERFACES</B></FONT>>,
                likec4_depth=2,
                likec4_id="toop.interfaces",
                likec4_level=1,
                margin=32,
                style=filled
            ];
            subgraph cluster_lfresults {
                graph [color="#2d333d",
                    fillcolor="#3e4651",
                    label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>LOADFLOWRESULTS</B></FONT>>,
                    likec4_depth=1,
                    likec4_id="toop.interfaces.lfResults",
                    likec4_level=2,
                    margin=40,
                    style=filled
                ];
                {
                    graph [rank=same];
                    branchres [color="#2d5d39",
                        fillcolor="#428a4f",
                        fontcolor="#f8fafc",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">branch_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Flows and loading per monitored branch, per<BR/>contingency and timestep.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.branchRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    noderes [color="#2d5d39",
                        fillcolor="#428a4f",
                        fontcolor="#f8fafc",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">node_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Voltage magnitude and angle per monitored<BR/>node.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.nodeRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    regres [color="#2d5d39",
                        fillcolor="#428a4f",
                        fontcolor="#f8fafc",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">regulating_element_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Tap positions and setpoints of regulating<BR/>elements.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.regRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                }
                {
                    graph [rank=same];
                    vadiffres [color="#2d5d39",
                        fillcolor="#428a4f",
                        fontcolor="#f8fafc",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">va_diff_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Voltage angle differences across the ends of<BR/>an outaged branch and<BR/>across open switches. What tells you whether<BR/>a split can be closed<BR/>again.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.vaDiffRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    convergedres [color="#2d5d39",
                        fillcolor="#428a4f",
                        fontcolor="#f8fafc",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">converged</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Convergence status per contingency and<BR/>timestep. The index of what actually ran.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.convergedRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    switchres [color="#7E451D",
                        fillcolor="#A35829",
                        fontcolor="#FFE0C2",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">switch_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Power through each monitored switch,<BR/>aggregated from everything connected to one<BR/>side.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.switchRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                }
                {
                    graph [rank=same];
                    connectivityres [color="#7E451D",
                        fillcolor="#A35829",
                        fontcolor="#FFE0C2",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">connectivity_result</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Which elements each contingency takes out.<BR/>Populated by outage grouping.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.connectivityRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    sppsres [color="#7E451D",
                        fillcolor="#A35829",
                        fontcolor="#FFE0C2",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">spps_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Per-case SpPS run summaries.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.sppsRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                    cascaderes [color="#7E451D",
                        fillcolor="#A35829",
                        fontcolor="#FFE0C2",
                        height=2.5,
                        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">cascade_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">One row per cascade event. Empty when cascade<BR/>screening is off.</FONT></TD></TR></TABLE>>,
                        likec4_id="toop.interfaces.lfResults.cascadeRes",
                        likec4_level=3,
                        margin="0.223,0.223",
                        width=4.445];
                }
                branchres -> vadiffres [style=invis];
                vadiffres -> connectivityres [minlen=1,
                    style=invis];
            }
        }
        subgraph cluster_importer {
            graph [color="#2d333d",
                fillcolor="#3e4651",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>IMPORTER</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.importer",
                likec4_level=1,
                margin=32,
                style=filled
            ];
            securityanalysis [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">import_grid_with_security_analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The only AC N-1 of the import, with the<BR/>converging loadflow<BR/>parameters saved next to the grid. Threads<BR/>and batch size come from<BR/>n_contingency_worker and</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.securityAnalysis",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
        subgraph cluster_acvalidator {
            graph [color="#2d333d",
                fillcolor="#3e4651",
                label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>AC-VALIDATOR</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.acValidator",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            worstk [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group=toop,
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Worst-k epoch</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Reruns only the handful of contingencies the<BR/>DC stage flagged as worst<BR/>for this topology. A candidate that already<BR/>fails there, or converges<BR/>poorly, is rejected without the full</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acValidator.worstK",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            remainingca [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                group=toop,
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Remaining contingencies</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">PyPowSyBl security analysis, multiprocess</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Full AC N-1 on the survivors, batched over<BR/>runner processes. Hundreds<BR/>of contingencies rather than a handful.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acValidator.remainingCa",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
        }
    }
    pwlimitcache -> branchres [arrowhead=normal,
        lhead=cluster_lfresults,
        likec4_id="1vmgu8e",
        ltail=cluster_pwca,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">fills the five common tables</FONT></TD></TR></TABLE>>];
    noderes -> regres [style=invis];
    regres -> convergedres [style=invis];
    convergedres -> switchres [style=invis];
    loadflowstore [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>,
        likec4_id=loadflowStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    cascaderes -> loadflowstore [arrowhead=normal,
        likec4_id="1yf3ugz",
        ltail=cluster_lfresults,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">persisted per job</FONT></TD></TR></TABLE>>];
    securityanalysis -> pwlimitcache [arrowhead=normal,
        lhead=cluster_pwca,
        likec4_id="1jomjox",
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">base grid N-1</FONT></TD></TR></TABLE>>];
    worstk -> remainingca [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">survivors</FONT></TD></TR></TABLE>>,
        likec4_id=qevprm,
        minlen=0,
        style=dashed,
        weight=3];
    worstk -> dispatcher [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">worst-k contingencies</FONT></TD></TR></TABLE>>,
        likec4_id=k8otn4,
        style=dashed];
    remainingca -> dispatcher [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">full N-1</FONT></TD></TR></TABLE>>,
        likec4_id="15zb3fs",
        style=dashed];
    remainingca -> loadflowstore [arrowhead=normal,
        likec4_id="1ma18vr",
        ltail=cluster_acvalidator,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>];
    dispatcher -> pwlimitcache [arrowhead=normal,
        lhead=cluster_pwca,
        likec4_id=wlczuw,
        style=dashed,
        weight=4,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">if PyPowSyBl Network</FONT></TD></TR></TABLE>>];
    dispatcher -> ppoutagegrouping [arrowhead=normal,
        lhead=cluster_ppca,
        likec4_id=wlcz67,
        style=dashed,
        weight=3,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">if pandapowerNet</FONT></TD></TR></TABLE>>];
    ppoutagegrouping -> ppslack [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">grouped contingencies</FONT></TD></TR></TABLE>>,
        likec4_id="1mjun2m",
        style=dashed];
    ppslack -> ppspps [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">solvable islands</FONT></TD></TR></TABLE>>,
        likec4_id=tav2t8,
        style=dashed];
    loadflowstore -> worstk [arrowhead=normal,
        lhead=cluster_acvalidator,
        likec4_id="1v66hnb",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial loadflow as baseline</FONT></TD></TR></TABLE>>];
    ppspps -> ppcascade [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">post-scheme state</FONT></TD></TR></TABLE>>,
        likec4_id=yjaapk,
        style=dashed];
    ppcascade -> branchres [arrowhead=normal,
        lhead=cluster_lfresults,
        likec4_id="1rpo70p",
        ltail=cluster_ppca,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">fills all nine tables</FONT></TD></TR></TABLE>>];
}`;case`assetTopology`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=assetTopology,
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
    subgraph cluster_topologymodel {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>GET_MASTER_ASSET_TOPOLOGY_ARTIFACT</B></FONT>>,
            likec4_depth=1,
            likec4_id="toop.importer.importStage.topologyModel",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        {
            graph [rank=same];
            busbreakerextract [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_bus_breaker_master_asset_topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">UCTE. Bus-breaker source, so bays and busbars<BR/>have to be inferred rather than read.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.topologyModel.busBreakerExtract",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            nodebreakerextract [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_node_breaker_master_asset_topology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">CGMES. Node-breaker source, walked as a<BR/>station graph -- the richest input, and the<BR/>one the model is shaped after.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.importer.importStage.topologyModel.nodeBreakerExtract",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
        }
        ppextract [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">get_master_asset_topology_from_network</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">pandapower nets, read through the pandapower<BR/>switch and bus tables.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.importStage.topologyModel.ppExtract",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        busbreakerextract -> ppextract [style=invis];
    }
    subgraph cluster_materialize {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>GET_RUNTIME_ASSET_TOPOLOGY</B></FONT>>,
            likec4_depth=1,
            likec4_id="toop.importer.dcPreprocess.materialize",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        pwmaterialize [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">materialize_runtime_bus_groups_from_network_state</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Reads switch positions straight off the<BR/>node-breaker network, station by station.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.dcPreprocess.materialize.pwMaterialize",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        compactmaterialize [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">materialize_runtime_bus_group_from_runtime_state</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The backend-neutral half: a canonical bus<BR/>group plus a compact<BR/>RuntimeSwitchingState overlay in, one runtime<BR/>bus group out. The<BR/>pandapower path runs through here, and so</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.dcPreprocess.materialize.compactMaterialize",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    subgraph cluster_simplify {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>SIMPLIFY_ASSET_TOPOLOGY</B></FONT>>,
            likec4_depth=1,
            likec4_id="toop.importer.dcPreprocess.simplify",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        prepareseparation [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">prepare_for_separation_set</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where the reduction actually happens, one bus<BR/>group at a time:<BR/>order assets to the solver index order, drop<BR/>out-of-service assets<BR/>and disconnected busbars, remove duplicate</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.dcPreprocess.simplify.prepareSeparation",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        bbsimplify [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">simplify_asset_topology_for_bb_outages</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The second reduction, for busbar-outage<BR/>preprocessing, run with<BR/>couplers forced closed. Yields a separate<BR/>simplified topology<BR/>rather than replacing the first one.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer.dcPreprocess.simplify.bbSimplify",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    gridsnapshot [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">grid.xiidm / grid.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The normalized backend grid, written by the<BR/>importer.</FONT></TD></TR></TABLE>>,
        likec4_id="processedGrid.gridSnapshot",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    gridsnapshot -> busbreakerextract [arrowhead=normal,
        lhead=cluster_topologymodel,
        likec4_id="1jt8oy5",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">normalized network</FONT></TD></TR></TABLE>>];
    gridsnapshot -> pwmaterialize [arrowhead=normal,
        lhead=cluster_materialize,
        likec4_id=sejkvo,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">live switch, coupler and busbar state</FONT></TD></TR></TABLE>>];
    master [color="#4f46e5",
        fillcolor="#6366f1",
        fontcolor="#eef2ff",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">1. MasterAssetTopology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Structure, no state. Bus groups with their<BR/>busbars, couplers, asset<BR/>bays and circuit groups, the branch and<BR/>injection assets they<BR/>connect, and branch_connectivity /</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces.assetTopo.master",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    ppextract -> master [arrowhead=normal,
        likec4_id=yd7v4h,
        ltail=cluster_topologymodel,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">bus groups, bays, circuit groups,<BR/>possible connectivity</FONT></TD></TR></TABLE>>];
    assettopomaster [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">initial_topology/asset_topology_master_data.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">A serialized MasterAssetTopology, and the<BR/>only form of the asset<BR/>topology that gets a file of its own. Written<BR/>by the importer, read back<BR/>at the start of DC preprocessing. The runtime</FONT></TD></TR></TABLE>>,
        likec4_id="processedGrid.assetTopoMaster",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    master -> assettopomaster [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">serialized once per import</FONT></TD></TR></TABLE>>,
        likec4_id="106tqru",
        style=dashed];
    assettopomaster -> pwmaterialize [arrowhead=normal,
        lhead=cluster_materialize,
        likec4_id=e0ty2w,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">canonical structure</FONT></TD></TR></TABLE>>];
    runtime [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">2. RuntimeAssetTopology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">The same structure, materialized against one<BR/>grid state. What it<BR/>adds is a second pair of matrices:<BR/>branch_switching_table and<BR/>injection_switching_table say what is closed</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces.assetTopo.runtime",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    compactmaterialize -> runtime [arrowhead=normal,
        likec4_id="1hhkwi0",
        ltail=cluster_materialize,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">structure + what is closed now</FONT></TD></TR></TABLE>>];
    runtime -> prepareseparation [arrowhead=normal,
        lhead=cluster_simplify,
        likec4_id="161xup4",
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">full physical bus groups</FONT></TD></TR></TABLE>>];
    storedactionset [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Stored action set</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The action set in physical terms, keyed to<BR/>the asset topology, as<BR/>opposed to the electrical index form the JAX<BR/>solver uses. Two<BR/>representations of one thing: the JAX one is</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces.storedActionSet",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    runtime -> storedactionset [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">starting_bus_groups -- to reach the real<BR/>switches</FONT></TD></TR></TABLE>>,
        likec4_id="46832j",
        style=dashed,
        weight=4];
    simplified [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">3. SimplifiedAssetTopology</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">The runtime form reduced to what the DC<BR/>solver can search -- and a<BR/>*subclass* of it, so the reduction is carried<BR/>in the type system: a<BR/>function that needs a simplified bus group</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces.assetTopo.simplified",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    bbsimplify -> simplified [arrowhead=normal,
        likec4_id="514am0",
        ltail=cluster_simplify,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">one reduced slice per electrical node</FONT></TD></TR></TABLE>>];
    electricalactions [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">compute_electrical_actions</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Stage one of action set enumeration: every<BR/>electrically distinct<BR/>two-node split of a station, filtered for<BR/>islanding and<BR/>connectivity, clipped if a station exceeds</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.dcPreprocess.electricalActions",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    simplified -> electricalactions [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">the geometry splits are enumerated in</FONT></TD></TR></TABLE>>,
        likec4_id="18wz1ja",
        style=dashed];
    stationrealisations [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">enumerate_station_realisations</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Stage two: map each electrical split onto a<BR/>reachable node-breaker<BR/>realization and precompute its reassignment<BR/>distance. Splits with<BR/>no valid realization are discarded.</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.dcPreprocess.stationRealisations",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    simplified -> stationrealisations [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">station to realize a split against</FONT></TD></TR></TABLE>>,
        likec4_id="1yrai0h",
        style=dashed];
    bboutage [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">preprocess_bb_outage</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Optional busbar outage contingencies, used by<BR/>the do-not-make-it-worse criterion.</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.dcPreprocess.bbOutage",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    simplified -> bboutage [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">reduced again with couplers closed</FONT></TD></TR></TABLE>>,
        likec4_id="19wkt54",
        style=dashed];
    simplified -> storedactionset [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">simplified_starting_bus_groups -- the<BR/>ordering local_actions is indexed<BR/>against</FONT></TD></TR></TABLE>>,
        likec4_id=m5c4m3,
        style=dashed,
        weight=4];
    electricalactions -> stationrealisations [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">electrical splits</FONT></TD></TR></TABLE>>,
        likec4_id=rhzugt,
        style=dashed,
        weight=5];
    stationrealisations -> bboutage [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">action set A</FONT></TD></TR></TABLE>>,
        likec4_id="1z14037",
        style=dashed,
        weight=5];
    stationrealisations -> storedactionset [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">physical switchings per action</FONT></TD></TR></TABLE>>,
        likec4_id=io3qsg,
        style=dashed,
        weight=2];
}`;case`loadflowFormat`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=loadflowFormat,
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
    subgraph cluster_loadflowstore {
        graph [color="#2d333d",
            fillcolor="#3e4651",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>LOADFLOW RESULT STORE</B></FONT>>,
            likec4_depth=1,
            likec4_id=loadflowStore,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        {
            graph [rank=same];
            lfmetadata [color="#475569",
                fillcolor="#64748b",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">metadata.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The only non-Parquet file: job_id and the<BR/>global warnings list. Written<BR/>first, so its presence marks the folder as<BR/>started.</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfMetadata",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfbranch [color="#2d5d39",
                fillcolor="#428a4f",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">branch_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">index: timestep, contingency, element, side<BR/>columns: i, p, q, loading, element_name,<BR/>contingency_name<BR/>Indexed per branch *end*, so a branch appears<BR/>twice per case. \`loading\`</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfBranch",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfnode [color="#2d5d39",
                fillcolor="#428a4f",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">node_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">index: timestep, contingency, element<BR/>columns: vm, vm_loading, va, p, q,<BR/>vm_basecase_deviation, element_name,<BR/>contingency_name</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfNode",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
        }
        {
            graph [rank=same];
            lfconverged [color="#2d5d39",
                fillcolor="#428a4f",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">converged.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">index: timestep, contingency<BR/>columns: status, iteration_count, warnings,<BR/>contingency_name<BR/>The index of what actually ran. Read this<BR/>first: non-converging cases</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfConverged",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfvadiff [color="#2d5d39",
                fillcolor="#428a4f",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">va_diff_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">index: timestep, contingency, element<BR/>columns: va_diff, element_name,<BR/>contingency_name</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfVaDiff",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfreg [color="#2d5d39",
                fillcolor="#428a4f",
                fontcolor="#f8fafc",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">regulating_element_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">index: timestep, contingency, element<BR/>columns: value, regulating_element_type,<BR/>element_name, contingency_name</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfReg",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
        }
        {
            graph [rank=same];
            lfswitch [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">switch_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">index: timestep, contingency, element<BR/>columns: p, q, vm, i, element_name,<BR/>contingency_name, side<BR/>Optional -- the file is absent unless the<BR/>table was populated. The one</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfSwitch",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfspps [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">spps_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">index: timestep, contingency<BR/>columns: iterations,<BR/>activated_schemes_per_iter,<BR/>max_iterations_reached, power_flow_failed<BR/>Optional.</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfSpps",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
            lfcascade [color="#7E451D",
                fillcolor="#A35829",
                fontcolor="#FFE0C2",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">cascade_results.parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">index: timestep, contingency, cascade_number,<BR/>element_mrid<BR/>columns: element_id, contingency_outage_id,<BR/>element_outage_group_id,<BR/>element_name, contingency_name,</FONT></TD></TR></TABLE>>,
                likec4_id="loadflowStore.lfCascade",
                likec4_level=1,
                margin="0.223,0.223",
                width=4.445];
        }
        lfmetadata -> lfconverged [style=invis];
        lfconverged -> lfswitch [minlen=1,
            style=invis];
    }
    lfbranch -> lfnode [style=invis];
    lfnode -> lfvadiff [style=invis];
    lfvadiff -> lfreg [style=invis];
    acvalidator [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>,
        likec4_id="toop.acValidator",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    lfcascade -> acvalidator [arrowhead=normal,
        likec4_id="1v66hnb",
        ltail=cluster_loadflowstore,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial loadflow as baseline</FONT></TD></TR></TABLE>>];
    lfresults [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">LoadflowResults</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">One container per computation job, holding a<BR/>pandera-validated frame<BR/>per result family, plus warnings. Mirrored by<BR/>LoadflowResultsPolars,<BR/>whose schemas subclass the pandas ones so the</FONT></TD></TR></TABLE>>,
        likec4_id="toop.interfaces.lfResults",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    lfresults -> lfmetadata [arrowhead=normal,
        lhead=cluster_loadflowstore,
        likec4_id="1yf3ugz",
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">persisted per job</FONT></TD></TR></TABLE>>];
    initialloadflow [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">store_initial_ca_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Stores the AC N-1 of the import, with<BR/>permanent and artificial branch<BR/>limits, as the reference every proposed<BR/>topology is later compared<BR/>against. Runs before the DC preprocessing and</FONT></TD></TR></TABLE>>,
        likec4_id="toop.importer.initialLoadflow",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    initialloadflow -> lfmetadata [arrowhead=normal,
        lhead=cluster_loadflowstore,
        likec4_id=sprnet,
        minlen=1,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial AC N-1 results</FONT></TD></TR></TABLE>>];
    acvalidator -> lfmetadata [arrowhead=normal,
        lhead=cluster_loadflowstore,
        likec4_id="1ma18vr",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>];
}`;case`index`:return`digraph {
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
    subgraph cluster_toop {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=1,
            likec4_id=toop,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        interfaces [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Interfaces</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">toop_engine_interfaces</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">The shared vocabulary. Everything here exists<BR/>so that two packages can<BR/>agree on a payload without depending on each<BR/>other.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.interfaces",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        postprocess [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Postprocessing and export</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">toop_engine_dc_solver.postprocess,</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Turns an action index back into something a<BR/>grid tool can open.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.postprocess",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        lfservice [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC loadflow service</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">A standalone N-1 service on its own<BR/>loadflow_commands / loadflow_results<BR/>/ loadflow_heartbeat topics. Present in the<BR/>codebase but off the main<BR/>path: dev-deployment does not create its</FONT></TD></TR></TABLE>>,
            likec4_id="toop.lfService",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        importerparams [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Importer parameters</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Carried by the StartPreprocessingCommand.<BR/>Fixes the scope of the whole<BR/>run before any search happens: which grid,<BR/>which area, which stations<BR/>may be switched, which contingencies, and how</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importerParams",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcparams [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC optimizer parameters</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Carried by the StartOptimizationCommand.<BR/>Everything about how the<BR/>search behaves and what it optimizes for.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcParams",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acparams [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC validator parameters</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Carried by the same StartOptimizationCommand<BR/>as the DC parameters.<BR/>Mostly about what to reject and how much<BR/>compute to spend.</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acParams",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        importer [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Importer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl, pandapower, JAX</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Normalizes a raw grid into a processed grid<BR/>folder and derives the<BR/>solver artifacts. Most of it depends only on<BR/>the initial grid topology,<BR/>so it can run before the forecast is</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcoptimizer [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acvalidator [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        contingency [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Contingency analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">toop_engine_contingency_analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Runs an N-1 analysis against whichever<BR/>backend holds the grid, and<BR/>normalizes both into the same result object.<BR/>The two backends are not<BR/>at feature parity, so which one you import</FONT></TD></TR></TABLE>>,
            likec4_id="toop.contingency",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    client [color="#0369a1",
        fillcolor="#0284c7",
        fontcolor="#f0f9ff",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>,
        likec4_id=client,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    kafka [color="#7E451D",
        fillcolor="#A35829",
        fontcolor="#FFE0C2",
        height=2.389,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Kafka</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#f9b27c">confluent-kafka</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">Six topics, created by<BR/>dev-deployment/docker-compose.yaml. Every<BR/>payload<BR/>is a pydantic model dumped to JSON and<BR/>wrapped in a single protobuf</FONT></TD></TR></TABLE>>,
        likec4_id=kafka,
        likec4_level=0,
        margin="0.278,0.223",
        width=4.445];
    client -> kafka [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=nk4drp,
        style=dashed,
        weight=2];
    client -> importerparams [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in StartPreprocessingCommand</FONT></TD></TR></TABLE>>,
        likec4_id=z29c3z,
        style=dashed];
    client -> dcparams [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in StartOptimizationCommand</FONT></TD></TR></TABLE>>,
        likec4_id="1d2rqyq",
        style=dashed];
    client -> acparams [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in the same StartOptimizationCommand</FONT></TD></TR></TABLE>>,
        likec4_id="7i8so7",
        style=dashed];
    unprocessedgridstore [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Unprocessed grid store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec AbstractFileSystem</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Where the source grid files land before<BR/>anything touches them. The same<BR/>kind of thing as the loadflow result store --<BR/>an fsspec filesystem the<BR/>worker is handed, local disk or object</FONT></TD></TR></TABLE>>,
        likec4_id=unprocessedGridStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    unprocessedgridstore -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1tyg1gc",
        minlen=1,
        style=dashed];
    loadflowstore [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loadflow result store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec, polars, Parquet</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Loadflow tables addressed by a<BR/>StoredLoadflowReference passed in messages,<BR/>so the tables themselves stay out of Kafka.<BR/>The AC-Validator is the main producer: every<BR/>topology it evaluates gets</FONT></TD></TR></TABLE>>,
        likec4_id=loadflowStore,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    interfaces -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">persisted per job</FONT></TD></TR></TABLE>>,
        likec4_id=esngd,
        style=dashed];
    interfaces -> postprocess [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="3qsei7",
        style=dashed];
    processedgrid [color="#2d5d39",
        fillcolor="#428a4f",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Processed grid folder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec AbstractFileSystem</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">One folder per import job, shared by all<BR/>three stages and the only large<BR/>payload that never travels through Kafka.<BR/>fsspec keeps the backend open:<BR/>local disk in the dev setup, object storage</FONT></TD></TR></TABLE>>,
        likec4_id=processedGrid,
        likec4_level=0,
        margin="0.223,0",
        penwidth=2,
        shape=cylinder,
        width=4.445];
    interfaces -> processedgrid [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">serialized once per import</FONT></TD></TR></TABLE>>,
        likec4_id="1mp5nus",
        style=dashed];
    interfaces -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1vbdvxy",
        style=dashed];
    interfaces -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">monitored elements and contingencies</FONT></TD></TR></TABLE>>,
        likec4_id=kt7ewr,
        style=dashed];
    downstream [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>,
        likec4_id=downstream,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    kafka -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">validated topologies for review</FONT></TD></TR></TABLE>>,
        likec4_id=fty1qa,
        style=dashed,
        weight=2];
    kafka -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">consumes command</FONT></TD></TR></TABLE>>,
        likec4_id=jw2tws,
        style=dashed];
    kafka -> dcoptimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">consumes command</FONT></TD></TR></TABLE>>,
        likec4_id="1g6gj8q",
        style=dashed];
    kafka -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=jhh4pm,
        style=dashed];
    loadflowstore -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial loadflow as baseline</FONT></TD></TR></TABLE>>,
        likec4_id="1v66hnb",
        style=dashed];
    postprocess -> interfaces [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">switch id and new state</FONT></TD></TR></TABLE>>,
        likec4_id="18a92jz",
        style=dashed];
    postprocess -> processedgrid [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="197we3l",
        style=dashed];
    postprocess -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">grid with topology applied</FONT></TD></TR></TABLE>>,
        likec4_id="136kr9a",
        style=dashed,
        weight=2];
    processedgrid -> downstream [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">UCTE, DGS, OpenRAO summaries, single<BR/>line diagrams</FONT></TD></TR></TABLE>>,
        likec4_id="19tydzq",
        style=dashed,
        weight=2];
    processedgrid -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=oi4j94,
        style=dashed];
    processedgrid -> dcoptimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1ht8cji",
        style=dashed];
    processedgrid -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1vo1sem",
        style=dashed];
    importerparams -> importer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">scope, limits, contingencies</FONT></TD></TR></TABLE>>,
        likec4_id=e2spcm,
        style=dashed,
        weight=2];
    dcparams -> dcoptimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">search bounds, fitness, operator<BR/>probabilities</FONT></TD></TR></TABLE>>,
        likec4_id="1psk56l",
        style=dashed,
        weight=2];
    acparams -> acvalidator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">compute budget, pruning, rejection<BR/>thresholds</FONT></TD></TR></TABLE>>,
        likec4_id="4idlag",
        style=dashed,
        weight=2];
    importer -> interfaces [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1dn8dx2",
        style=dashed];
    importer -> kafka [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="3v31ik",
        style=dashed];
    importer -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initial AC N-1 results</FONT></TD></TR></TABLE>>,
        likec4_id=luns8x,
        style=dashed];
    importer -> processedgrid [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="19veoxk",
        style=dashed];
    importer -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">base grid N-1</FONT></TD></TR></TABLE>>,
        likec4_id=yiu5on,
        style=dashed,
        weight=2];
    dcoptimizer -> kafka [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=n3hqi2,
        style=dashed];
    acvalidator -> interfaces [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">accepted topology</FONT></TD></TR></TABLE>>,
        likec4_id=mkoqgw,
        style=dashed,
        weight=2];
    acvalidator -> kafka [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="7eed96",
        style=dashed];
    acvalidator -> loadflowstore [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>,
        likec4_id="1ma18vr",
        style=dashed];
    acvalidator -> processedgrid [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">summaries and diagrams</FONT></TD></TR></TABLE>>,
        likec4_id="19h6qz2",
        style=dashed];
    acvalidator -> contingency [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=vp9241,
        style=dashed,
        weight=2];
    contingency -> interfaces [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="10hp2t7",
        style=dashed];
}`;case`overview`:return`digraph {
  likec4_viewId = "overview";
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
    likec4_id = "client";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#B6ECF7">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#0284c7";
    fontcolor = "#f0f9ff";
    color = "#0369a1";
  ];
  "importercommands" [
    likec4_id = "kafka.importerCommands";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">importer_commands</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">StartPreprocessingCommand, ShutdownCommand.<BR/>24 partitions.</FONT></TD></TR></TABLE>>;
    margin = "0.278,0.223";
    width = 4.445;
    height = 2.389;
    fillcolor = "#A35829";
    fontcolor = "#FFE0C2";
    color = "#7E451D";
  ];
  "unprocessedgridstore" [
    likec4_id = "unprocessedGridStore";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Unprocessed grid store</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec AbstractFileSystem</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">Where the source grid files land before<BR/>anything touches them. The same<BR/>kind of thing as the loadflow result store --<BR/>an fsspec filesystem the<BR/>worker is handed, local disk or object</FONT></TD></TR></TABLE>>;
    margin = "0.223,0";
    width = 4.445;
    height = 2.5;
    fillcolor = "#428a4f";
    fontcolor = "#f8fafc";
    color = "#2d5d39";
    penwidth = 2;
    shape = "cylinder";
  ];
  "importer" [
    likec4_id = "toop.importer";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Importer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl, pandapower, JAX</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Normalizes a raw grid into a processed grid<BR/>folder and derives the<BR/>solver artifacts. Most of it depends only on<BR/>the initial grid topology,<BR/>so it can run before the forecast is</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "processedgrid" [
    likec4_id = "processedGrid";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Processed grid folder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#c2f0c2">fsspec AbstractFileSystem</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c2f0c2">One folder per import job, shared by all<BR/>three stages and the only large<BR/>payload that never travels through Kafka.<BR/>fsspec keeps the backend open:<BR/>local disk in the dev setup, object storage</FONT></TD></TR></TABLE>>;
    margin = "0.223,0";
    width = 4.445;
    height = 2.5;
    fillcolor = "#428a4f";
    fontcolor = "#f8fafc";
    color = "#2d5d39";
    penwidth = 2;
    shape = "cylinder";
  ];
  "loadflowstore" [
    likec4_id = "loadflowStore";
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
  "importerresults" [
    likec4_id = "kafka.importerResults";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">importer_results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">PreprocessingStartedResult,<BR/>PreprocessingSuccessResult, ErrorResult</FONT></TD></TR></TABLE>>;
    margin = "0.278,0.223";
    width = 4.445;
    height = 2.389;
    fillcolor = "#A35829";
    fontcolor = "#FFE0C2";
    color = "#7E451D";
  ];
  "commands" [
    likec4_id = "kafka.commands";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">commands</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">StartOptimizationCommand, ShutdownCommand. 4<BR/>partitions.</FONT></TD></TR></TABLE>>;
    margin = "0.278,0.223";
    width = 4.445;
    height = 2.389;
    fillcolor = "#A35829";
    fontcolor = "#FFE0C2";
    color = "#7E451D";
  ];
  "dcoptimizer" [
    likec4_id = "toop.dcOptimizer";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "acvalidator" [
    likec4_id = "toop.acValidator";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
  ];
  "results" [
    likec4_id = "kafka.results";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">results</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f9b27c">The one shared topic. Both stages publish<BR/>topologies here and the<BR/>AC-Validator also consumes it to pick up DC<BR/>candidates.</FONT></TD></TR></TABLE>>;
    margin = "0.278,0.223";
    width = 4.445;
    height = 2.389;
    fillcolor = "#A35829";
    fontcolor = "#FFE0C2";
    color = "#7E451D";
  ];
  "downstream" [
    likec4_id = "downstream";
    likec4_level = 0;
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Frontend / downstream systems</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Where an operator reviews the proposed<BR/>actions and exports the accepted<BR/>ones. Not part of this repository.</FONT></TD></TR></TABLE>>;
    margin = "0.223,0.223";
    width = 4.445;
    height = 2.5;
    fillcolor = "#64748b";
    fontcolor = "#f8fafc";
    color = "#475569";
  ];
  "client" -> "importercommands" [
    likec4_id = "step-01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>0</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">StartPreprocessingCommand</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "importercommands" -> "importer" [
    likec4_id = "step-02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>1</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">picks up the job</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "unprocessedgridstore" -> "importer" [
    likec4_id = "step-03";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>2</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">UCTE / CGMES / PowerFactory file</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "importer" -> "processedgrid" [
    likec4_id = "step-04";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>3</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">normalized snapshot, masks, asset<BR/>topology</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "importer" -> "processedgrid" [
    likec4_id = "step-05";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>4</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">PTDF, action set, contingency set</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "importer" -> "loadflowstore" [
    likec4_id = "step-06";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>5</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">initial AC N-1 and reference metrics</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "importer" -> "importerresults" [
    likec4_id = "step-07";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>6</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">PreprocessingSuccessResult</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "client" -> "importerresults" [
    likec4_id = "step-08";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>7</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">data folder is ready</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "client" -> "commands" [
    likec4_id = "step-09";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>8</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">StartOptimizationCommand</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "commands" -> "dcoptimizer" [
    likec4_id = "step-10:par.01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>9</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">starts the DC run</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "commands" -> "acvalidator" [
    likec4_id = "step-10:par.02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>10</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">starts the AC run</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "processedgrid" -> "dcoptimizer" [
    likec4_id = "step-11:par.01";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>11</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">loads static information onto the GPU</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "processedgrid" -> "acvalidator" [
    likec4_id = "step-11:par.02";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>12</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">loads base grid and action set</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "loadflowstore" -> "acvalidator" [
    likec4_id = "step-12";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>13</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">reads the initial loadflow as baseline</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "dcoptimizer" -> "results" [
    likec4_id = "step-13";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>14</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">Strategy, once per epoch</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "acvalidator" -> "results" [
    likec4_id = "step-14";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>15</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">DC topologies to validate</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "acvalidator" -> "acvalidator" [
    likec4_id = "step-15";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>16</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">prune, worst-k, then full N-1</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "loadflowstore" -> "acvalidator" [
    likec4_id = "step-16";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>17</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">AC loadflow results per evaluated<BR/>topology</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "acvalidator" -> "results" [
    likec4_id = "step-17";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>18</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">AC-validated Strategy, referencing its<BR/>loadflow</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "processedgrid" -> "acvalidator" [
    likec4_id = "step-18";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>19</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">summaries, diagrams, loadflow tables</FONT></TD></TR></TABLE>>;
    arrowtail = "normal";
    dir = "back";
  ];
  "results" -> "downstream" [
    likec4_id = "step-19";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>20</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">topologies for review</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
  "processedgrid" -> "downstream" [
    likec4_id = "step-20";
    label = <<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="3"><TR><TD><TABLE BORDER="0" CELLPADDING="6" BGCOLOR="#18191BA0"><TR><TD WIDTH="20" HEIGHT="20"><FONT POINT-SIZE="14"><B>21</B></FONT></TD></TR></TABLE></TD><TD BGCOLOR="#18191BA0" CELLPADDING="3"><FONT POINT-SIZE="14">UCTE, DGS, OpenRAO summaries, single<BR/>line diagrams</FONT></TD></TR></TABLE>>;
    arrowhead = "normal";
  ];
}`;case`parameters`:return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=parameters,
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
        graph [color="#292f37",
            fillcolor="#3a404a",
            label=<<FONT POINT-SIZE="11" COLOR="#cbd5e1b3"><B>TOOP ENGINE</B></FONT>>,
            likec4_depth=2,
            likec4_id=toop,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        subgraph cluster_importerparams {
            graph [color="#1b3d88",
                fillcolor="#194b9e",
                label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>IMPORTER PARAMETERS</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.importerParams",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            {
                graph [rank=same];
                pareasettings [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AreaSettings</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">control_area and view_area, plus<BR/>cutoff_voltage (220 kV by default).<BR/>Also where the limit adjustments live:<BR/>dso_trafo_factors and<BR/>border_line_factors, each a</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pAreaSettings",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                pstationrules [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">RelevantStationRules</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">What makes a station worth switching:<BR/>min_busbars (2),<BR/>min_connected_branches (4),<BR/>min_connected_elements (4). This decides<BR/>the set of switchable substations.</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pStationRules",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                plists [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">White / black / ignore lists</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">white_list_file, black_list_file,<BR/>ignore_list_file and<BR/>select_by_voltage_level_id_list. Operator<BR/>overrides on top of the<BR/>area rules, applied during importer</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pLists",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
            }
            {
                graph [rank=same];
                pcontingencies [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Contingency list</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">contingency_list_file plus its schema_format,<BR/>either the PowerFactory<BR/>import schema or the generic one. Becomes the<BR/>N-1 definition. Without<BR/>it, contingencies are derived from the</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pContingencies",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                psecurityanalysis [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Import security analysis</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">n_contingency_worker (8 OpenLoadFlow threads)<BR/>and<BR/>contingency_batch_size (all outages at once<BR/>by default) size the one<BR/>AC N-1 of the import. Its results set the</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pSecurityAnalysis",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                ppreprocess [height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">PreprocessParameters</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">How hard to work on the action space.<BR/>action_set_clip caps a station<BR/>at 2^23 configurations;<BR/>action_set_filter_bridge_lookup and<BR/>action_set_filter_bsdf_lodf drop splits that</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.importerParams.pPreprocess",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
            }
            pareasettings -> pcontingencies [minlen=1,
                style=invis];
        }
        subgraph cluster_dcparams {
            graph [color="#2a2490",
                fillcolor="#2225aa",
                label=<<FONT POINT-SIZE="11" COLOR="#c7d2feb3"><B>DC OPTIMIZER PARAMETERS</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.dcParams",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            {
                graph [rank=same];
                pme [color="#4f46e5",
                    fillcolor="#6366f1",
                    fontcolor="#eef2ff",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">BatchedMEParameters (ga_config)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">The search itself. runtime_seconds and<BR/>iterations_per_epoch set the<BR/>budget and how often results are pushed.<BR/>target_metrics and<BR/>observed_metrics define the fitness --</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.dcParams.pMe",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                psolver [color="#4f46e5",
                    fillcolor="#6366f1",
                    fontcolor="#eef2ff",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">LoadflowSolverParameters</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">The shape of the search space and the batch.<BR/>max_num_splits (4) and<BR/>max_num_disconnections cap the genome;<BR/>batch_size sets how many<BR/>topologies the GPU evaluates at once;</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.dcParams.pSolver",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
            }
            pdoublelimits [color="#4f46e5",
                fillcolor="#6366f1",
                fontcolor="#eef2ff",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DoubleLimits</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#c7d2fe">Optional. Separate permanent and temporary<BR/>branch limits, so N-0 and<BR/>N-1 can be judged against different ratings.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.dcParams.pDoubleLimits",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            pme -> pdoublelimits [style=invis];
        }
        subgraph cluster_acparams {
            graph [color="#4b2720",
                fillcolor="#603329",
                label=<<FONT POINT-SIZE="11" COLOR="#f5b2a3b3"><B>AC VALIDATOR PARAMETERS</B></FONT>>,
                likec4_depth=1,
                likec4_id="toop.acParams",
                likec4_level=1,
                margin=40,
                style=filled
            ];
            {
                graph [rank=same];
                pacga [color="#853A2D",
                    fillcolor="#AC4D39",
                    fontcolor="#FBD3CB",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">ACGAParameters (ga_config)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">runtime_seconds (180) and<BR/>max_initial_wait_seconds bound the run;<BR/>runner_processes, contingency_processes and<BR/>their worst_k<BR/>counterparts set the CPU parallelism,</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.acParams.pAcGa",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
                prejection [color="#853A2D",
                    fillcolor="#AC4D39",
                    fontcolor="#FBD3CB",
                    height=2.5,
                    label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Rejection thresholds</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">What counts as a failure. enable_ac_rejection<BR/>switches the gate on,<BR/>then reject_overload_threshold (0.95),<BR/>reject_critical_branch_threshold<BR/>(1.1), reject_convergence_threshold,</FONT></TD></TR></TABLE>>,
                    likec4_id="toop.acParams.pRejection",
                    likec4_level=2,
                    margin="0.223,0.223",
                    width=4.445];
            }
            pinitialloadflow [color="#853A2D",
                fillcolor="#AC4D39",
                fontcolor="#FBD3CB",
                height=2.5,
                label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">initial_loadflow reference</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#f5b2a3">An optional StoredLoadflowReference to the<BR/>baseline. If absent the<BR/>validator computes and stores the initial AC<BR/>N-1 itself.</FONT></TD></TR></TABLE>>,
                likec4_id="toop.acParams.pInitialLoadflow",
                likec4_level=2,
                margin="0.223,0.223",
                width=4.445];
            pacga -> pinitialloadflow [style=invis];
        }
        importer [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Importer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl, pandapower, JAX</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Normalizes a raw grid into a processed grid<BR/>folder and derives the<BR/>solver artifacts. Most of it depends only on<BR/>the initial grid topology,<BR/>so it can run before the forecast is</FONT></TD></TR></TABLE>>,
            likec4_id="toop.importer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dcoptimizer [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DC-Optimizer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, JAX / XLA</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Quality-diversity search over the action set.<BR/>The whole loop is<BR/>GPU-resident, so no host transfer happens per<BR/>iteration; results leave<BR/>only once per epoch. JAX JIT costs about 13s</FONT></TD></TR></TABLE>>,
            likec4_id="toop.dcOptimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        acvalidator [color="#475569",
            fillcolor="#64748b",
            fontcolor="#f8fafc",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">AC-Validator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#cbd5e1">Python, PyPowSyBl, polars, SQLite</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Proposes no topologies of its own -- it is<BR/>the quality gate in front of<BR/>the operator. What it does produce is the AC<BR/>loadflow results: every<BR/>candidate it evaluates gets a full result</FONT></TD></TR></TABLE>>,
            likec4_id="toop.acValidator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    client [color="#475569",
        fillcolor="#64748b",
        fontcolor="#f8fafc",
        height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Operator / orchestration client</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#cbd5e1">Drives the engine either directly from Python<BR/>or by producing Kafka<BR/>commands. ToOp ships no GUI or system<BR/>integration of its own.<BR/>In operational use the whole run must finish</FONT></TD></TR></TABLE>>,
        likec4_id=client,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    client -> pareasettings [arrowhead=normal,
        lhead=cluster_importerparams,
        likec4_id=z29c3z,
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in StartPreprocessingCommand</FONT></TD></TR></TABLE>>];
    client -> pme [arrowhead=normal,
        lhead=cluster_dcparams,
        likec4_id="1d2rqyq",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in StartOptimizationCommand</FONT></TD></TR></TABLE>>];
    client -> pacga [arrowhead=normal,
        lhead=cluster_acparams,
        likec4_id="7i8so7",
        style=dashed,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">set in the same StartOptimizationCommand</FONT></TD></TR></TABLE>>];
    pstationrules -> plists [style=invis];
    plists -> psecurityanalysis [style=invis];
    psecurityanalysis -> psolver [style=invis];
    ppreprocess -> importer [arrowhead=normal,
        likec4_id=e2spcm,
        ltail=cluster_importerparams,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">scope, limits, contingencies</FONT></TD></TR></TABLE>>];
    pdoublelimits -> dcoptimizer [arrowhead=normal,
        likec4_id="1psk56l",
        ltail=cluster_dcparams,
        minlen=1,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">search bounds, fitness, operator<BR/>probabilities</FONT></TD></TR></TABLE>>];
    pinitialloadflow -> acvalidator [arrowhead=normal,
        likec4_id="4idlag",
        ltail=cluster_acparams,
        minlen=1,
        style=dashed,
        weight=2,
        xlabel=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">compute budget, pruning, rejection<BR/>thresholds</FONT></TD></TR></TABLE>>];
}`;default:throw Error(`Unknown viewId: `+e)}},t=e=>{switch(e){case`dataFlow`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="2330pt" height="3215pt"
 viewBox="0.00 0.00 2330.00 3215.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3199.92)">
<g id="clust1" class="cluster">
<title>cluster_processedgrid</title>
<polygon fill="#2c4e32" stroke="#1e3524" points="1162.68,-1784.77 1162.68,-2935.77 1619.8,-2935.77 1619.8,-1784.77 1162.68,-1784.77"/>
<text xml:space="preserve" text-anchor="start" x="1170.68" y="-2922.32" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c2f0c2" fill-opacity="0.701961">PROCESSED GRID FOLDER</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_kafka</title>
<polygon fill="#5a3620" stroke="#462a17" points="1153.22,-92.77 1153.22,-1775.77 1629.26,-1775.77 1629.26,-92.77 1153.22,-92.77"/>
<text xml:space="preserve" text-anchor="start" x="1161.22" y="-1762.32" font-family="Arial" font-weight="bold" font-size="11.00" fill="#f9b27c" fill-opacity="0.701961">KAFKA</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_toop</title>
<polygon fill="#3e4651" stroke="#2d333d" points="427.11,-1336.77 427.11,-2487.77 862.22,-2487.77 862.22,-1336.77 427.11,-1336.77"/>
<text xml:space="preserve" text-anchor="start" x="435.11" y="-2474.32" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<!-- gridsnapshot -->
<g id="node1" class="node">
<title>gridsnapshot</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1558.3,-2874.77 1224.18,-2874.77 1224.18,-2694.77 1558.3,-2694.77 1558.3,-2874.77"/>
<text xml:space="preserve" text-anchor="start" x="1303.24" y="-2816.77" font-family="Arial" font-size="20.00" fill="#f8fafc">grid.xiidm / grid.json</text>
<text xml:space="preserve" text-anchor="start" x="1244.24" y="-2788.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">The normalized backend grid, written by the</text>
<text xml:space="preserve" text-anchor="start" x="1361.74" y="-2770.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">importer.</text>
</g>
<!-- staticinfo -->
<g id="node2" class="node">
<title>staticinfo</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1563.8,-2294.77 1218.68,-2294.77 1218.68,-2114.77 1563.8,-2114.77 1563.8,-2294.77"/>
<text xml:space="preserve" text-anchor="start" x="1291.74" y="-2263.77" font-family="Arial" font-size="20.00" fill="#f8fafc">static_information.hdf5</text>
<text xml:space="preserve" text-anchor="start" x="1238.74" y="-2235.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">The critical asset: everything the GPU needs,</text>
<text xml:space="preserve" text-anchor="start" x="1312.24" y="-2217.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">and nothing it does not.</text>
<text xml:space="preserve" text-anchor="start" x="1270.24" y="-2199.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">One serialized StaticInformation &#45;&#45; a</text>
<text xml:space="preserve" text-anchor="start" x="1281.74" y="-2181.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">SolverConfig, which is static and</text>
<text xml:space="preserve" text-anchor="start" x="1286.74" y="-2163.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">part of the JIT signature, plus a</text>
</g>
<!-- actionset -->
<g id="node3" class="node">
<title>actionset</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1579.8,-2584.77 1202.68,-2584.77 1202.68,-2404.77 1579.8,-2404.77 1579.8,-2584.77"/>
<text xml:space="preserve" text-anchor="start" x="1222.74" y="-2553.77" font-family="Arial" font-size="20.00" fill="#f8fafc">action_set.json + action_set_diffs.hdf5</text>
<text xml:space="preserve" text-anchor="start" x="1252.24" y="-2525.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">The same action space in physical terms:</text>
<text xml:space="preserve" text-anchor="start" x="1294.74" y="-2507.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">station&#45;local reconfigurations</text>
<text xml:space="preserve" text-anchor="start" x="1228.74" y="-2489.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">A and disconnectable branches D, expressed as</text>
<text xml:space="preserve" text-anchor="start" x="1311.24" y="-2471.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">switch positions against</text>
<text xml:space="preserve" text-anchor="start" x="1328.24" y="-2453.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">the asset topology.</text>
</g>
<!-- snapshots -->
<g id="node4" class="node">
<title>snapshots</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1564.8,-2004.77 1217.68,-2004.77 1217.68,-1824.77 1564.8,-1824.77 1564.8,-2004.77"/>
<text xml:space="preserve" text-anchor="start" x="1286.74" y="-1946.77" font-family="Arial" font-size="20.00" fill="#f8fafc">optimizer_snapshots/ac</text>
<text xml:space="preserve" text-anchor="start" x="1240.74" y="-1918.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">Repertoire, realized asset topologies, AC/DC</text>
<text xml:space="preserve" text-anchor="start" x="1237.74" y="-1900.77" font-family="Arial" font-size="15.00" fill="#c2f0c2">loadflow tables, SLDs, OpenRAO summaries.</text>
</g>
<!-- importercommands -->
<g id="node5" class="node">
<title>importercommands</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1589.26,-868.77 1193.22,-868.77 1193.22,-696.76 1589.26,-696.76 1589.26,-868.77"/>
<text xml:space="preserve" text-anchor="start" x="1301.24" y="-814.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">importer_commands</text>
<text xml:space="preserve" text-anchor="start" x="1217.24" y="-786.77" font-family="Arial" font-size="15.00" fill="#f9b27c">StartPreprocessingCommand, ShutdownCommand.</text>
<text xml:space="preserve" text-anchor="start" x="1347.74" y="-768.77" font-family="Arial" font-size="15.00" fill="#f9b27c">24 partitions.</text>
</g>
<!-- commands -->
<g id="node6" class="node">
<title>commands</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1589.26,-304.77 1193.22,-304.77 1193.22,-132.76 1589.26,-132.76 1589.26,-304.77"/>
<text xml:space="preserve" text-anchor="start" x="1342.74" y="-250.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">commands</text>
<text xml:space="preserve" text-anchor="start" x="1217.24" y="-222.77" font-family="Arial" font-size="15.00" fill="#f9b27c">StartOptimizationCommand, ShutdownCommand. 4</text>
<text xml:space="preserve" text-anchor="start" x="1358.24" y="-204.77" font-family="Arial" font-size="15.00" fill="#f9b27c">partitions.</text>
</g>
<!-- importerresults -->
<g id="node7" class="node">
<title>importerresults</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1556.26,-1714.77 1226.22,-1714.77 1226.22,-1542.76 1556.26,-1542.76 1556.26,-1714.77"/>
<text xml:space="preserve" text-anchor="start" x="1320.74" y="-1660.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">importer_results</text>
<text xml:space="preserve" text-anchor="start" x="1295.24" y="-1632.77" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessingStartedResult,</text>
<text xml:space="preserve" text-anchor="start" x="1250.24" y="-1614.77" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessingSuccessResult, ErrorResult</text>
</g>
<!-- importerheartbeat -->
<g id="node8" class="node">
<title>importerheartbeat</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1553.76,-1432.77 1228.72,-1432.77 1228.72,-1260.76 1553.76,-1260.76 1553.76,-1432.77"/>
<text xml:space="preserve" text-anchor="start" x="1307.74" y="-1378.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">importer_heartbeat</text>
<text xml:space="preserve" text-anchor="start" x="1252.74" y="-1350.77" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessHeartbeat carrying the current</text>
<text xml:space="preserve" text-anchor="start" x="1333.74" y="-1332.77" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessStage</text>
</g>
<!-- results -->
<g id="node9" class="node">
<title>results</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1564.26,-1150.77 1218.22,-1150.77 1218.22,-978.76 1564.26,-978.76 1564.26,-1150.77"/>
<text xml:space="preserve" text-anchor="start" x="1362.74" y="-1114.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">results</text>
<text xml:space="preserve" text-anchor="start" x="1249.74" y="-1086.77" font-family="Arial" font-size="15.00" fill="#f9b27c">The one shared topic. Both stages publish</text>
<text xml:space="preserve" text-anchor="start" x="1312.24" y="-1068.77" font-family="Arial" font-size="15.00" fill="#f9b27c">topologies here and the</text>
<text xml:space="preserve" text-anchor="start" x="1242.24" y="-1050.77" font-family="Arial" font-size="15.00" fill="#f9b27c">AC&#45;Validator also consumes it to pick up DC</text>
<text xml:space="preserve" text-anchor="start" x="1352.74" y="-1032.77" font-family="Arial" font-size="15.00" fill="#f9b27c">candidates.</text>
</g>
<!-- heartbeat -->
<g id="node10" class="node">
<title>heartbeat</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1561.76,-586.77 1220.72,-586.77 1220.72,-414.76 1561.76,-414.76 1561.76,-586.77"/>
<text xml:space="preserve" text-anchor="start" x="1349.74" y="-532.77" font-family="Arial" font-size="20.00" fill="#ffe0c2">heartbeat</text>
<text xml:space="preserve" text-anchor="start" x="1244.74" y="-504.77" font-family="Arial" font-size="15.00" fill="#f9b27c">Heartbeat tagged with OptimizerType.DC or</text>
<text xml:space="preserve" text-anchor="start" x="1329.74" y="-486.77" font-family="Arial" font-size="15.00" fill="#f9b27c">OptimizerType.AC</text>
</g>
<!-- importer -->
<g id="node11" class="node">
<title>importer</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="810.22,-2136.77 479.11,-2136.77 479.11,-1956.77 810.22,-1956.77 810.22,-2136.77"/>
<text xml:space="preserve" text-anchor="start" x="608.17" y="-2115.57" font-family="Arial" font-size="20.00" fill="#f8fafc">Importer</text>
<text xml:space="preserve" text-anchor="start" x="533.17" y="-2087.57" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl, pandapower, JAX</text>
<text xml:space="preserve" text-anchor="start" x="499.17" y="-2067.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">Normalizes a raw grid into a processed grid</text>
<text xml:space="preserve" text-anchor="start" x="571.67" y="-2049.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">folder and derives the</text>
<text xml:space="preserve" text-anchor="start" x="502.67" y="-2031.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">solver artifacts. Most of it depends only on</text>
<text xml:space="preserve" text-anchor="start" x="567.67" y="-2013.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">the initial grid topology,</text>
<text xml:space="preserve" text-anchor="start" x="530.17" y="-1995.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">so it can run before the forecast is</text>
</g>
<!-- dcoptimizer -->
<g id="node12" class="node">
<title>dcoptimizer</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="822.22,-1556.77 467.11,-1556.77 467.11,-1376.77 822.22,-1376.77 822.22,-1556.77"/>
<text xml:space="preserve" text-anchor="start" x="585.17" y="-1535.57" font-family="Arial" font-size="20.00" fill="#eef2ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="591.67" y="-1507.57" font-family="Arial" font-size="13.00" fill="#c7d2fe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="499.17" y="-1487.97" font-family="Arial" font-size="15.00" fill="#c7d2fe">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="585.67" y="-1469.97" font-family="Arial" font-size="15.00" fill="#c7d2fe">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="487.17" y="-1451.97" font-family="Arial" font-size="15.00" fill="#c7d2fe">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="570.67" y="-1433.97" font-family="Arial" font-size="15.00" fill="#c7d2fe">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="490.17" y="-1415.97" font-family="Arial" font-size="15.00" fill="#c7d2fe">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acvalidator -->
<g id="node13" class="node">
<title>acvalidator</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="813.72,-1846.77 475.61,-1846.77 475.61,-1666.77 813.72,-1666.77 813.72,-1846.77"/>
<text xml:space="preserve" text-anchor="start" x="589.17" y="-1825.57" font-family="Arial" font-size="20.00" fill="#f8fafc">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="543.17" y="-1797.57" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="510.67" y="-1777.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="559.17" y="-1759.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="495.67" y="-1741.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="569.67" y="-1723.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="515.67" y="-1705.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">candidate it evaluates gets a full result</text>
</g>
<!-- lfservice -->
<g id="node14" class="node">
<title>lfservice</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="806.22,-2426.77 483.11,-2426.77 483.11,-2246.77 806.22,-2246.77 806.22,-2426.77"/>
<text xml:space="preserve" text-anchor="start" x="557.67" y="-2405.57" font-family="Arial" font-size="20.00" fill="#f8fafc">AC loadflow service</text>
<text xml:space="preserve" text-anchor="start" x="587.67" y="-2377.57" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="523.67" y="-2357.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">A standalone N&#45;1 service on its own</text>
<text xml:space="preserve" text-anchor="start" x="515.17" y="-2339.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow_commands / loadflow_results</text>
<text xml:space="preserve" text-anchor="start" x="503.17" y="-2321.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">/ loadflow_heartbeat topics. Present in the</text>
<text xml:space="preserve" text-anchor="start" x="557.67" y="-2303.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">codebase but off the main</text>
<text xml:space="preserve" text-anchor="start" x="507.67" y="-2285.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">path: dev&#45;deployment does not create its</text>
</g>
<!-- client -->
<g id="node15" class="node">
<title>client</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="813.72,-190.77 475.61,-190.77 475.61,-10.77 813.72,-10.77 813.72,-190.77"/>
<text xml:space="preserve" text-anchor="start" x="513.67" y="-159.77" font-family="Arial" font-size="20.00" fill="#f8fafc">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="495.67" y="-131.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="571.17" y="-113.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="504.17" y="-95.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="572.67" y="-77.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="498.67" y="-59.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">In operational use the whole run must finish</text>
</g>
<!-- unprocessedgridstore -->
<g id="node16" class="node">
<title>unprocessedgridstore</title>
<path fill="#64748b" stroke="#475569" stroke-width="2" d="M322.11,-2128.67C322.11,-2138.71 249.92,-2146.87 161.06,-2146.87 72.19,-2146.87 0,-2138.71 0,-2128.67 0,-2128.67 0,-1964.87 0,-1964.87 0,-1954.83 72.19,-1946.67 161.06,-1946.67 249.92,-1946.67 322.11,-1954.83 322.11,-1964.87 322.11,-1964.87 322.11,-2128.67 322.11,-2128.67"/>
<path fill="none" stroke="#475569" stroke-width="2" d="M322.11,-2128.67C322.11,-2118.63 249.92,-2110.47 161.06,-2110.47 72.19,-2110.47 0,-2118.63 0,-2128.67"/>
<text xml:space="preserve" text-anchor="start" x="58.56" y="-2115.57" font-family="Arial" font-size="20.00" fill="#f8fafc">Unprocessed grid store</text>
<text xml:space="preserve" text-anchor="start" x="85.56" y="-2087.57" font-family="Arial" font-size="13.00" fill="#cbd5e1">fsspec AbstractFileSystem</text>
<text xml:space="preserve" text-anchor="start" x="31.06" y="-2067.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where the source grid files land before</text>
<text xml:space="preserve" text-anchor="start" x="47.06" y="-2049.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">anything touches them. The same</text>
<text xml:space="preserve" text-anchor="start" x="20.06" y="-2031.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">kind of thing as the loadflow result store &#45;&#45;</text>
<text xml:space="preserve" text-anchor="start" x="80.06" y="-2013.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">an fsspec filesystem the</text>
<text xml:space="preserve" text-anchor="start" x="36.56" y="-1995.97" font-family="Arial" font-size="15.00" fill="#cbd5e1">worker is handed, local disk or object</text>
</g>
<!-- loadflowstore -->
<g id="node17" class="node">
<title>loadflowstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M1572.3,-3166.67C1572.3,-3176.71 1491.14,-3184.87 1391.24,-3184.87 1291.34,-3184.87 1210.18,-3176.71 1210.18,-3166.67 1210.18,-3166.67 1210.18,-3002.87 1210.18,-3002.87 1210.18,-2992.83 1291.34,-2984.67 1391.24,-2984.67 1491.14,-2984.67 1572.3,-2992.83 1572.3,-3002.87 1572.3,-3002.87 1572.3,-3166.67 1572.3,-3166.67"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M1572.3,-3166.67C1572.3,-3156.63 1491.14,-3148.47 1391.24,-3148.47 1291.34,-3148.47 1210.18,-3156.63 1210.18,-3166.67"/>
<text xml:space="preserve" text-anchor="start" x="1300.24" y="-3153.57" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="1325.24" y="-3125.57" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1285.24" y="-3105.97" font-family="Arial" font-size="15.00" fill="#c2f0c2">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="1230.24" y="-3087.97" font-family="Arial" font-size="15.00" fill="#c2f0c2">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="1246.74" y="-3069.97" font-family="Arial" font-size="15.00" fill="#c2f0c2">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="1240.74" y="-3051.97" font-family="Arial" font-size="15.00" fill="#c2f0c2">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="1305.74" y="-3033.97" font-family="Arial" font-size="15.00" fill="#c2f0c2">topology it evaluates gets</text>
</g>
<!-- downstream -->
<g id="node18" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2300.37,-1668.77 1978.26,-1668.77 1978.26,-1488.77 2300.37,-1488.77 2300.37,-1668.77"/>
<text xml:space="preserve" text-anchor="start" x="1998.31" y="-1619.77" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="2001.81" y="-1591.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="2027.81" y="-1573.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="2031.81" y="-1555.77" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- gridsnapshot&#45;&gt;importer -->
<g id="edge4" class="edge">
<title>gridsnapshot&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1224.3,-2859.39C1112.82,-2896.96 970.64,-2918.47 882.22,-2832.77 856.64,-2807.97 879.49,-2222.93 862.22,-2191.77 850.9,-2171.34 835.34,-2153.25 817.71,-2137.39"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="819.7,-2135.65 812.32,-2132.71 816.26,-2139.61 819.7,-2135.65"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="995.22,-2892.32 995.22,-2915.12 1020.22,-2915.12 1020.22,-2892.32 995.22,-2892.32"/>
<text xml:space="preserve" text-anchor="start" x="998.22" y="-2899.82" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- gridsnapshot&#45;&gt;acvalidator -->
<g id="edge5" class="edge">
<title>gridsnapshot&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1224.22,-2713.37C1195.92,-2693.68 1170.22,-2669.36 1153.22,-2639.77 1126.53,-2593.28 1171.52,-1712.48 1133.22,-1674.97 1051.76,-1595.18 923.55,-1620.24 819.04,-1662.8"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="818.04,-1660.38 812.12,-1665.68 820.05,-1665.23 818.04,-1660.38"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="976.72,-1674.97 976.72,-1697.77 1038.72,-1697.77 1038.72,-1674.97 976.72,-1674.97"/>
<text xml:space="preserve" text-anchor="start" x="979.72" y="-1694.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">base grid</text>
</g>
<!-- staticinfo&#45;&gt;dcoptimizer -->
<g id="edge15" class="edge">
<title>staticinfo&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1218.72,-2129.63C1192.57,-2110.63 1169.06,-2087.53 1153.22,-2059.77 1134.63,-2027.17 1159.95,-735.31 1133.22,-708.97 1053.76,-630.67 972.09,-642.88 882.22,-708.97 774.9,-787.9 694.77,-1179.75 661.97,-1366.92"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="659.42,-1366.22 660.72,-1374.05 664.6,-1367.12 659.42,-1366.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="909.72,-708.97 909.72,-731.77 1105.72,-731.77 1105.72,-708.97 909.72,-708.97"/>
<text xml:space="preserve" text-anchor="start" x="912.72" y="-728.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">loaded onto the GPU at startup</text>
</g>
<!-- actionset&#45;&gt;acvalidator -->
<g id="edge16" class="edge">
<title>actionset&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1202.75,-2406.91C1183.14,-2390.56 1165.82,-2371.6 1153.22,-2349.77 1130.66,-2310.64 1165.55,-1568.51 1133.22,-1536.97 1053.38,-1459.06 970.71,-1469.03 882.22,-1536.97 854.93,-1557.92 882.25,-1583.78 862.22,-1611.77 849.55,-1629.48 833.99,-1645.66 817.08,-1660.25"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="815.59,-1658.07 811.53,-1664.9 818.97,-1662.09 815.59,-1658.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="943.22,-1536.97 943.22,-1559.77 1072.22,-1559.77 1072.22,-1536.97 943.22,-1536.97"/>
<text xml:space="preserve" text-anchor="start" x="946.22" y="-1556.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">to realize topologies</text>
</g>
<!-- snapshots&#45;&gt;downstream -->
<g id="edge26" class="edge">
<title>snapshots&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1564.34,-1837.26C1685.97,-1782.48 1848.22,-1709.41 1969.18,-1654.94"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1970.03,-1657.43 1975.79,-1651.96 1967.87,-1652.65 1970.03,-1657.43"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1649.26,-1796.5 1649.26,-1836.1 1918.26,-1836.1 1918.26,-1796.5 1649.26,-1796.5"/>
<text xml:space="preserve" text-anchor="start" x="1652.26" y="-1833.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">UCTE, DGS, OpenRAO summaries, single</text>
<text xml:space="preserve" text-anchor="start" x="1652.26" y="-1816.3" font-family="Arial" font-size="14.00" fill="#c9c9c9">line diagrams</text>
</g>
<!-- importercommands&#45;&gt;importer -->
<g id="edge6" class="edge">
<title>importercommands&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1201.29,-868.59C1182.28,-884.36 1165.49,-902.65 1153.22,-923.77 1123.05,-975.71 1174.24,-1960.89 1133.22,-2004.77 1055.73,-2087.67 926.38,-2095.5 820.38,-2084.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="820.96,-2081.45 813.21,-2083.2 820.36,-2086.66 820.96,-2081.45"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="941.22,-2088.19 941.22,-2110.99 1074.22,-2110.99 1074.22,-2088.19 941.22,-2088.19"/>
<text xml:space="preserve" text-anchor="start" x="944.22" y="-2107.99" font-family="Arial" font-size="14.00" fill="#c9c9c9">consumes command</text>
</g>
<!-- commands&#45;&gt;dcoptimizer -->
<g id="edge7" class="edge">
<title>commands&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1193.26,-227.06C1086.97,-241.98 961.66,-278.15 882.22,-362.97 745.61,-508.83 676.53,-1122.84 654.09,-1366.51"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="651.49,-1366.05 653.42,-1373.76 656.72,-1366.53 651.49,-1366.05"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="941.22,-362.97 941.22,-385.77 1074.22,-385.77 1074.22,-362.97 941.22,-362.97"/>
<text xml:space="preserve" text-anchor="start" x="944.22" y="-382.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">consumes command</text>
</g>
<!-- commands&#45;&gt;acvalidator -->
<g id="edge8" class="edge">
<title>commands&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1202.25,-304.76C1183,-320.51 1165.9,-338.76 1153.22,-359.77 1108.85,-433.31 1183.11,-675.85 1133.22,-745.77 1063.37,-843.69 950.82,-748.16 882.22,-846.97 833.75,-916.8 903.24,-1537.31 862.22,-1611.77 851.67,-1630.93 837.38,-1648.02 821.14,-1663.15"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="819.63,-1660.98 815.8,-1667.94 823.14,-1664.89 819.63,-1660.98"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="910.72,-846.97 910.72,-869.77 1104.72,-869.77 1104.72,-846.97 910.72,-846.97"/>
<text xml:space="preserve" text-anchor="start" x="913.72" y="-866.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">consumes the same command</text>
</g>
<!-- results&#45;&gt;acvalidator -->
<g id="edge20" class="edge">
<title>results&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1218.56,-1131.31C1079.47,-1186.87 903.22,-1261.7 882.22,-1291.97 841.65,-1350.48 897.7,-1550.03 862.22,-1611.77 851.47,-1630.48 837.21,-1647.25 821.12,-1662.16"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="819.7,-1659.9 815.86,-1666.86 823.2,-1663.82 819.7,-1659.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="961.22,-1291.97 961.22,-1314.77 1054.22,-1314.77 1054.22,-1291.97 961.22,-1291.97"/>
<text xml:space="preserve" text-anchor="start" x="964.22" y="-1311.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">DC topologies</text>
</g>
<!-- results&#45;&gt;downstream -->
<g id="edge21" class="edge">
<title>results&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1541.52,-1150.62C1570.92,-1168.35 1601.3,-1187.28 1629.26,-1205.77 1764.04,-1294.94 1912.5,-1405.19 2014.14,-1482.62"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2012.33,-1484.54 2019.88,-1487 2015.51,-1480.37 2012.33,-1484.54"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1687.76,-1403.92 1687.76,-1426.72 1879.76,-1426.72 1879.76,-1403.92 1687.76,-1403.92"/>
<text xml:space="preserve" text-anchor="start" x="1690.76" y="-1423.72" font-family="Arial" font-size="14.00" fill="#c9c9c9">validated topologies for review</text>
</g>
<!-- importer&#45;&gt;gridsnapshot -->
<g id="edge9" class="edge">
<title>importer&#45;&gt;gridsnapshot</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M810.01,-2131.11C830.71,-2148.44 849.18,-2168.61 862.22,-2191.77 886.35,-2234.6 851.94,-2595.04 882.22,-2633.77 960.06,-2733.33 1100.62,-2769.53 1214.19,-2781.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1213.65,-2784.27 1221.38,-2782.42 1214.18,-2779.05 1213.65,-2784.27"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="995.22,-2767.09 995.22,-2789.89 1020.22,-2789.89 1020.22,-2767.09 995.22,-2767.09"/>
<text xml:space="preserve" text-anchor="start" x="998.22" y="-2774.59" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- importer&#45;&gt;staticinfo -->
<g id="edge12" class="edge">
<title>importer&#45;&gt;staticinfo</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M810.19,-2132.5C830.59,-2149.56 848.92,-2169.28 862.22,-2191.77 888.01,-2235.35 844.88,-2384.57 882.22,-2418.77 964.49,-2494.11 1044.04,-2485.78 1133.22,-2418.77 1158.75,-2399.59 1133.85,-2375.15 1153.22,-2349.77 1169.18,-2328.86 1189.1,-2310.28 1210.56,-2293.98"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1211.95,-2296.22 1216.42,-2289.65 1208.83,-2291.99 1211.95,-2296.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="995.22,-2472.19 995.22,-2494.99 1020.22,-2494.99 1020.22,-2472.19 995.22,-2472.19"/>
<text xml:space="preserve" text-anchor="start" x="998.22" y="-2479.69" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- importer&#45;&gt;actionset -->
<g id="edge13" class="edge">
<title>importer&#45;&gt;actionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M810.22,-2131.74C830.76,-2148.94 849.11,-2168.91 862.22,-2191.77 880.5,-2223.63 855.48,-2496.59 882.22,-2521.77 963.22,-2598 1087.47,-2592.7 1192.81,-2567.63"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1193.35,-2570.2 1200.01,-2565.86 1192.1,-2565.1 1193.35,-2570.2"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="882.22,-2584.28 882.22,-2607.08 1133.22,-2607.08 1133.22,-2584.28 882.22,-2584.28"/>
<text xml:space="preserve" text-anchor="start" x="885.22" y="-2604.08" font-family="Arial" font-size="14.00" fill="#c9c9c9">the same actions as physical switchings</text>
</g>
<!-- importer&#45;&gt;importerresults -->
<g id="edge10" class="edge">
<title>importer&#45;&gt;importerresults</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M809.97,-2136.16C829.81,-2152.51 848.03,-2171.06 862.22,-2191.77 885.15,-2225.21 850.57,-2255.44 882.22,-2280.77 969.32,-2350.47 1052.85,-2358.13 1133.22,-2280.77 1175.12,-2240.44 1122.57,-1807.18 1153.22,-1757.77 1169.2,-1732.02 1192.24,-1711.04 1217.74,-1694.08"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1219.04,-1696.37 1223.93,-1690.11 1216.2,-1691.95 1219.04,-1696.37"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="914.72,-2335.94 914.72,-2358.74 1100.72,-2358.74 1100.72,-2335.94 914.72,-2335.94"/>
<text xml:space="preserve" text-anchor="start" x="917.72" y="-2355.74" font-family="Arial" font-size="14.00" fill="#c9c9c9">PreprocessingSuccessResult</text>
</g>
<!-- importer&#45;&gt;importerheartbeat -->
<g id="edge11" class="edge">
<title>importer&#45;&gt;importerheartbeat</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M795.31,-2136.71C903.16,-2188.71 1044.75,-2228.59 1133.22,-2142.77 1159.36,-2117.42 1134.75,-1519.14 1153.22,-1487.77 1169.51,-1460.13 1193.5,-1437.3 1220.05,-1418.65"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1221.52,-1420.83 1226.25,-1414.45 1218.57,-1416.49 1221.52,-1420.83"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="908.72,-2192.78 908.72,-2215.58 1106.72,-2215.58 1106.72,-2192.78 908.72,-2192.78"/>
<text xml:space="preserve" text-anchor="start" x="911.72" y="-2212.58" font-family="Arial" font-size="14.00" fill="#c9c9c9">PreprocessHeartbeat per stage</text>
</g>
<!-- importer&#45;&gt;loadflowstore -->
<g id="edge14" class="edge">
<title>importer&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M810.13,-2130.68C830.89,-2148.08 849.34,-2168.39 862.22,-2191.77 902.46,-2264.78 831.01,-2875.99 882.22,-2941.77 956.18,-3036.76 1088.06,-3071.92 1198.91,-3083.63"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1198.49,-3086.23 1206.21,-3084.36 1199.01,-3081 1198.49,-3086.23"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="942.22,-3072.06 942.22,-3094.86 1073.22,-3094.86 1073.22,-3072.06 942.22,-3072.06"/>
<text xml:space="preserve" text-anchor="start" x="945.22" y="-3091.86" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial AC N&#45;1 results</text>
</g>
<!-- dcoptimizer&#45;&gt;results -->
<g id="edge18" class="edge">
<title>dcoptimizer&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M655.67,-1376.86C680.73,-1169.91 753.59,-670.04 882.22,-570.97 970.6,-502.9 1052.14,-494.35 1133.22,-570.97 1161.76,-597.93 1132.86,-890.2 1153.22,-923.77 1167.6,-947.47 1187.6,-967.65 1209.94,-984.71"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1208.23,-986.72 1215.83,-989.06 1211.35,-982.49 1208.23,-986.72"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="909.22,-570.97 909.22,-593.77 1106.22,-593.77 1106.22,-570.97 909.22,-570.97"/>
<text xml:space="preserve" text-anchor="start" x="912.22" y="-590.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">TopologyPushResult per epoch</text>
</g>
<!-- dcoptimizer&#45;&gt;heartbeat -->
<g id="edge19" class="edge">
<title>dcoptimizer&#45;&gt;heartbeat</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M653.02,-1377.04C673.4,-1153.28 738.83,-580.39 882.22,-466.97 973.32,-394.91 1104.74,-405.73 1211.08,-433.33"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1210.12,-435.79 1218.04,-435.18 1211.47,-430.72 1210.12,-435.79"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="920.22,-466.97 920.22,-489.77 1095.22,-489.77 1095.22,-466.97 920.22,-466.97"/>
<text xml:space="preserve" text-anchor="start" x="923.22" y="-486.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">OptimizationStatsHeartbeat</text>
</g>
<!-- acvalidator&#45;&gt;snapshots -->
<g id="edge25" class="edge">
<title>acvalidator&#45;&gt;snapshots</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.5,-1668.06C832.54,-1651.69 849.6,-1632.95 862.22,-1611.77 886.53,-1570.96 847.1,-1430.94 882.22,-1398.97 964.71,-1323.87 1052.41,-1322.06 1133.22,-1398.97 1165.76,-1429.94 1128.25,-1765.43 1153.22,-1802.77 1167.57,-1824.23 1187.2,-1841.67 1209.06,-1855.84"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1207.55,-1857.99 1215.3,-1859.71 1210.31,-1853.52 1207.55,-1857.99"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="927.22,-1398.97 927.22,-1421.77 1088.22,-1421.77 1088.22,-1398.97 927.22,-1398.97"/>
<text xml:space="preserve" text-anchor="start" x="930.22" y="-1418.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">summaries and diagrams</text>
</g>
<!-- acvalidator&#45;&gt;results -->
<g id="edge23" class="edge">
<title>acvalidator&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.51,-1669.57C832.81,-1652.9 849.95,-1633.67 862.22,-1611.77 888.8,-1564.35 845.22,-1162.78 882.22,-1122.97 962.95,-1036.11 1097.75,-1024.05 1208.21,-1032.38"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1207.78,-1034.97 1215.47,-1032.97 1208.21,-1029.74 1207.78,-1034.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="937.72,-1122.97 937.72,-1145.77 1077.72,-1145.77 1077.72,-1122.97 937.72,-1122.97"/>
<text xml:space="preserve" text-anchor="start" x="940.72" y="-1142.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC&#45;validated Strategy</text>
</g>
<!-- acvalidator&#45;&gt;heartbeat -->
<g id="edge24" class="edge">
<title>acvalidator&#45;&gt;heartbeat</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.69,-1669.67C832.97,-1652.98 850.05,-1633.73 862.22,-1611.77 896.02,-1550.83 842.32,-1042.09 882.22,-984.97 951.1,-886.36 1061.88,-980.6 1133.22,-883.77 1165.23,-840.33 1124.64,-687.54 1153.22,-641.77 1168.37,-617.52 1189.34,-596.91 1212.6,-579.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1213.92,-581.86 1218.46,-575.34 1210.85,-577.6 1213.92,-581.86"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="920.22,-984.97 920.22,-1007.77 1095.22,-1007.77 1095.22,-984.97 920.22,-984.97"/>
<text xml:space="preserve" text-anchor="start" x="923.22" y="-1004.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">OptimizationStatsHeartbeat</text>
</g>
<!-- acvalidator&#45;&gt;loadflowstore -->
<g id="edge22" class="edge">
<title>acvalidator&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.7,-1823.69C836.68,-1832.59 860,-1841.49 882.22,-1849.77 993.06,-1891.06 1063.94,-1842.31 1133.22,-1938.17 1166.57,-1984.31 1123.1,-2914.46 1153.22,-2962.77 1165.54,-2982.52 1182.17,-2999.23 1200.94,-3013.31"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1199.33,-3015.39 1206.95,-3017.63 1202.39,-3011.12 1199.33,-3015.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="901.72,-1938.17 901.72,-1977.77 1113.72,-1977.77 1113.72,-1938.17 901.72,-1938.17"/>
<text xml:space="preserve" text-anchor="start" x="904.72" y="-1974.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="904.72" y="-1957.97" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- client&#45;&gt;importercommands -->
<g id="edge1" class="edge">
<title>client&#45;&gt;importercommands</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.45,-70.37C920.7,-61.44 1054.06,-71.76 1133.22,-156.97 1169.92,-196.47 1125.61,-595.46 1153.22,-641.77 1164.02,-659.88 1178.12,-675.93 1194.07,-690.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1192.32,-692.07 1199.73,-694.94 1195.73,-688.07 1192.32,-692.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="914.22,-156.97 914.22,-179.77 1101.22,-179.77 1101.22,-156.97 914.22,-156.97"/>
<text xml:space="preserve" text-anchor="start" x="917.22" y="-176.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">StartPreprocessingCommand</text>
</g>
<!-- client&#45;&gt;commands -->
<g id="edge2" class="edge">
<title>client&#45;&gt;commands</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M813.48,-32.54C909.08,-3.5 1030.02,16.43 1133.22,-18.97 1196.27,-40.59 1255.78,-84.7 1301.62,-125.98"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1299.6,-127.69 1306.91,-130.8 1303.14,-123.81 1299.6,-127.69"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="919.72,-18.97 919.72,-41.77 1095.72,-41.77 1095.72,-18.97 919.72,-18.97"/>
<text xml:space="preserve" text-anchor="start" x="922.72" y="-38.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">StartOptimizationCommand</text>
</g>
<!-- unprocessedgridstore&#45;&gt;importer -->
<g id="edge3" class="edge">
<title>unprocessedgridstore&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M323.1,-2046.77C369.97,-2046.77 421.39,-2046.77 469.06,-2046.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="468.97,-2049.39 476.47,-2046.77 468.97,-2044.14 468.97,-2049.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="382.11,-2046.77 382.11,-2069.57 407.11,-2069.57 407.11,-2046.77 382.11,-2046.77"/>
<text xml:space="preserve" text-anchor="start" x="385.11" y="-2054.27" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- loadflowstore&#45;&gt;acvalidator -->
<g id="edge17" class="edge">
<title>loadflowstore&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1209.36,-3019.42C1187.15,-3004.13 1167.34,-2985.46 1153.22,-2962.77 1119.48,-2908.52 1176.15,-1860.29 1133.22,-1812.97 1057.46,-1729.45 929.73,-1717.26 824.01,-1724.98"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="823.81,-1722.36 816.55,-1725.57 824.23,-1727.59 823.81,-1722.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="924.72,-1812.97 924.72,-1835.77 1090.72,-1835.77 1090.72,-1812.97 924.72,-1812.97"/>
<text xml:space="preserve" text-anchor="start" x="927.72" y="-1832.77" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial loadflow as baseline</text>
</g>
</g>
</svg>`;case`importerInternals`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="4146pt" height="6023pt"
 viewBox="0.00 0.00 4146.00 6023.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 6007.85)">
<g id="clust1" class="cluster">
<title>cluster_unprocessedgridstore</title>
<polygon fill="#3e4651" stroke="#2d333d" points="8,-5650.4 8,-5915.6 442,-5915.6 442,-5650.4 8,-5650.4"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-5902.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">UNPROCESSED GRID STORE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_processedgrid</title>
<polygon fill="#3a404a" stroke="#292f37" points="788,-8 788,-374.4 4102,-374.4 4102,-8 788,-8"/>
<text xml:space="preserve" text-anchor="start" x="796" y="-360.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">PROCESSED GRID FOLDER</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_staticinfo</title>
<polygon fill="#3e4651" stroke="#2d333d" points="828,-48 828,-313.2 1240,-313.2 1240,-48 828,-48"/>
<text xml:space="preserve" text-anchor="start" x="836" y="-299.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">STATIC_INFORMATION.HDF5</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_importer</title>
<polygon fill="#3a404a" stroke="#292f37" points="450,-391.4 450,-5984.8 2244,-5984.8 2244,-391.4 450,-391.4"/>
<text xml:space="preserve" text-anchor="start" x="458" y="-5971.35" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">IMPORTER</text>
</g>
<g id="clust5" class="cluster">
<title>cluster_importstage</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="1292,-3034 1292,-5923.6 2204,-5923.6 2204,-3034 1292,-3034"/>
<text xml:space="preserve" text-anchor="start" x="1300" y="-5910.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">IMPORTER PREPROCESSING</text>
</g>
<g id="clust6" class="cluster">
<title>cluster_dcpreprocess</title>
<polygon fill="#2225aa" stroke="#2a2490" points="490,-431.4 490,-3638 1252,-3638 1252,-431.4 490,-431.4"/>
<text xml:space="preserve" text-anchor="start" x="498" y="-3624.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c7d2fe" fill-opacity="0.701961">LOAD_GRID (DC PREPROCESSING)</text>
</g>
<!-- inputnminus1 -->
<g id="node1" class="node">
<title>inputnminus1</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="410.06,-5862.4 39.94,-5862.4 39.94,-5682.4 410.06,-5682.4 410.06,-5862.4"/>
<text xml:space="preserve" text-anchor="start" x="98.5" y="-5831.4" font-family="Arial" font-size="20.00" fill="#f8fafc">Input N&#45;1 definition (optional)</text>
<text xml:space="preserve" text-anchor="start" x="60" y="-5803.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">A JSON dump of Nminus1Definition with element</text>
<text xml:space="preserve" text-anchor="start" x="147.5" y="-5785.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">ids already resolved by</text>
<text xml:space="preserve" text-anchor="start" x="150" y="-5767.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">its producer, stored as</text>
<text xml:space="preserve" text-anchor="start" x="87.5" y="-5749.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">input_nminus1_definition.json in the data</text>
<text xml:space="preserve" text-anchor="start" x="160.5" y="-5731.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">folder or passed as</text>
</g>
<!-- branchactionset -->
<g id="node2" class="node">
<title>branchactionset</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1208.06,-260 859.94,-260 859.94,-80 1208.06,-80 1208.06,-260"/>
<text xml:space="preserve" text-anchor="start" x="960.5" y="-229" font-family="Arial" font-size="20.00" fill="#f8fafc">BranchActionSet</text>
<text xml:space="preserve" text-anchor="start" x="880" y="-201" font-family="Arial" font-size="15.00" fill="#cbd5e1">What the DC&#45;Optimizer actually samples from</text>
<text xml:space="preserve" text-anchor="start" x="955.5" y="-183" font-family="Arial" font-size="15.00" fill="#cbd5e1">&#45;&#45; a different asset from</text>
<text xml:space="preserve" text-anchor="start" x="892.5" y="-165" font-family="Arial" font-size="15.00" fill="#cbd5e1">action_set.json, in a different format and a</text>
<text xml:space="preserve" text-anchor="start" x="993" y="-147" font-family="Arial" font-size="15.00" fill="#cbd5e1">different file.</text>
<text xml:space="preserve" text-anchor="start" x="898" y="-129" font-family="Arial" font-size="15.00" fill="#cbd5e1">Padded boolean arrays (branch_actions,</text>
</g>
<!-- assettopomaster -->
<g id="node3" class="node">
<title>assettopomaster</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2713.56,-260 2240.44,-260 2240.44,-80 2713.56,-80 2713.56,-260"/>
<text xml:space="preserve" text-anchor="start" x="2260.5" y="-229" font-family="Arial" font-size="20.00" fill="#f8fafc">initial_topology/asset_topology_master_data.json</text>
<text xml:space="preserve" text-anchor="start" x="2334.5" y="-201" font-family="Arial" font-size="15.00" fill="#cbd5e1">A serialized MasterAssetTopology, and the</text>
<text xml:space="preserve" text-anchor="start" x="2404.5" y="-183" font-family="Arial" font-size="15.00" fill="#cbd5e1">only form of the asset</text>
<text xml:space="preserve" text-anchor="start" x="2336" y="-165" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology that gets a file of its own. Written</text>
<text xml:space="preserve" text-anchor="start" x="2389.5" y="-147" font-family="Arial" font-size="15.00" fill="#cbd5e1">by the importer, read back</text>
<text xml:space="preserve" text-anchor="start" x="2325" y="-129" font-family="Arial" font-size="15.00" fill="#cbd5e1">at the start of DC preprocessing. The runtime</text>
</g>
<!-- masks -->
<g id="node4" class="node">
<title>masks</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="4062.06,-260 3701.94,-260 3701.94,-80 4062.06,-80 4062.06,-260"/>
<text xml:space="preserve" text-anchor="start" x="3828.5" y="-229" font-family="Arial" font-size="20.00" fill="#f8fafc">masks/*.npy</text>
<text xml:space="preserve" text-anchor="start" x="3722" y="-201" font-family="Arial" font-size="15.00" fill="#cbd5e1">~35 boolean and weight masks per asset class:</text>
<text xml:space="preserve" text-anchor="start" x="3797.5" y="-183" font-family="Arial" font-size="15.00" fill="#cbd5e1">which branches count for</text>
<text xml:space="preserve" text-anchor="start" x="3747.5" y="-165" font-family="Arial" font-size="15.00" fill="#cbd5e1">N&#45;1, which are disconnectable, overload</text>
<text xml:space="preserve" text-anchor="start" x="3788" y="-147" font-family="Arial" font-size="15.00" fill="#cbd5e1">weights, TSO/DSO borders,</text>
<text xml:space="preserve" text-anchor="start" x="3849" y="-129" font-family="Arial" font-size="15.00" fill="#cbd5e1">blacklists.</text>
</g>
<!-- gridsnapshot -->
<g id="node5" class="node">
<title>gridsnapshot</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="3158.06,-260 2823.94,-260 2823.94,-80 3158.06,-80 3158.06,-260"/>
<text xml:space="preserve" text-anchor="start" x="2903" y="-202" font-family="Arial" font-size="20.00" fill="#f8fafc">grid.xiidm / grid.json</text>
<text xml:space="preserve" text-anchor="start" x="2844" y="-174" font-family="Arial" font-size="15.00" fill="#cbd5e1">The normalized backend grid, written by the</text>
<text xml:space="preserve" text-anchor="start" x="2961.5" y="-156" font-family="Arial" font-size="15.00" fill="#cbd5e1">importer.</text>
</g>
<!-- nminus1 -->
<g id="node6" class="node">
<title>nminus1</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="3592.06,-260 3267.94,-260 3267.94,-80 3592.06,-80 3592.06,-260"/>
<text xml:space="preserve" text-anchor="start" x="3327" y="-229" font-family="Arial" font-size="20.00" fill="#f8fafc">nminus1_definition.json</text>
<text xml:space="preserve" text-anchor="start" x="3288" y="-201" font-family="Arial" font-size="15.00" fill="#cbd5e1">The grid&#45;validated contingency set, written</text>
<text xml:space="preserve" text-anchor="start" x="3349.5" y="-183" font-family="Arial" font-size="15.00" fill="#cbd5e1">by the importer from the</text>
<text xml:space="preserve" text-anchor="start" x="3306.5" y="-165" font-family="Arial" font-size="15.00" fill="#cbd5e1">input N&#45;1 definition (journey A) or the</text>
<text xml:space="preserve" text-anchor="start" x="3331.5" y="-147" font-family="Arial" font-size="15.00" fill="#cbd5e1">network masks (journey B). It</text>
<text xml:space="preserve" text-anchor="start" x="3312" y="-129" font-family="Arial" font-size="15.00" fill="#cbd5e1">is a separate file from the input N&#45;1</text>
</g>
<!-- actionset -->
<g id="node7" class="node">
<title>actionset</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1695.56,-260 1318.44,-260 1318.44,-80 1695.56,-80 1695.56,-260"/>
<text xml:space="preserve" text-anchor="start" x="1338.5" y="-229" font-family="Arial" font-size="20.00" fill="#f8fafc">action_set.json + action_set_diffs.hdf5</text>
<text xml:space="preserve" text-anchor="start" x="1368" y="-201" font-family="Arial" font-size="15.00" fill="#cbd5e1">The same action space in physical terms:</text>
<text xml:space="preserve" text-anchor="start" x="1410.5" y="-183" font-family="Arial" font-size="15.00" fill="#cbd5e1">station&#45;local reconfigurations</text>
<text xml:space="preserve" text-anchor="start" x="1344.5" y="-165" font-family="Arial" font-size="15.00" fill="#cbd5e1">A and disconnectable branches D, expressed as</text>
<text xml:space="preserve" text-anchor="start" x="1427" y="-147" font-family="Arial" font-size="15.00" fill="#cbd5e1">switch positions against</text>
<text xml:space="preserve" text-anchor="start" x="1444" y="-129" font-family="Arial" font-size="15.00" fill="#cbd5e1">the asset topology.</text>
</g>
<!-- dcnminus1 -->
<g id="node8" class="node">
<title>dcnminus1</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2130.06,-260 1805.94,-260 1805.94,-80 2130.06,-80 2130.06,-260"/>
<text xml:space="preserve" text-anchor="start" x="1848.5" y="-220" font-family="Arial" font-size="20.00" fill="#f8fafc">dc_nminus1_definition.json</text>
<text xml:space="preserve" text-anchor="start" x="1826" y="-192" font-family="Arial" font-size="15.00" fill="#cbd5e1">The contingencies DC computes, in solver</text>
<text xml:space="preserve" text-anchor="start" x="1869.5" y="-174" font-family="Arial" font-size="15.00" fill="#cbd5e1">order. DC preprocessing also</text>
<text xml:space="preserve" text-anchor="start" x="1841" y="-156" font-family="Arial" font-size="15.00" fill="#cbd5e1">writes them as the N&#45;1 definition if the</text>
<text xml:space="preserve" text-anchor="start" x="1914" y="-138" font-family="Arial" font-size="15.00" fill="#cbd5e1">folder has none.</text>
</g>
<!-- loadgrid -->
<g id="node9" class="node">
<title>loadgrid</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1696.56,-5862.4 1335.44,-5862.4 1335.44,-5682.4 1696.56,-5682.4 1696.56,-5862.4"/>
<text xml:space="preserve" text-anchor="start" x="1444.5" y="-5804.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Load and merge</text>
<text xml:space="preserve" text-anchor="start" x="1355.5" y="-5776.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">Parse UCTE/CGMES/PowerFactory. Dominates</text>
<text xml:space="preserve" text-anchor="start" x="1418.5" y="-5758.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">importer runtime on CGMES.</text>
</g>
<!-- inputvalidation -->
<g id="node10" class="node">
<title>inputvalidation</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1700.06,-5539.6 1331.94,-5539.6 1331.94,-5359.6 1700.06,-5359.6 1700.06,-5539.6"/>
<text xml:space="preserve" text-anchor="start" x="1352" y="-5508.6" font-family="Arial" font-size="20.00" fill="#eff6ff">load_nminus1_definition_for_network</text>
<text xml:space="preserve" text-anchor="start" x="1372.5" y="-5480.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">Journey A, stage 1. Validates the input N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="1446.5" y="-5462.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">definition against the</text>
<text xml:space="preserve" text-anchor="start" x="1378" y="-5444.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">grid before its three&#45;winding transformers</text>
<text xml:space="preserve" text-anchor="start" x="1437" y="-5426.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">are converted, because</text>
<text xml:space="preserve" text-anchor="start" x="1376.5" y="-5408.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">the input references them by their original</text>
</g>
<!-- trafoconversion -->
<g id="node11" class="node">
<title>trafoconversion</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2164.06,-5539.6 1809.94,-5539.6 1809.94,-5359.6 2164.06,-5359.6 2164.06,-5539.6"/>
<text xml:space="preserve" text-anchor="start" x="1876" y="-5499.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Convert 3W transformers</text>
<text xml:space="preserve" text-anchor="start" x="1830" y="-5471.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">Replaces every three&#45;winding transformer &lt;id&gt;</text>
<text xml:space="preserve" text-anchor="start" x="1917" y="-5453.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">in the network by the</text>
<text xml:space="preserve" text-anchor="start" x="1835" y="-5435.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">two&#45;winding legs &lt;id&gt;&#45;Leg1/2/3 around a star</text>
<text xml:space="preserve" text-anchor="start" x="1943" y="-5417.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">voltage level.</text>
</g>
<!-- networkreduction -->
<g id="node12" class="node">
<title>networkreduction</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2036.56,-5216.8 1627.44,-5216.8 1627.44,-5036.8 2036.56,-5036.8 2036.56,-5216.8"/>
<text xml:space="preserve" text-anchor="start" x="1647.5" y="-5185.8" font-family="Arial" font-size="20.00" fill="#eff6ff">reduce_network_based_on_area_settings</text>
<text xml:space="preserve" text-anchor="start" x="1689.5" y="-5157.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">Optional. Keeps the voltage levels within a</text>
<text xml:space="preserve" text-anchor="start" x="1772.5" y="-5139.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">range of the view,</text>
<text xml:space="preserve" text-anchor="start" x="1700" y="-5121.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">control and N&#45;1 area voltage levels and</text>
<text xml:space="preserve" text-anchor="start" x="1765" y="-5103.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">removes the rest. In</text>
<text xml:space="preserve" text-anchor="start" x="1688.5" y="-5085.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">journey A the voltage levels of all input N&#45;1</text>
</g>
<!-- whitelists -->
<g id="node13" class="node">
<title>whitelists</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1992.02,-4894 1671.98,-4894 1671.98,-4714 1992.02,-4714 1992.02,-4894"/>
<text xml:space="preserve" text-anchor="start" x="1764.5" y="-4836" font-family="Arial" font-size="20.00" fill="#eff6ff">Apply whitelists</text>
<text xml:space="preserve" text-anchor="start" x="1698.5" y="-4808" font-family="Arial" font-size="15.00" fill="#bfdbfe">Apply the CB / black&#45; and whitelists that</text>
<text xml:space="preserve" text-anchor="start" x="1742" y="-4790" font-family="Arial" font-size="15.00" fill="#bfdbfe">scope the switchable area.</text>
</g>
<!-- convergingparams -->
<g id="node14" class="node">
<title>convergingparams</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2002.56,-4554.4 1661.44,-4554.4 1661.44,-4374.4 2002.56,-4374.4 2002.56,-4554.4"/>
<text xml:space="preserve" text-anchor="start" x="1681.5" y="-4514.4" font-family="Arial" font-size="20.00" fill="#eff6ff">find_converging_loadflow_params</text>
<text xml:space="preserve" text-anchor="start" x="1685.5" y="-4486.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">Sweep loadflow parameters and voltage init</text>
<text xml:space="preserve" text-anchor="start" x="1756" y="-4468.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">methods until the base</text>
<text xml:space="preserve" text-anchor="start" x="1701" y="-4450.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">case converges. Some grid files do not</text>
<text xml:space="preserve" text-anchor="start" x="1759.5" y="-4432.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">converge on defaults.</text>
</g>
<!-- networkmasks -->
<g id="node15" class="node">
<title>networkmasks</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1992.02,-4231.6 1671.98,-4231.6 1671.98,-4051.6 1992.02,-4051.6 1992.02,-4231.6"/>
<text xml:space="preserve" text-anchor="start" x="1743.5" y="-4200.6" font-family="Arial" font-size="20.00" fill="#eff6ff">get_network_masks</text>
<text xml:space="preserve" text-anchor="start" x="1698" y="-4172.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">Build the per&#45;asset masks from the area</text>
<text xml:space="preserve" text-anchor="start" x="1778" y="-4154.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">settings: what is</text>
<text xml:space="preserve" text-anchor="start" x="1712.5" y="-4136.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">monitored, outaged, switchable and</text>
<text xml:space="preserve" text-anchor="start" x="1717.5" y="-4118.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">disconnectable. Computed in both</text>
<text xml:space="preserve" text-anchor="start" x="1700" y="-4100.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">journeys; only journey B derives its N&#45;1</text>
</g>
<!-- nminus1definition -->
<g id="node16" class="node">
<title>nminus1definition</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1679.56,-3908.8 1352.44,-3908.8 1352.44,-3728.8 1679.56,-3728.8 1679.56,-3908.8"/>
<text xml:space="preserve" text-anchor="start" x="1400.5" y="-3868.8" font-family="Arial" font-size="20.00" fill="#eff6ff">create_nminus1_definition</text>
<text xml:space="preserve" text-anchor="start" x="1376" y="-3840.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">Builds the N&#45;1 definition saved next to the</text>
<text xml:space="preserve" text-anchor="start" x="1443.5" y="-3822.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">processed grid, in the</text>
<text xml:space="preserve" text-anchor="start" x="1372.5" y="-3804.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">same order for both journeys: source, area</text>
<text xml:space="preserve" text-anchor="start" x="1447" y="-3786.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">settings, conversion.</text>
</g>
<!-- securityanalysis -->
<g id="node17" class="node">
<title>securityanalysis</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1689.56,-3576.8 1342.44,-3576.8 1342.44,-3396.8 1689.56,-3396.8 1689.56,-3576.8"/>
<text xml:space="preserve" text-anchor="start" x="1362.5" y="-3555.6" font-family="Arial" font-size="20.00" fill="#eff6ff">import_grid_with_security_analysis</text>
<text xml:space="preserve" text-anchor="start" x="1483" y="-3527.6" font-family="Arial" font-size="13.00" fill="#bfdbfe">PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="1385.5" y="-3508" font-family="Arial" font-size="15.00" fill="#bfdbfe">The only AC N&#45;1 of the import, with the</text>
<text xml:space="preserve" text-anchor="start" x="1449" y="-3490" font-family="Arial" font-size="15.00" fill="#bfdbfe">converging loadflow</text>
<text xml:space="preserve" text-anchor="start" x="1371" y="-3472" font-family="Arial" font-size="15.00" fill="#bfdbfe">parameters saved next to the grid. Threads</text>
<text xml:space="preserve" text-anchor="start" x="1429" y="-3454" font-family="Arial" font-size="15.00" fill="#bfdbfe">and batch size come from</text>
<text xml:space="preserve" text-anchor="start" x="1425.5" y="-3436" font-family="Arial" font-size="15.00" fill="#bfdbfe">n_contingency_worker and</text>
</g>
<!-- artificiallimits -->
<g id="node18" class="node">
<title>artificiallimits</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1688.06,-3254 1343.94,-3254 1343.94,-3074 1688.06,-3074 1688.06,-3254"/>
<text xml:space="preserve" text-anchor="start" x="1373" y="-3223" font-family="Arial" font-size="20.00" fill="#eff6ff">attach_artificial_limits_to_results</text>
<text xml:space="preserve" text-anchor="start" x="1378" y="-3195" font-family="Arial" font-size="15.00" fill="#bfdbfe">Writes the artificial limits derived from the</text>
<text xml:space="preserve" text-anchor="start" x="1451.5" y="-3177" font-family="Arial" font-size="15.00" fill="#bfdbfe">worst currents onto</text>
<text xml:space="preserve" text-anchor="start" x="1364" y="-3159" font-family="Arial" font-size="15.00" fill="#bfdbfe">the network and attaches them to the AC N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="1437" y="-3141" font-family="Arial" font-size="15.00" fill="#bfdbfe">results. Currents do not</text>
<text xml:space="preserve" text-anchor="start" x="1374" y="-3123" font-family="Arial" font-size="15.00" fill="#bfdbfe">depend on limits, so no second analysis is</text>
</g>
<!-- topologymodel -->
<g id="node19" class="node">
<title>topologymodel</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2146.56,-3908.8 1789.44,-3908.8 1789.44,-3728.8 2146.56,-3728.8 2146.56,-3908.8"/>
<text xml:space="preserve" text-anchor="start" x="1810.5" y="-3877.8" font-family="Arial" font-size="20.00" fill="#eff6ff">get_master_asset_topology_artifact</text>
<text xml:space="preserve" text-anchor="start" x="1839.5" y="-3849.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">Extraction. Dispatches on the importer</text>
<text xml:space="preserve" text-anchor="start" x="1878" y="-3831.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">data_type and hands off to</text>
<text xml:space="preserve" text-anchor="start" x="1835.5" y="-3813.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">one of the readers below &#45;&#45; which is the</text>
<text xml:space="preserve" text-anchor="start" x="1887" y="-3795.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">whole reason the rest of</text>
<text xml:space="preserve" text-anchor="start" x="1809.5" y="-3777.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">the engine never has to know which framework</text>
</g>
<!-- materialize -->
<g id="node20" class="node">
<title>materialize</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1211.56,-3576.8 864.44,-3576.8 864.44,-3396.8 1211.56,-3396.8 1211.56,-3576.8"/>
<text xml:space="preserve" text-anchor="start" x="913.5" y="-3545.8" font-family="Arial" font-size="20.00" fill="#eef2ff">get_runtime_asset_topology</text>
<text xml:space="preserve" text-anchor="start" x="884.5" y="-3517.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">Transition 1. Reads the master data back and</text>
<text xml:space="preserve" text-anchor="start" x="965" y="-3499.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">materializes it against</text>
<text xml:space="preserve" text-anchor="start" x="910" y="-3481.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">the loaded network: structure from the</text>
<text xml:space="preserve" text-anchor="start" x="945" y="-3463.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">importer artifact, switch and</text>
<text xml:space="preserve" text-anchor="start" x="933.5" y="-3445.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">busbar states from the grid file.</text>
</g>
<!-- bridges -->
<g id="node21" class="node">
<title>bridges</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1176.06,-3254 825.94,-3254 825.94,-3074 1176.06,-3074 1176.06,-3254"/>
<text xml:space="preserve" text-anchor="start" x="875.5" y="-3214" font-family="Arial" font-size="20.00" fill="#eef2ff">compute_bridging_branches</text>
<text xml:space="preserve" text-anchor="start" x="846" y="-3186" font-family="Arial" font-size="15.00" fill="#c7d2fe">Tarjan bridge finding. A branch whose removal</text>
<text xml:space="preserve" text-anchor="start" x="948" y="-3168" font-family="Arial" font-size="15.00" fill="#c7d2fe">islands the grid,</text>
<text xml:space="preserve" text-anchor="start" x="864.5" y="-3150" font-family="Arial" font-size="15.00" fill="#c7d2fe">under N&#45;0 or any contingency, cannot be</text>
<text xml:space="preserve" text-anchor="start" x="954.5" y="-3132" font-family="Arial" font-size="15.00" fill="#c7d2fe">disconnected.</text>
</g>
<!-- relevantnodes -->
<g id="node22" class="node">
<title>relevantnodes</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1161.02,-2931.2 840.98,-2931.2 840.98,-2751.2 1161.02,-2751.2 1161.02,-2931.2"/>
<text xml:space="preserve" text-anchor="start" x="909" y="-2882.2" font-family="Arial" font-size="20.00" fill="#eef2ff">filter_relevant_nodes</text>
<text xml:space="preserve" text-anchor="start" x="883" y="-2854.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">Drop substations that are not worth</text>
<text xml:space="preserve" text-anchor="start" x="905.5" y="-2836.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">switching: too few branches,</text>
<text xml:space="preserve" text-anchor="start" x="888" y="-2818.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">no assets, or double connections.</text>
</g>
<!-- factors -->
<g id="node23" class="node">
<title>factors</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1181.06,-2598.3 820.94,-2598.3 820.94,-2418.3 1181.06,-2418.3 1181.06,-2598.3"/>
<text xml:space="preserve" text-anchor="start" x="899.5" y="-2558.3" font-family="Arial" font-size="20.00" fill="#eef2ff">compute PTDF / PSDF</text>
<text xml:space="preserve" text-anchor="start" x="841" y="-2530.3" font-family="Arial" font-size="15.00" fill="#c7d2fe">The reference PTDF matrix, solved once. Every</text>
<text xml:space="preserve" text-anchor="start" x="927" y="-2512.3" font-family="Arial" font-size="15.00" fill="#c7d2fe">topology the optimizer</text>
<text xml:space="preserve" text-anchor="start" x="867" y="-2494.3" font-family="Arial" font-size="15.00" fill="#c7d2fe">later evaluates is a low&#45;rank update of it</text>
<text xml:space="preserve" text-anchor="start" x="906" y="-2476.3" font-family="Arial" font-size="15.00" fill="#c7d2fe">rather than a refactorization.</text>
</g>
<!-- reduce -->
<g id="node24" class="node">
<title>reduce</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1162.56,-2265.4 839.44,-2265.4 839.44,-2085.4 1162.56,-2085.4 1162.56,-2265.4"/>
<text xml:space="preserve" text-anchor="start" x="859.5" y="-2234.4" font-family="Arial" font-size="20.00" fill="#eef2ff">reduce node / branch dimension</text>
<text xml:space="preserve" text-anchor="start" x="865" y="-2206.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">Collapse nodes that never change into a</text>
<text xml:space="preserve" text-anchor="start" x="920" y="-2188.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">single static column and</text>
<text xml:space="preserve" text-anchor="start" x="861.5" y="-2170.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">drop branches that are neither monitored,</text>
<text xml:space="preserve" text-anchor="start" x="927" y="-2152.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">outaged nor switched.</text>
<text xml:space="preserve" text-anchor="start" x="861" y="-2134.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">Directly shrinks the PTDF the GPU has to</text>
</g>
<!-- nminus2filter -->
<g id="node25" class="node">
<title>nminus2filter</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1201.06,-1942.6 800.94,-1942.6 800.94,-1762.6 1201.06,-1762.6 1201.06,-1942.6"/>
<text xml:space="preserve" text-anchor="start" x="821" y="-1884.6" font-family="Arial" font-size="20.00" fill="#eef2ff">filter_disconnectable_branches_nminus2</text>
<text xml:space="preserve" text-anchor="start" x="869" y="-1856.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">Exclude branches that island the grid in</text>
<text xml:space="preserve" text-anchor="start" x="894" y="-1838.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">combination with a contingency.</text>
</g>
<!-- simplify -->
<g id="node26" class="node">
<title>simplify</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="858.06,-1619.8 529.94,-1619.8 529.94,-1439.8 858.06,-1439.8 858.06,-1619.8"/>
<text xml:space="preserve" text-anchor="start" x="588.5" y="-1588.8" font-family="Arial" font-size="20.00" fill="#eef2ff">simplify_asset_topology</text>
<text xml:space="preserve" text-anchor="start" x="550.5" y="-1560.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">Transition 2. Projects each relevant station</text>
<text xml:space="preserve" text-anchor="start" x="632.5" y="-1542.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">onto one electrical</text>
<text xml:space="preserve" text-anchor="start" x="614.5" y="-1524.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">node at a time and runs</text>
<text xml:space="preserve" text-anchor="start" x="558.5" y="-1506.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">prepare_for_separation_set on the slice.</text>
<text xml:space="preserve" text-anchor="start" x="550" y="-1488.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">Stations that survive become the simplified</text>
</g>
<!-- electricalactions -->
<g id="node27" class="node">
<title>electricalactions</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1198.56,-1297 869.44,-1297 869.44,-1117 1198.56,-1117 1198.56,-1297"/>
<text xml:space="preserve" text-anchor="start" x="914" y="-1266" font-family="Arial" font-size="20.00" fill="#eef2ff">compute_electrical_actions</text>
<text xml:space="preserve" text-anchor="start" x="889.5" y="-1238" font-family="Arial" font-size="15.00" fill="#c7d2fe">Stage one of action set enumeration: every</text>
<text xml:space="preserve" text-anchor="start" x="973" y="-1220" font-family="Arial" font-size="15.00" fill="#c7d2fe">electrically distinct</text>
<text xml:space="preserve" text-anchor="start" x="909.5" y="-1202" font-family="Arial" font-size="15.00" fill="#c7d2fe">two&#45;node split of a station, filtered for</text>
<text xml:space="preserve" text-anchor="start" x="989.5" y="-1184" font-family="Arial" font-size="15.00" fill="#c7d2fe">islanding and</text>
<text xml:space="preserve" text-anchor="start" x="899" y="-1166" font-family="Arial" font-size="15.00" fill="#c7d2fe">connectivity, clipped if a station exceeds</text>
</g>
<!-- stationrealisations -->
<g id="node28" class="node">
<title>stationrealisations</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1202.56,-974.2 865.44,-974.2 865.44,-794.2 1202.56,-794.2 1202.56,-974.2"/>
<text xml:space="preserve" text-anchor="start" x="896" y="-943.2" font-family="Arial" font-size="20.00" fill="#eef2ff">enumerate_station_realisations</text>
<text xml:space="preserve" text-anchor="start" x="891.5" y="-915.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">Stage two: map each electrical split onto a</text>
<text xml:space="preserve" text-anchor="start" x="954" y="-897.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">reachable node&#45;breaker</text>
<text xml:space="preserve" text-anchor="start" x="885.5" y="-879.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">realization and precompute its reassignment</text>
<text xml:space="preserve" text-anchor="start" x="967.5" y="-861.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">distance. Splits with</text>
<text xml:space="preserve" text-anchor="start" x="921" y="-843.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">no valid realization are discarded.</text>
</g>
<!-- bboutage -->
<g id="node29" class="node">
<title>bboutage</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1212.06,-651.4 855.94,-651.4 855.94,-471.4 1212.06,-471.4 1212.06,-651.4"/>
<text xml:space="preserve" text-anchor="start" x="933" y="-593.4" font-family="Arial" font-size="20.00" fill="#eef2ff">preprocess_bb_outage</text>
<text xml:space="preserve" text-anchor="start" x="876" y="-565.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">Optional busbar outage contingencies, used by</text>
<text xml:space="preserve" text-anchor="start" x="918.5" y="-547.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">the do&#45;not&#45;make&#45;it&#45;worse criterion.</text>
</g>
<!-- initialloadflow -->
<g id="node30" class="node">
<title>initialloadflow</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1696.56,-2931.2 1335.44,-2931.2 1335.44,-2751.2 1696.56,-2751.2 1696.56,-2931.2"/>
<text xml:space="preserve" text-anchor="start" x="1415" y="-2910" font-family="Arial" font-size="20.00" fill="#fbd3cb">store_initial_ca_results</text>
<text xml:space="preserve" text-anchor="start" x="1493.5" y="-2882" font-family="Arial" font-size="13.00" fill="#f5b2a3">Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1392.5" y="-2862.4" font-family="Arial" font-size="15.00" fill="#f5b2a3">Stores the AC N&#45;1 of the import, with</text>
<text xml:space="preserve" text-anchor="start" x="1412" y="-2844.4" font-family="Arial" font-size="15.00" fill="#f5b2a3">permanent and artificial branch</text>
<text xml:space="preserve" text-anchor="start" x="1385" y="-2826.4" font-family="Arial" font-size="15.00" fill="#f5b2a3">limits, as the reference every proposed</text>
<text xml:space="preserve" text-anchor="start" x="1427" y="-2808.4" font-family="Arial" font-size="15.00" fill="#f5b2a3">topology is later compared</text>
<text xml:space="preserve" text-anchor="start" x="1355.5" y="-2790.4" font-family="Arial" font-size="15.00" fill="#f5b2a3">against. Runs before the DC preprocessing and</text>
</g>
<!-- loadflowstore -->
<g id="node31" class="node">
<title>loadflowstore</title>
<path fill="#64748b" stroke="#475569" stroke-width="2" d="M2646.06,-2590.2C2646.06,-2600.24 2564.9,-2608.4 2465,-2608.4 2365.1,-2608.4 2283.94,-2600.24 2283.94,-2590.2 2283.94,-2590.2 2283.94,-2426.4 2283.94,-2426.4 2283.94,-2416.36 2365.1,-2408.2 2465,-2408.2 2564.9,-2408.2 2646.06,-2416.36 2646.06,-2426.4 2646.06,-2426.4 2646.06,-2590.2 2646.06,-2590.2"/>
<path fill="none" stroke="#475569" stroke-width="2" d="M2646.06,-2590.2C2646.06,-2580.16 2564.9,-2572 2465,-2572 2365.1,-2572 2283.94,-2580.16 2283.94,-2590.2"/>
<text xml:space="preserve" text-anchor="start" x="2374" y="-2577.1" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="2399" y="-2549.1" font-family="Arial" font-size="13.00" fill="#cbd5e1">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="2359" y="-2529.5" font-family="Arial" font-size="15.00" fill="#cbd5e1">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="2304" y="-2511.5" font-family="Arial" font-size="15.00" fill="#cbd5e1">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="2320.5" y="-2493.5" font-family="Arial" font-size="15.00" fill="#cbd5e1">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="2314.5" y="-2475.5" font-family="Arial" font-size="15.00" fill="#cbd5e1">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="2379.5" y="-2457.5" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology it evaluates gets</text>
</g>
<!-- inputnminus1&#45;&gt;inputvalidation -->
<g id="edge2" class="edge">
<title>inputnminus1&#45;&gt;inputvalidation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M409.97,-5725.44C652.29,-5665.22 1075.28,-5560.11 1322,-5498.81"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1322.57,-5501.37 1329.22,-5497.01 1321.31,-5496.27 1322.57,-5501.37"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="909.21,-5599.6 909.21,-5622.4 1095.21,-5622.4 1095.21,-5599.6 909.21,-5599.6"/>
<text xml:space="preserve" text-anchor="start" x="912.21" y="-5619.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: input N&#45;1 definition</text>
</g>
<!-- assettopomaster&#45;&gt;materialize -->
<g id="edge1" class="edge">
<title>assettopomaster&#45;&gt;materialize</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2599.43,-259.85C2735.11,-360.5 2935,-516.27 2935,-560.4 2935,-1692.2 2935,-1692.2 2935,-1692.2 2935,-2057.13 3034,-2142.37 3034,-2507.3 3034,-3165 3034,-3165 3034,-3165 3034,-3363.24 1458.55,-3353.96 1265,-3396.8 1250.51,-3400.01 1235.72,-3404.01 1221,-3408.52"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1220.6,-3405.9 1214.23,-3410.65 1222.17,-3410.91 1220.6,-3405.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2953.07,-1841.2 2953.07,-1864 3075.07,-1864 3075.07,-1841.2 2953.07,-1841.2"/>
<text xml:space="preserve" text-anchor="start" x="2956.07" y="-1861" font-family="Arial" font-size="14.00" fill="#c9c9c9">canonical structure</text>
</g>
<!-- gridsnapshot&#45;&gt;topologymodel -->
<g id="edge19" class="edge">
<title>gridsnapshot&#45;&gt;topologymodel</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3063.1,-259.91C3102.64,-302.12 3155.3,-348.82 3213,-374.4 3230.55,-382.18 3279.94,-380.11 3299,-382.4 3326.21,-385.67 3333.3,-385.25 3360,-391.4 3602.58,-447.3 3894,-311.46 3894,-560.4 3894,-3487.8 3894,-3487.8 3894,-3487.8 3894,-3589.23 3844.03,-3620.21 3755,-3668.8 3617.22,-3743.99 2587.48,-3793.18 2156.65,-3810.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2156.69,-3808.05 2149.3,-3810.97 2156.9,-3813.29 2156.69,-3808.05"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3894,-2002.6 3894,-2025.4 4020,-2025.4 4020,-2002.6 3894,-2002.6"/>
<text xml:space="preserve" text-anchor="start" x="3897" y="-2022.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">normalized network</text>
</g>
<!-- gridsnapshot&#45;&gt;materialize -->
<g id="edge20" class="edge">
<title>gridsnapshot&#45;&gt;materialize</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3106.91,-259.65C3186.19,-331.13 3277,-439.4 3277,-560.4 3277,-3165 3277,-3165 3277,-3165 3277,-3390.03 1485.02,-3349.54 1265,-3396.8 1250.49,-3399.92 1235.69,-3403.85 1220.96,-3408.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1220.56,-3405.69 1214.18,-3410.43 1222.12,-3410.7 1220.56,-3405.69"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3277,-1841.2 3277,-1864 3506,-1864 3506,-1841.2 3277,-1841.2"/>
<text xml:space="preserve" text-anchor="start" x="3280" y="-1861" font-family="Arial" font-size="14.00" fill="#c9c9c9">live switch, coupler and busbar state</text>
</g>
<!-- loadgrid&#45;&gt;inputvalidation -->
<g id="edge3" class="edge">
<title>loadgrid&#45;&gt;inputvalidation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1516,-5682.47C1516,-5641.27 1516,-5592.16 1516,-5549.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1518.63,-5549.96 1516,-5542.46 1513.38,-5549.96 1518.63,-5549.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1516,-5599.6 1516,-5622.4 1715,-5622.4 1715,-5599.6 1516,-5599.6"/>
<text xml:space="preserve" text-anchor="start" x="1519" y="-5619.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: unconverted network</text>
</g>
<!-- loadgrid&#45;&gt;trafoconversion -->
<g id="edge4" class="edge">
<title>loadgrid&#45;&gt;trafoconversion</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1652.36,-5682.4C1682.02,-5662.8 1713.15,-5642.02 1742,-5622.4 1778.19,-5597.79 1817.09,-5570.77 1853.01,-5545.59"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1854.47,-5547.77 1859.1,-5541.32 1851.46,-5543.47 1854.47,-5547.77"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1768.79,-5599.6 1768.79,-5622.4 1868.79,-5622.4 1868.79,-5599.6 1768.79,-5599.6"/>
<text xml:space="preserve" text-anchor="start" x="1771.79" y="-5619.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">parsed network</text>
</g>
<!-- inputvalidation&#45;&gt;networkreduction -->
<g id="edge5" class="edge">
<title>inputvalidation&#45;&gt;networkreduction</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1595.6,-5359.63C1620.33,-5332.65 1647.95,-5303.18 1674,-5276.8 1691.31,-5259.27 1710.07,-5240.99 1728.42,-5223.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1729.89,-5225.71 1733.51,-5218.63 1726.27,-5221.9 1729.89,-5225.71"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1674,-5276.8 1674,-5299.6 1878,-5299.6 1878,-5276.8 1674,-5276.8"/>
<text xml:space="preserve" text-anchor="start" x="1677" y="-5296.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: voltage levels to keep</text>
</g>
<!-- inputvalidation&#45;&gt;nminus1definition -->
<g id="edge6" class="edge">
<title>inputvalidation&#45;&gt;nminus1definition</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1516,-5359.75C1516,-5295.35 1516,-5206.19 1516,-5127.8 1516,-5127.8 1516,-5127.8 1516,-4140.6 1516,-4066.34 1516,-3982.42 1516,-3919.04"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1518.63,-3919.18 1516,-3911.68 1513.38,-3919.18 1518.63,-3919.18"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1516,-4614.4 1516,-4654 1704,-4654 1704,-4614.4 1516,-4614.4"/>
<text xml:space="preserve" text-anchor="start" x="1519" y="-4651" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: grid&#45;validated input</text>
<text xml:space="preserve" text-anchor="start" x="1519" y="-4634.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">definition</text>
</g>
<!-- trafoconversion&#45;&gt;networkreduction -->
<g id="edge7" class="edge">
<title>trafoconversion&#45;&gt;networkreduction</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1944.03,-5359.67C1923.96,-5318.12 1900,-5268.53 1879.4,-5225.9"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1881.86,-5224.97 1876.24,-5219.36 1877.14,-5227.25 1881.86,-5224.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1914.15,-5276.8 1914.15,-5299.6 2033.15,-5299.6 2033.15,-5276.8 1914.15,-5276.8"/>
<text xml:space="preserve" text-anchor="start" x="1917.15" y="-5296.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">converted network</text>
</g>
<!-- networkreduction&#45;&gt;whitelists -->
<g id="edge8" class="edge">
<title>networkreduction&#45;&gt;whitelists</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1832,-5036.87C1832,-4995.67 1832,-4946.56 1832,-4904.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1834.63,-4904.36 1832,-4896.86 1829.38,-4904.36 1834.63,-4904.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1832,-4954 1832,-4976.8 1940,-4976.8 1940,-4954 1832,-4954"/>
<text xml:space="preserve" text-anchor="start" x="1835" y="-4973.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">reduced network</text>
</g>
<!-- whitelists&#45;&gt;convergingparams -->
<g id="edge9" class="edge">
<title>whitelists&#45;&gt;convergingparams</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1832,-4714.3C1832,-4668.34 1832,-4612.07 1832,-4564.67"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1834.63,-4564.91 1832,-4557.41 1829.38,-4564.91 1834.63,-4564.91"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1832,-4622.8 1832,-4645.6 1934,-4645.6 1934,-4622.8 1832,-4622.8"/>
<text xml:space="preserve" text-anchor="start" x="1835" y="-4642.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">scoped network</text>
</g>
<!-- convergingparams&#45;&gt;networkmasks -->
<g id="edge10" class="edge">
<title>convergingparams&#45;&gt;networkmasks</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1832,-4374.47C1832,-4333.27 1832,-4284.16 1832,-4241.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1834.63,-4241.96 1832,-4234.46 1829.38,-4241.96 1834.63,-4241.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1832,-4291.6 1832,-4314.4 1979,-4314.4 1979,-4291.6 1832,-4291.6"/>
<text xml:space="preserve" text-anchor="start" x="1835" y="-4311.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">converging parameters</text>
</g>
<!-- networkmasks&#45;&gt;nminus1definition -->
<g id="edge11" class="edge">
<title>networkmasks&#45;&gt;nminus1definition</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1744.4,-4051.67C1702.87,-4009.51 1653.19,-3959.07 1610.78,-3916.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1612.76,-3914.29 1605.62,-3910.79 1609.02,-3917.97 1612.76,-3914.29"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1683.47,-3968.8 1683.47,-3991.6 1795.47,-3991.6 1795.47,-3968.8 1683.47,-3968.8"/>
<text xml:space="preserve" text-anchor="start" x="1686.47" y="-3988.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey B: masks</text>
</g>
<!-- networkmasks&#45;&gt;topologymodel -->
<g id="edge12" class="edge">
<title>networkmasks&#45;&gt;topologymodel</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1869.7,-4051.67C1887.28,-4010.21 1908.25,-3960.74 1926.3,-3918.16"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1928.65,-3919.35 1929.16,-3911.42 1923.82,-3917.3 1928.65,-3919.35"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1904.08,-3968.8 1904.08,-3991.6 1950.08,-3991.6 1950.08,-3968.8 1904.08,-3968.8"/>
<text xml:space="preserve" text-anchor="start" x="1907.08" y="-3988.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">masks</text>
</g>
<!-- nminus1definition&#45;&gt;nminus1 -->
<g id="edge14" class="edge">
<title>nminus1definition&#45;&gt;nminus1</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1679.29,-3745.78C1697.52,-3739.35 1716,-3733.5 1734,-3728.8 1943.94,-3673.93 3672,-3704.79 3672,-3487.8 3672,-3487.8 3672,-3487.8 3672,-560.4 3672,-447.66 3597.92,-340.48 3531.74,-267.21"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3533.99,-265.78 3526.99,-262.02 3530.11,-269.32 3533.99,-265.78"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3672,-2002.6 3672,-2025.4 3697,-2025.4 3697,-2002.6 3672,-2002.6"/>
<text xml:space="preserve" text-anchor="start" x="3675" y="-2010.1" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- nminus1definition&#45;&gt;securityanalysis -->
<g id="edge13" class="edge">
<title>nminus1definition&#45;&gt;securityanalysis</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1516,-3728.93C1516,-3685.1 1516,-3632.08 1516,-3586.94"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1518.63,-3587.07 1516,-3579.57 1513.38,-3587.07 1518.63,-3587.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1516,-3646 1516,-3668.8 1603,-3668.8 1603,-3646 1516,-3646"/>
<text xml:space="preserve" text-anchor="start" x="1519" y="-3665.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">N&#45;1 definition</text>
</g>
<!-- securityanalysis&#45;&gt;artificiallimits -->
<g id="edge15" class="edge">
<title>securityanalysis&#45;&gt;artificiallimits</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1516,-3396.87C1516,-3355.67 1516,-3306.56 1516,-3264.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1518.63,-3264.36 1516,-3256.86 1513.38,-3264.36 1518.63,-3264.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1516,-3314 1516,-3336.8 1612,-3336.8 1612,-3314 1516,-3314"/>
<text xml:space="preserve" text-anchor="start" x="1519" y="-3333.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC N&#45;1 results</text>
</g>
<!-- artificiallimits&#45;&gt;gridsnapshot -->
<g id="edge17" class="edge">
<title>artificiallimits&#45;&gt;gridsnapshot</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1687.93,-3156.14C1847.03,-3143.71 2086.15,-3109.19 2271,-3014 2504.57,-2893.73 2596.93,-2849.62 2701,-2608.4 2736.25,-2526.7 2703.61,-2497.14 2701,-2408.2 2691.66,-2089.57 2659,-2010.97 2659,-1692.2 2659,-1692.2 2659,-1692.2 2659,-560.4 2659,-489.94 2786.46,-357.3 2883.42,-266.59"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2885.03,-268.68 2888.72,-261.65 2881.45,-264.84 2885.03,-268.68"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2659.01,-1679.8 2659.01,-1702.6 2747.01,-1702.6 2747.01,-1679.8 2659.01,-1679.8"/>
<text xml:space="preserve" text-anchor="start" x="2662.01" y="-1699.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">artificial limits</text>
</g>
<!-- artificiallimits&#45;&gt;initialloadflow -->
<g id="edge16" class="edge">
<title>artificiallimits&#45;&gt;initialloadflow</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1516,-3074.07C1516,-3032.87 1516,-2983.76 1516,-2941.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1518.63,-2941.56 1516,-2934.06 1513.38,-2941.56 1518.63,-2941.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1516,-2991.2 1516,-3014 1727,-3014 1727,-2991.2 1516,-2991.2"/>
<text xml:space="preserve" text-anchor="start" x="1519" y="-3011" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC N&#45;1 results incl. artificial limits</text>
</g>
<!-- topologymodel&#45;&gt;masks -->
<g id="edge21" class="edge">
<title>topologymodel&#45;&gt;masks</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2204,-3807.95C2638.73,-3788.34 3549.78,-3740.12 3858,-3668.8 3994.47,-3637.23 4116,-3627.87 4116,-3487.8 4116,-3487.8 4116,-3487.8 4116,-560.4 4116,-448.5 4044.13,-341.03 3980.05,-267.45"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3982.4,-266.15 3975.48,-262.26 3978.46,-269.62 3982.4,-266.15"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3120.98,-3746.24 3120.98,-3769.04 3227.98,-3769.04 3227.98,-3746.24 3120.98,-3746.24"/>
<text xml:space="preserve" text-anchor="start" x="3123.98" y="-3766.04" font-family="Arial" font-size="14.00" fill="#c9c9c9">per&#45;asset masks</text>
</g>
<!-- topologymodel&#45;&gt;materialize -->
<g id="edge22" class="edge">
<title>topologymodel&#45;&gt;materialize</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1292,-3648.53C1283.27,-3645.29 1274.28,-3641.79 1265,-3638 1263.78,-3637.5 1262.56,-3637 1261.34,-3636.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1262.7,-3634.21 1254.78,-3633.64 1260.62,-3639.03 1262.7,-3634.21"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1189.33,-3640.96 1189.33,-3663.76 1272.33,-3663.76 1272.33,-3640.96 1189.33,-3640.96"/>
<text xml:space="preserve" text-anchor="start" x="1192.33" y="-3660.76" font-family="Arial" font-size="14.00" fill="#c9c9c9">ImportResult</text>
</g>
<!-- materialize&#45;&gt;bridges -->
<g id="edge23" class="edge">
<title>materialize&#45;&gt;bridges</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1027.74,-3396.87C1022.99,-3355.67 1017.33,-3306.56 1012.44,-3264.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1015.06,-3263.99 1011.59,-3256.84 1009.84,-3264.59 1015.06,-3263.99"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1020.61,-3314 1020.61,-3336.8 1231.61,-3336.8 1231.61,-3314 1020.61,-3314"/>
<text xml:space="preserve" text-anchor="start" x="1023.61" y="-3333.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">runtime topology on NetworkData</text>
</g>
<!-- bridges&#45;&gt;relevantnodes -->
<g id="edge24" class="edge">
<title>bridges&#45;&gt;relevantnodes</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M953.72,-3074.07C945.56,-3054.77 938.31,-3034.11 934,-3014 928.78,-2989.65 932.91,-2964.24 941.2,-2940.66"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="943.58,-2941.8 943.78,-2933.85 938.67,-2939.94 943.58,-2941.8"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="934,-2991.2 934,-3014 1011,-3014 1011,-2991.2 934,-2991.2"/>
<text xml:space="preserve" text-anchor="start" x="937" y="-3011" font-family="Arial" font-size="14.00" fill="#c9c9c9">bridge flags</text>
</g>
<!-- relevantnodes&#45;&gt;factors -->
<g id="edge25" class="edge">
<title>relevantnodes&#45;&gt;factors</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1001,-2751.52C1001,-2707.48 1001,-2654.11 1001,-2608.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1003.63,-2608.75 1001,-2601.25 998.38,-2608.75 1003.63,-2608.75"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1001,-2668.4 1001,-2691.2 1116,-2691.2 1116,-2668.4 1001,-2668.4"/>
<text xml:space="preserve" text-anchor="start" x="1004" y="-2688.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">switchable subset</text>
</g>
<!-- factors&#45;&gt;reduce -->
<g id="edge26" class="edge">
<title>factors&#45;&gt;reduce</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1001,-2418.62C1001,-2374.58 1001,-2321.21 1001,-2275.78"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1003.63,-2275.85 1001,-2268.35 998.38,-2275.85 1003.63,-2275.85"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1001,-2325.4 1001,-2348.2 1091,-2348.2 1091,-2325.4 1001,-2325.4"/>
<text xml:space="preserve" text-anchor="start" x="1004" y="-2345.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">PTDF / PSDF</text>
</g>
<!-- reduce&#45;&gt;nminus2filter -->
<g id="edge27" class="edge">
<title>reduce&#45;&gt;nminus2filter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1001,-2085.47C1001,-2044.27 1001,-1995.16 1001,-1952.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1003.63,-1952.96 1001,-1945.46 998.38,-1952.96 1003.63,-1952.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1001,-2002.6 1001,-2025.4 1130,-2025.4 1130,-2002.6 1001,-2002.6"/>
<text xml:space="preserve" text-anchor="start" x="1004" y="-2022.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">reduced dimensions</text>
</g>
<!-- nminus2filter&#45;&gt;simplify -->
<g id="edge28" class="edge">
<title>nminus2filter&#45;&gt;simplify</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M898.74,-1762.73C877.81,-1743.43 856.29,-1722.74 837,-1702.6 814.6,-1679.21 791.6,-1652.8 770.78,-1627.84"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="772.85,-1626.22 766.04,-1622.12 768.81,-1629.57 772.85,-1626.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="837,-1679.8 837,-1702.6 1050,-1702.6 1050,-1679.8 837,-1679.8"/>
<text xml:space="preserve" text-anchor="start" x="840" y="-1699.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">final branch and injection ordering</text>
</g>
<!-- nminus2filter&#45;&gt;electricalactions -->
<g id="edge29" class="edge">
<title>nminus2filter&#45;&gt;electricalactions</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1051.78,-1762.98C1060.56,-1743.66 1068.35,-1722.9 1073,-1702.6 1107.28,-1552.87 1092.61,-1509.34 1073,-1357 1070.87,-1340.48 1067.47,-1323.19 1063.55,-1306.46"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1066.21,-1306.31 1061.89,-1299.64 1061.11,-1307.54 1066.21,-1306.31"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1093.57,-1518.4 1093.57,-1541.2 1227.57,-1541.2 1227.57,-1518.4 1093.57,-1518.4"/>
<text xml:space="preserve" text-anchor="start" x="1096.57" y="-1538.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">disconnectable set D</text>
</g>
<!-- simplify&#45;&gt;electricalactions -->
<g id="edge30" class="edge">
<title>simplify&#45;&gt;electricalactions</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M765.15,-1440.18C788.85,-1412.5 816.08,-1382.55 843,-1357 862.13,-1338.83 883.36,-1320.62 904.51,-1303.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="906.05,-1305.61 910.24,-1298.86 902.75,-1301.52 906.05,-1305.61"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="843,-1357 843,-1379.8 1050,-1379.8 1050,-1357 843,-1357"/>
<text xml:space="preserve" text-anchor="start" x="846" y="-1376.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">reduced stations to enumerate in</text>
</g>
<!-- electricalactions&#45;&gt;stationrealisations -->
<g id="edge31" class="edge">
<title>electricalactions&#45;&gt;stationrealisations</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1034,-1117.07C1034,-1075.87 1034,-1026.76 1034,-984.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1036.63,-984.56 1034,-977.06 1031.38,-984.56 1036.63,-984.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1034,-1034.2 1034,-1057 1130,-1057 1130,-1034.2 1034,-1034.2"/>
<text xml:space="preserve" text-anchor="start" x="1037" y="-1054" font-family="Arial" font-size="14.00" fill="#c9c9c9">electrical splits</text>
</g>
<!-- stationrealisations&#45;&gt;bboutage -->
<g id="edge32" class="edge">
<title>stationrealisations&#45;&gt;bboutage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1034,-794.27C1034,-753.07 1034,-703.96 1034,-661.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1036.63,-661.76 1034,-654.26 1031.38,-661.76 1036.63,-661.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1034,-711.4 1034,-734.2 1111,-734.2 1111,-711.4 1034,-711.4"/>
<text xml:space="preserve" text-anchor="start" x="1037" y="-731.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">action set A</text>
</g>
<!-- bboutage&#45;&gt;branchactionset -->
<g id="edge33" class="edge">
<title>bboutage&#45;&gt;branchactionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1034,-431.4C1034,-378.84 1034,-319.03 1034,-270.01"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1036.63,-270.33 1034,-262.83 1031.38,-270.33 1036.63,-270.33"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="822,-345.49 822,-368.29 1034,-368.29 1034,-345.49 822,-345.49"/>
<text xml:space="preserve" text-anchor="start" x="825" y="-365.29" font-family="Arial" font-size="14.00" fill="#c9c9c9">padded action arrays for the GPU</text>
</g>
<!-- bboutage&#45;&gt;actionset -->
<g id="edge34" class="edge">
<title>bboutage&#45;&gt;actionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1196.85,-431.4C1220.47,-412.4 1244.43,-392.97 1267,-374.4 1309.31,-339.59 1355.31,-300.83 1395.62,-266.56"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1397.28,-268.59 1401.3,-261.73 1393.88,-264.6 1397.28,-268.59"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1049.81,-346.41 1049.81,-369.21 1300.81,-369.21 1300.81,-346.41 1049.81,-346.41"/>
<text xml:space="preserve" text-anchor="start" x="1052.81" y="-366.21" font-family="Arial" font-size="14.00" fill="#c9c9c9">the same actions as physical switchings</text>
</g>
<!-- bboutage&#45;&gt;dcnminus1 -->
<g id="edge35" class="edge">
<title>bboutage&#45;&gt;dcnminus1</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1252,-532.76C1399.79,-507.27 1595.73,-459.92 1751,-374.4 1800.82,-346.96 1847.94,-305.48 1885.46,-267.12"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1886.92,-269.39 1890.25,-262.17 1883.14,-265.74 1886.92,-269.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1412.18,-446.1 1412.18,-468.9 1590.18,-468.9 1590.18,-446.1 1412.18,-446.1"/>
<text xml:space="preserve" text-anchor="start" x="1415.18" y="-465.9" font-family="Arial" font-size="14.00" fill="#c9c9c9">contingencies DC computes</text>
</g>
<!-- initialloadflow&#45;&gt;loadflowstore -->
<g id="edge18" class="edge">
<title>initialloadflow&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1696.45,-2777.28C1861.54,-2719.72 2105.01,-2634.82 2273.49,-2576.08"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2274.14,-2578.63 2280.36,-2573.68 2272.41,-2573.67 2274.14,-2578.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1990.5,-2668.4 1990.5,-2691.2 2121.5,-2691.2 2121.5,-2668.4 1990.5,-2668.4"/>
<text xml:space="preserve" text-anchor="start" x="1993.5" y="-2688.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial AC N&#45;1 results</text>
</g>
</g>
</svg>`;case`nminus1Journeys`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1821pt" height="3875pt"
 viewBox="0.00 0.00 1821.00 3875.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3859.65)">
<g id="clust1" class="cluster">
<title>cluster_unprocessedgridstore</title>
<polygon fill="#3e4651" stroke="#2d333d" points="8,-3563.4 8,-3828.6 442,-3828.6 442,-3563.4 8,-3563.4"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-3815.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">UNPROCESSED GRID STORE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_importstage</title>
<polygon fill="#3a404a" stroke="#292f37" points="477,-662.8 477,-3836.6 1389,-3836.6 1389,-662.8 477,-662.8"/>
<text xml:space="preserve" text-anchor="start" x="485" y="-3823.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">IMPORTER PREPROCESSING</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_nminus1definition</title>
<polygon fill="#3e4651" stroke="#2d333d" points="685,-963.8 685,-1890.6 1331,-1890.6 1331,-963.8 685,-963.8"/>
<text xml:space="preserve" text-anchor="start" x="693" y="-1877.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">CREATE_NMINUS1_DEFINITION</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_processedgrid</title>
<polygon fill="#3e4651" stroke="#2d333d" points="537,-8 537,-289.2 1375,-289.2 1375,-8 537,-8"/>
<text xml:space="preserve" text-anchor="start" x="545" y="-275.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">PROCESSED GRID FOLDER</text>
</g>
<!-- inputnminus1 -->
<g id="node1" class="node">
<title>inputnminus1</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="410.06,-3775.4 39.94,-3775.4 39.94,-3595.4 410.06,-3595.4 410.06,-3775.4"/>
<text xml:space="preserve" text-anchor="start" x="98.5" y="-3744.4" font-family="Arial" font-size="20.00" fill="#f0f9ff">Input N&#45;1 definition (optional)</text>
<text xml:space="preserve" text-anchor="start" x="60" y="-3716.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">A JSON dump of Nminus1Definition with element</text>
<text xml:space="preserve" text-anchor="start" x="147.5" y="-3698.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">ids already resolved by</text>
<text xml:space="preserve" text-anchor="start" x="150" y="-3680.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">its producer, stored as</text>
<text xml:space="preserve" text-anchor="start" x="87.5" y="-3662.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">input_nminus1_definition.json in the data</text>
<text xml:space="preserve" text-anchor="start" x="160.5" y="-3644.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">folder or passed as</text>
</g>
<!-- maskdefinition -->
<g id="node2" class="node">
<title>maskdefinition</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1203.56,-1829.4 812.44,-1829.4 812.44,-1649.4 1203.56,-1649.4 1203.56,-1829.4"/>
<text xml:space="preserve" text-anchor="start" x="832.5" y="-1789.4" font-family="Arial" font-size="20.00" fill="#ffe0c2">create_nminus1_definition_from_masks</text>
<text xml:space="preserve" text-anchor="start" x="861" y="-1761.4" font-family="Arial" font-size="15.00" fill="#f9b27c">Journey B source. Area settings are applied</text>
<text xml:space="preserve" text-anchor="start" x="943.5" y="-1743.4" font-family="Arial" font-size="15.00" fill="#f9b27c">through the masks.</text>
<text xml:space="preserve" text-anchor="start" x="866.5" y="-1725.4" font-family="Arial" font-size="15.00" fill="#f9b27c">Three&#45;winding transformers still carry their</text>
<text xml:space="preserve" text-anchor="start" x="956.5" y="-1707.4" font-family="Arial" font-size="15.00" fill="#f9b27c">original id here.</text>
</g>
<!-- converttrafo3w -->
<g id="node3" class="node">
<title>converttrafo3w</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1291.06,-1506.6 724.94,-1506.6 724.94,-1326.6 1291.06,-1326.6 1291.06,-1506.6"/>
<text xml:space="preserve" text-anchor="start" x="745" y="-1475.6" font-family="Arial" font-size="20.00" fill="#eff6ff">convert_three_winding_transformers_in_nminus1_definition</text>
<text xml:space="preserve" text-anchor="start" x="860" y="-1447.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">Both journeys. Replaces each three&#45;winding</text>
<text xml:space="preserve" text-anchor="start" x="961.5" y="-1429.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">transformer in</text>
<text xml:space="preserve" text-anchor="start" x="858.5" y="-1411.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">contingencies and monitored elements by its</text>
<text xml:space="preserve" text-anchor="start" x="934" y="-1393.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">three legs, and copies</text>
<text xml:space="preserve" text-anchor="start" x="862.5" y="-1375.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">SPPS conditions and actions on it once per</text>
</g>
<!-- finalcheck -->
<g id="node4" class="node">
<title>finalcheck</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="1189.06,-1183.8 826.94,-1183.8 826.94,-1003.8 1189.06,-1003.8 1189.06,-1183.8"/>
<text xml:space="preserve" text-anchor="start" x="847" y="-1152.8" font-family="Arial" font-size="20.00" fill="#f0f9ff">filter_nminus1_definition_to_network</text>
<text xml:space="preserve" text-anchor="start" x="862.5" y="-1124.8" font-family="Arial" font-size="15.00" fill="#b6ecf7">Journey A only. Checks the converted input</text>
<text xml:space="preserve" text-anchor="start" x="938.5" y="-1106.8" font-family="Arial" font-size="15.00" fill="#b6ecf7">definition against the</text>
<text xml:space="preserve" text-anchor="start" x="854.5" y="-1088.8" font-family="Arial" font-size="15.00" fill="#b6ecf7">processed grid again, dropping elements later</text>
<text xml:space="preserve" text-anchor="start" x="940" y="-1070.8" font-family="Arial" font-size="15.00" fill="#b6ecf7">preprocessing steps</text>
<text xml:space="preserve" text-anchor="start" x="976.5" y="-1052.8" font-family="Arial" font-size="15.00" fill="#b6ecf7">removed.</text>
</g>
<!-- loadgrid -->
<g id="node5" class="node">
<title>loadgrid</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="881.56,-3775.4 520.44,-3775.4 520.44,-3595.4 881.56,-3595.4 881.56,-3775.4"/>
<text xml:space="preserve" text-anchor="start" x="629.5" y="-3717.4" font-family="Arial" font-size="20.00" fill="#f8fafc">Load and merge</text>
<text xml:space="preserve" text-anchor="start" x="540.5" y="-3689.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Parse UCTE/CGMES/PowerFactory. Dominates</text>
<text xml:space="preserve" text-anchor="start" x="603.5" y="-3671.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">importer runtime on CGMES.</text>
</g>
<!-- inputvalidation -->
<g id="node6" class="node">
<title>inputvalidation</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="885.06,-3452.6 516.94,-3452.6 516.94,-3272.6 885.06,-3272.6 885.06,-3452.6"/>
<text xml:space="preserve" text-anchor="start" x="537" y="-3421.6" font-family="Arial" font-size="20.00" fill="#f0f9ff">load_nminus1_definition_for_network</text>
<text xml:space="preserve" text-anchor="start" x="557.5" y="-3393.6" font-family="Arial" font-size="15.00" fill="#b6ecf7">Journey A, stage 1. Validates the input N&#45;1</text>
<text xml:space="preserve" text-anchor="start" x="631.5" y="-3375.6" font-family="Arial" font-size="15.00" fill="#b6ecf7">definition against the</text>
<text xml:space="preserve" text-anchor="start" x="563" y="-3357.6" font-family="Arial" font-size="15.00" fill="#b6ecf7">grid before its three&#45;winding transformers</text>
<text xml:space="preserve" text-anchor="start" x="622" y="-3339.6" font-family="Arial" font-size="15.00" fill="#b6ecf7">are converted, because</text>
<text xml:space="preserve" text-anchor="start" x="561.5" y="-3321.6" font-family="Arial" font-size="15.00" fill="#b6ecf7">the input references them by their original</text>
</g>
<!-- trafoconversion -->
<g id="node7" class="node">
<title>trafoconversion</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1349.06,-3452.6 994.94,-3452.6 994.94,-3272.6 1349.06,-3272.6 1349.06,-3452.6"/>
<text xml:space="preserve" text-anchor="start" x="1061" y="-3412.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Convert 3W transformers</text>
<text xml:space="preserve" text-anchor="start" x="1015" y="-3384.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">Replaces every three&#45;winding transformer &lt;id&gt;</text>
<text xml:space="preserve" text-anchor="start" x="1102" y="-3366.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">in the network by the</text>
<text xml:space="preserve" text-anchor="start" x="1020" y="-3348.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">two&#45;winding legs &lt;id&gt;&#45;Leg1/2/3 around a star</text>
<text xml:space="preserve" text-anchor="start" x="1128" y="-3330.6" font-family="Arial" font-size="15.00" fill="#bfdbfe">voltage level.</text>
</g>
<!-- networkreduction -->
<g id="node8" class="node">
<title>networkreduction</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1221.56,-3129.8 812.44,-3129.8 812.44,-2949.8 1221.56,-2949.8 1221.56,-3129.8"/>
<text xml:space="preserve" text-anchor="start" x="832.5" y="-3098.8" font-family="Arial" font-size="20.00" fill="#f8fafc">reduce_network_based_on_area_settings</text>
<text xml:space="preserve" text-anchor="start" x="874.5" y="-3070.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Optional. Keeps the voltage levels within a</text>
<text xml:space="preserve" text-anchor="start" x="957.5" y="-3052.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">range of the view,</text>
<text xml:space="preserve" text-anchor="start" x="885" y="-3034.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">control and N&#45;1 area voltage levels and</text>
<text xml:space="preserve" text-anchor="start" x="950" y="-3016.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">removes the rest. In</text>
<text xml:space="preserve" text-anchor="start" x="873.5" y="-2998.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">journey A the voltage levels of all input N&#45;1</text>
</g>
<!-- whitelists -->
<g id="node9" class="node">
<title>whitelists</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1272.02,-2807 951.98,-2807 951.98,-2627 1272.02,-2627 1272.02,-2807"/>
<text xml:space="preserve" text-anchor="start" x="1044.5" y="-2749" font-family="Arial" font-size="20.00" fill="#f8fafc">Apply whitelists</text>
<text xml:space="preserve" text-anchor="start" x="978.5" y="-2721" font-family="Arial" font-size="15.00" fill="#cbd5e1">Apply the CB / black&#45; and whitelists that</text>
<text xml:space="preserve" text-anchor="start" x="1022" y="-2703" font-family="Arial" font-size="15.00" fill="#cbd5e1">scope the switchable area.</text>
</g>
<!-- convergingparams -->
<g id="node10" class="node">
<title>convergingparams</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1340.56,-2484.2 999.44,-2484.2 999.44,-2304.2 1340.56,-2304.2 1340.56,-2484.2"/>
<text xml:space="preserve" text-anchor="start" x="1019.5" y="-2444.2" font-family="Arial" font-size="20.00" fill="#f8fafc">find_converging_loadflow_params</text>
<text xml:space="preserve" text-anchor="start" x="1023.5" y="-2416.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Sweep loadflow parameters and voltage init</text>
<text xml:space="preserve" text-anchor="start" x="1094" y="-2398.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">methods until the base</text>
<text xml:space="preserve" text-anchor="start" x="1039" y="-2380.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">case converges. Some grid files do not</text>
<text xml:space="preserve" text-anchor="start" x="1097.5" y="-2362.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">converge on defaults.</text>
</g>
<!-- networkmasks -->
<g id="node11" class="node">
<title>networkmasks</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1330.02,-2161.4 1009.98,-2161.4 1009.98,-1981.4 1330.02,-1981.4 1330.02,-2161.4"/>
<text xml:space="preserve" text-anchor="start" x="1081.5" y="-2130.4" font-family="Arial" font-size="20.00" fill="#f8fafc">get_network_masks</text>
<text xml:space="preserve" text-anchor="start" x="1036" y="-2102.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Build the per&#45;asset masks from the area</text>
<text xml:space="preserve" text-anchor="start" x="1116" y="-2084.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">settings: what is</text>
<text xml:space="preserve" text-anchor="start" x="1050.5" y="-2066.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">monitored, outaged, switchable and</text>
<text xml:space="preserve" text-anchor="start" x="1055.5" y="-2048.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">disconnectable. Computed in both</text>
<text xml:space="preserve" text-anchor="start" x="1038" y="-2030.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">journeys; only journey B derives its N&#45;1</text>
</g>
<!-- securityanalysis -->
<g id="node12" class="node">
<title>securityanalysis</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="912.56,-882.8 565.44,-882.8 565.44,-702.8 912.56,-702.8 912.56,-882.8"/>
<text xml:space="preserve" text-anchor="start" x="585.5" y="-861.6" font-family="Arial" font-size="20.00" fill="#f8fafc">import_grid_with_security_analysis</text>
<text xml:space="preserve" text-anchor="start" x="706" y="-833.6" font-family="Arial" font-size="13.00" fill="#cbd5e1">PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="608.5" y="-814" font-family="Arial" font-size="15.00" fill="#cbd5e1">The only AC N&#45;1 of the import, with the</text>
<text xml:space="preserve" text-anchor="start" x="672" y="-796" font-family="Arial" font-size="15.00" fill="#cbd5e1">converging loadflow</text>
<text xml:space="preserve" text-anchor="start" x="594" y="-778" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters saved next to the grid. Threads</text>
<text xml:space="preserve" text-anchor="start" x="652" y="-760" font-family="Arial" font-size="15.00" fill="#cbd5e1">and batch size come from</text>
<text xml:space="preserve" text-anchor="start" x="648.5" y="-742" font-family="Arial" font-size="15.00" fill="#cbd5e1">n_contingency_worker and</text>
</g>
<!-- nminus1 -->
<g id="node13" class="node">
<title>nminus1</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1335.06,-228 1010.94,-228 1010.94,-48 1335.06,-48 1335.06,-228"/>
<text xml:space="preserve" text-anchor="start" x="1070" y="-197" font-family="Arial" font-size="20.00" fill="#f8fafc">nminus1_definition.json</text>
<text xml:space="preserve" text-anchor="start" x="1031" y="-169" font-family="Arial" font-size="15.00" fill="#cbd5e1">The grid&#45;validated contingency set, written</text>
<text xml:space="preserve" text-anchor="start" x="1092.5" y="-151" font-family="Arial" font-size="15.00" fill="#cbd5e1">by the importer from the</text>
<text xml:space="preserve" text-anchor="start" x="1049.5" y="-133" font-family="Arial" font-size="15.00" fill="#cbd5e1">input N&#45;1 definition (journey A) or the</text>
<text xml:space="preserve" text-anchor="start" x="1074.5" y="-115" font-family="Arial" font-size="15.00" fill="#cbd5e1">network masks (journey B). It</text>
<text xml:space="preserve" text-anchor="start" x="1055" y="-97" font-family="Arial" font-size="15.00" fill="#cbd5e1">is a separate file from the input N&#45;1</text>
</g>
<!-- dcnminus1 -->
<g id="node14" class="node">
<title>dcnminus1</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="901.06,-228 576.94,-228 576.94,-48 901.06,-48 901.06,-228"/>
<text xml:space="preserve" text-anchor="start" x="619.5" y="-188" font-family="Arial" font-size="20.00" fill="#f8fafc">dc_nminus1_definition.json</text>
<text xml:space="preserve" text-anchor="start" x="597" y="-160" font-family="Arial" font-size="15.00" fill="#cbd5e1">The contingencies DC computes, in solver</text>
<text xml:space="preserve" text-anchor="start" x="640.5" y="-142" font-family="Arial" font-size="15.00" fill="#cbd5e1">order. DC preprocessing also</text>
<text xml:space="preserve" text-anchor="start" x="612" y="-124" font-family="Arial" font-size="15.00" fill="#cbd5e1">writes them as the N&#45;1 definition if the</text>
<text xml:space="preserve" text-anchor="start" x="685" y="-106" font-family="Arial" font-size="15.00" fill="#cbd5e1">folder has none.</text>
</g>
<!-- dcpreprocess -->
<g id="node15" class="node">
<title>dcpreprocess</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="905.56,-560 572.44,-560 572.44,-380 905.56,-380 905.56,-560"/>
<text xml:space="preserve" text-anchor="start" x="609.5" y="-538.8" font-family="Arial" font-size="20.00" fill="#eef2ff">load_grid (DC preprocessing)</text>
<text xml:space="preserve" text-anchor="start" x="702.5" y="-510.8" font-family="Arial" font-size="13.00" fill="#c7d2fe">JAX, NumPy</text>
<text xml:space="preserve" text-anchor="start" x="606.5" y="-491.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">A chain of NetworkData transforms that</text>
<text xml:space="preserve" text-anchor="start" x="658" y="-473.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">shrinks the problem and</text>
<text xml:space="preserve" text-anchor="start" x="592.5" y="-455.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">enumerates the action space. This is where</text>
<text xml:space="preserve" text-anchor="start" x="681" y="-437.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">the linearized DC</text>
<text xml:space="preserve" text-anchor="start" x="665" y="-419.2" font-family="Arial" font-size="15.00" fill="#c7d2fe">representation is built.</text>
</g>
<!-- initialloadflow -->
<g id="node16" class="node">
<title>initialloadflow</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1790.56,-3775.4 1429.44,-3775.4 1429.44,-3595.4 1790.56,-3595.4 1790.56,-3775.4"/>
<text xml:space="preserve" text-anchor="start" x="1509" y="-3754.2" font-family="Arial" font-size="20.00" fill="#fbd3cb">store_initial_ca_results</text>
<text xml:space="preserve" text-anchor="start" x="1587.5" y="-3726.2" font-family="Arial" font-size="13.00" fill="#f5b2a3">Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1486.5" y="-3706.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">Stores the AC N&#45;1 of the import, with</text>
<text xml:space="preserve" text-anchor="start" x="1506" y="-3688.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">permanent and artificial branch</text>
<text xml:space="preserve" text-anchor="start" x="1479" y="-3670.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">limits, as the reference every proposed</text>
<text xml:space="preserve" text-anchor="start" x="1521" y="-3652.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">topology is later compared</text>
<text xml:space="preserve" text-anchor="start" x="1449.5" y="-3634.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">against. Runs before the DC preprocessing and</text>
</g>
<!-- inputnminus1&#45;&gt;inputvalidation -->
<g id="edge1" class="edge">
<title>inputnminus1&#45;&gt;inputvalidation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M356.96,-3595.47C420.29,-3552.79 496.2,-3501.62 560.59,-3458.23"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="561.93,-3460.49 566.69,-3454.12 559,-3456.14 561.93,-3460.49"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="477.27,-3512.6 477.27,-3535.4 663.27,-3535.4 663.27,-3512.6 477.27,-3512.6"/>
<text xml:space="preserve" text-anchor="start" x="480.27" y="-3532.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: input N&#45;1 definition</text>
</g>
<!-- maskdefinition&#45;&gt;converttrafo3w -->
<g id="edge12" class="edge">
<title>maskdefinition&#45;&gt;converttrafo3w</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1008,-1649.47C1008,-1608.27 1008,-1559.16 1008,-1516.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1010.63,-1516.96 1008,-1509.46 1005.38,-1516.96 1010.63,-1516.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1008,-1566.6 1008,-1589.4 1222,-1589.4 1222,-1566.6 1008,-1566.6"/>
<text xml:space="preserve" text-anchor="start" x="1011" y="-1586.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey B: mask&#45;derived definition</text>
</g>
<!-- converttrafo3w&#45;&gt;finalcheck -->
<g id="edge14" class="edge">
<title>converttrafo3w&#45;&gt;finalcheck</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1008,-1326.67C1008,-1285.47 1008,-1236.36 1008,-1193.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1010.63,-1194.16 1008,-1186.66 1005.38,-1194.16 1010.63,-1194.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1008,-1243.8 1008,-1266.6 1199,-1266.6 1199,-1243.8 1008,-1243.8"/>
<text xml:space="preserve" text-anchor="start" x="1011" y="-1263.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: converted definition</text>
</g>
<!-- converttrafo3w&#45;&gt;nminus1 -->
<g id="edge15" class="edge">
<title>converttrafo3w&#45;&gt;nminus1</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1192.93,-1326.7C1275.71,-1272.92 1355,-1195.2 1355,-1094.8 1355,-1094.8 1355,-1094.8 1355,-469 1355,-382.9 1306.91,-298.12 1260.17,-236.06"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1262.37,-234.61 1255.73,-230.24 1258.19,-237.79 1262.37,-234.61"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1355,-781.4 1355,-804.2 1547,-804.2 1547,-781.4 1355,-781.4"/>
<text xml:space="preserve" text-anchor="start" x="1358" y="-801.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey B: converted definition</text>
</g>
<!-- finalcheck&#45;&gt;securityanalysis -->
<g id="edge16" class="edge">
<title>finalcheck&#45;&gt;securityanalysis</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M891.94,-963.8C869.77,-939.15 846.91,-913.74 825.84,-890.33"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="827.9,-888.69 820.93,-884.87 823.99,-892.2 827.9,-888.69"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="768.35,-923.13 768.35,-945.93 855.35,-945.93 855.35,-923.13 768.35,-923.13"/>
<text xml:space="preserve" text-anchor="start" x="771.35" y="-942.93" font-family="Arial" font-size="14.00" fill="#c9c9c9">N&#45;1 definition</text>
</g>
<!-- finalcheck&#45;&gt;nminus1 -->
<g id="edge17" class="edge">
<title>finalcheck&#45;&gt;nminus1</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1014.25,-1003.89C1021.73,-908.52 1036.19,-752.83 1059,-620 1081.94,-486.44 1119.95,-334.63 1145.72,-237.91"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1148.23,-238.7 1147.63,-230.77 1143.15,-237.34 1148.23,-238.7"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1059,-620 1059,-642.8 1272,-642.8 1272,-620 1059,-620"/>
<text xml:space="preserve" text-anchor="start" x="1062" y="-639.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: grid&#45;validated definition</text>
</g>
<!-- loadgrid&#45;&gt;inputvalidation -->
<g id="edge2" class="edge">
<title>loadgrid&#45;&gt;inputvalidation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M701,-3595.47C701,-3554.27 701,-3505.16 701,-3462.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="703.63,-3462.96 701,-3455.46 698.38,-3462.96 703.63,-3462.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="701,-3512.6 701,-3535.4 900,-3535.4 900,-3512.6 701,-3512.6"/>
<text xml:space="preserve" text-anchor="start" x="704" y="-3532.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: unconverted network</text>
</g>
<!-- loadgrid&#45;&gt;trafoconversion -->
<g id="edge3" class="edge">
<title>loadgrid&#45;&gt;trafoconversion</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M837.36,-3595.4C867.02,-3575.8 898.15,-3555.02 927,-3535.4 963.19,-3510.79 1002.09,-3483.77 1038.01,-3458.59"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1039.47,-3460.77 1044.1,-3454.32 1036.46,-3456.47 1039.47,-3460.77"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="953.79,-3512.6 953.79,-3535.4 1053.79,-3535.4 1053.79,-3512.6 953.79,-3512.6"/>
<text xml:space="preserve" text-anchor="start" x="956.79" y="-3532.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">parsed network</text>
</g>
<!-- inputvalidation&#45;&gt;converttrafo3w -->
<g id="edge6" class="edge">
<title>inputvalidation&#45;&gt;converttrafo3w</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M701,-3272.75C701,-3208.35 701,-3119.19 701,-3040.8 701,-3040.8 701,-3040.8 701,-1738.4 701,-1644.69 770.23,-1567.64 842.93,-1512.56"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="844.48,-1514.68 848.92,-1508.09 841.34,-1510.47 844.48,-1514.68"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="701,-2374.4 701,-2414 889,-2414 889,-2374.4 701,-2374.4"/>
<text xml:space="preserve" text-anchor="start" x="704" y="-2411" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: grid&#45;validated input</text>
<text xml:space="preserve" text-anchor="start" x="704" y="-2394.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">definition</text>
</g>
<!-- inputvalidation&#45;&gt;networkreduction -->
<g id="edge5" class="edge">
<title>inputvalidation&#45;&gt;networkreduction</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M780.6,-3272.63C805.33,-3245.65 832.95,-3216.18 859,-3189.8 876.31,-3172.27 895.07,-3153.99 913.42,-3136.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="914.89,-3138.71 918.51,-3131.63 911.27,-3134.9 914.89,-3138.71"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="859,-3189.8 859,-3212.6 1063,-3212.6 1063,-3189.8 859,-3189.8"/>
<text xml:space="preserve" text-anchor="start" x="862" y="-3209.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey A: voltage levels to keep</text>
</g>
<!-- trafoconversion&#45;&gt;networkreduction -->
<g id="edge7" class="edge">
<title>trafoconversion&#45;&gt;networkreduction</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1129.03,-3272.67C1108.96,-3231.12 1085,-3181.53 1064.4,-3138.9"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1066.86,-3137.97 1061.24,-3132.36 1062.14,-3140.25 1066.86,-3137.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1099.15,-3189.8 1099.15,-3212.6 1218.15,-3212.6 1218.15,-3189.8 1099.15,-3189.8"/>
<text xml:space="preserve" text-anchor="start" x="1102.15" y="-3209.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">converted network</text>
</g>
<!-- networkreduction&#45;&gt;whitelists -->
<g id="edge8" class="edge">
<title>networkreduction&#45;&gt;whitelists</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1043.34,-2949.87C1055.56,-2908.58 1070.14,-2859.35 1082.71,-2816.9"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1085.22,-2817.67 1084.83,-2809.73 1080.19,-2816.18 1085.22,-2817.67"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1067.35,-2867 1067.35,-2889.8 1175.35,-2889.8 1175.35,-2867 1067.35,-2867"/>
<text xml:space="preserve" text-anchor="start" x="1070.35" y="-2886.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">reduced network</text>
</g>
<!-- whitelists&#45;&gt;convergingparams -->
<g id="edge9" class="edge">
<title>whitelists&#45;&gt;convergingparams</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1128.08,-2627.07C1135.53,-2585.87 1144.41,-2536.76 1152.07,-2494.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1154.65,-2494.86 1153.4,-2487.01 1149.48,-2493.92 1154.65,-2494.86"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1142.74,-2544.2 1142.74,-2567 1244.74,-2567 1244.74,-2544.2 1142.74,-2544.2"/>
<text xml:space="preserve" text-anchor="start" x="1145.74" y="-2564" font-family="Arial" font-size="14.00" fill="#c9c9c9">scoped network</text>
</g>
<!-- convergingparams&#45;&gt;networkmasks -->
<g id="edge10" class="edge">
<title>convergingparams&#45;&gt;networkmasks</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1170,-2304.27C1170,-2263.07 1170,-2213.96 1170,-2171.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1172.63,-2171.76 1170,-2164.26 1167.38,-2171.76 1172.63,-2171.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1170,-2221.4 1170,-2244.2 1317,-2244.2 1317,-2221.4 1170,-2221.4"/>
<text xml:space="preserve" text-anchor="start" x="1173" y="-2241.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">converging parameters</text>
</g>
<!-- networkmasks&#45;&gt;maskdefinition -->
<g id="edge11" class="edge">
<title>networkmasks&#45;&gt;maskdefinition</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1126.37,-1981.53C1104.67,-1937.33 1078.39,-1883.79 1056.11,-1838.4"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1058.56,-1837.43 1052.9,-1831.86 1053.85,-1839.75 1058.56,-1837.43"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1093.86,-1898.6 1093.86,-1921.4 1205.86,-1921.4 1205.86,-1898.6 1093.86,-1898.6"/>
<text xml:space="preserve" text-anchor="start" x="1096.86" y="-1918.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">journey B: masks</text>
</g>
<!-- securityanalysis&#45;&gt;dcpreprocess -->
<g id="edge13" class="edge">
<title>securityanalysis&#45;&gt;dcpreprocess</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M739,-662.8C739,-631.93 739,-599.45 739,-570.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="741.63,-570.36 739,-562.86 736.38,-570.36 741.63,-570.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="656,-611.13 656,-633.93 739,-633.93 739,-611.13 656,-611.13"/>
<text xml:space="preserve" text-anchor="start" x="659" y="-630.93" font-family="Arial" font-size="14.00" fill="#c9c9c9">ImportResult</text>
</g>
<!-- dcpreprocess&#45;&gt;dcnminus1 -->
<g id="edge4" class="edge">
<title>dcpreprocess&#45;&gt;dcnminus1</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M739,-380.13C739,-336.3 739,-283.28 739,-238.14"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="741.63,-238.27 739,-230.77 736.38,-238.27 741.63,-238.27"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="739,-297.2 739,-320 917,-320 917,-297.2 739,-297.2"/>
<text xml:space="preserve" text-anchor="start" x="742" y="-317" font-family="Arial" font-size="14.00" fill="#c9c9c9">contingencies DC computes</text>
</g>
</g>
</svg>`;case`dcWorkerInternals`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1975pt" height="2615pt"
 viewBox="0.00 0.00 1975.00 2615.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 2600.45)">
<g id="clust1" class="cluster">
<title>cluster_staticinfo</title>
<polygon fill="#3e4651" stroke="#2d333d" points="1045,-2312.2 1045,-2577.4 1457,-2577.4 1457,-2312.2 1045,-2312.2"/>
<text xml:space="preserve" text-anchor="start" x="1053" y="-2563.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">STATIC_INFORMATION.HDF5</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_dcoptimizer</title>
<polygon fill="#3a404a" stroke="#292f37" points="75,-8 75,-2275.2 1519,-2275.2 1519,-8 75,-8"/>
<text xml:space="preserve" text-anchor="start" x="83" y="-2261.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">DC&#45;OPTIMIZER</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_dcsolver</title>
<polygon fill="#2225aa" stroke="#2a2490" points="187,-48 187,-1297.6 705,-1297.6 705,-48 187,-48"/>
<text xml:space="preserve" text-anchor="start" x="195" y="-1284.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c7d2fe" fill-opacity="0.701961">GPU DC LOADFLOW SOLVER</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_kafka</title>
<polygon fill="#3e4651" stroke="#2d333d" points="1527,-1028.4 1527,-1285.6 1937,-1285.6 1937,-1028.4 1527,-1028.4"/>
<text xml:space="preserve" text-anchor="start" x="1535" y="-1272.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">KAFKA</text>
</g>
<!-- branchactionset -->
<g id="node1" class="node">
<title>branchactionset</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1425.06,-2524.2 1076.94,-2524.2 1076.94,-2344.2 1425.06,-2344.2 1425.06,-2524.2"/>
<text xml:space="preserve" text-anchor="start" x="1177.5" y="-2493.2" font-family="Arial" font-size="20.00" fill="#f8fafc">BranchActionSet</text>
<text xml:space="preserve" text-anchor="start" x="1097" y="-2465.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">What the DC&#45;Optimizer actually samples from</text>
<text xml:space="preserve" text-anchor="start" x="1172.5" y="-2447.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">&#45;&#45; a different asset from</text>
<text xml:space="preserve" text-anchor="start" x="1109.5" y="-2429.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">action_set.json, in a different format and a</text>
<text xml:space="preserve" text-anchor="start" x="1210" y="-2411.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">different file.</text>
<text xml:space="preserve" text-anchor="start" x="1115" y="-2393.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Padded boolean arrays (branch_actions,</text>
</g>
<!-- bsdfstage -->
<g id="node2" class="node">
<title>bsdfstage</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="583.06,-1236.4 226.94,-1236.4 226.94,-1056.4 583.06,-1056.4 583.06,-1236.4"/>
<text xml:space="preserve" text-anchor="start" x="264" y="-1205.4" font-family="Arial" font-size="20.00" fill="#eef2ff">compute_bsdf_lodf_static_flows</text>
<text xml:space="preserve" text-anchor="start" x="247" y="-1177.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">Everything that changes the PTDF, in one pass</text>
<text xml:space="preserve" text-anchor="start" x="336" y="-1159.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">per branch topology:</text>
<text xml:space="preserve" text-anchor="start" x="264.5" y="-1141.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">BSDF for each busbar split, MODF for the</text>
<text xml:space="preserve" text-anchor="start" x="321.5" y="-1123.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">remedial disconnections,</text>
<text xml:space="preserve" text-anchor="start" x="278" y="-1105.4" font-family="Arial" font-size="15.00" fill="#c7d2fe">the LODF matrix, and the static flows.</text>
</g>
<!-- n0stage -->
<g id="node3" class="node">
<title>n0stage</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="576.06,-913.6 237.94,-913.6 237.94,-733.6 576.06,-733.6 576.06,-913.6"/>
<text xml:space="preserve" text-anchor="start" x="366" y="-882.6" font-family="Arial" font-size="20.00" fill="#eef2ff">N&#45;0 flows</text>
<text xml:space="preserve" text-anchor="start" x="258" y="-854.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">Nodal injections against the already&#45;updated</text>
<text xml:space="preserve" text-anchor="start" x="358" y="-836.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">PTDF, plus the</text>
<text xml:space="preserve" text-anchor="start" x="281" y="-818.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">cross&#45;coupler flows across each split,</text>
<text xml:space="preserve" text-anchor="start" x="312" y="-800.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">corrected for disconnections</text>
<text xml:space="preserve" text-anchor="start" x="259.5" y="-782.6" font-family="Arial" font-size="15.00" fill="#c7d2fe">and PST taps. Cheap, because the PTDF is</text>
</g>
<!-- n1stage -->
<g id="node4" class="node">
<title>n1stage</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="590.06,-590.8 227.94,-590.8 227.94,-410.8 590.06,-410.8 590.06,-590.8"/>
<text xml:space="preserve" text-anchor="start" x="289.5" y="-550.8" font-family="Arial" font-size="20.00" fill="#eef2ff">Contingency analysis (N&#45;1)</text>
<text xml:space="preserve" text-anchor="start" x="248" y="-522.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">The N&#45;1 matrix from the LODF and multi&#45;outage</text>
<text xml:space="preserve" text-anchor="start" x="343" y="-504.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">factors, plus busbar</text>
<text xml:space="preserve" text-anchor="start" x="255" y="-486.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">outage and injection outage cases. Runs over</text>
<text xml:space="preserve" text-anchor="start" x="354" y="-468.8" font-family="Arial" font-size="15.00" fill="#c7d2fe">the whole batch.</text>
</g>
<!-- resultextraction -->
<g id="node5" class="node">
<title>resultextraction</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="593.06,-268 226.94,-268 226.94,-88 593.06,-88 593.06,-268"/>
<text xml:space="preserve" text-anchor="start" x="247" y="-237" font-family="Arial" font-size="20.00" fill="#eef2ff">Result aggregation and sparsification</text>
<text xml:space="preserve" text-anchor="start" x="270.5" y="-209" font-family="Arial" font-size="15.00" fill="#c7d2fe">The full N&#45;1 matrix is far too large to keep</text>
<text xml:space="preserve" text-anchor="start" x="340.5" y="-191" font-family="Arial" font-size="15.00" fill="#c7d2fe">per topology, so only</text>
<text xml:space="preserve" text-anchor="start" x="271" y="-173" font-family="Arial" font-size="15.00" fill="#c7d2fe">the worst entries survive: a top&#45;k over the</text>
<text xml:space="preserve" text-anchor="start" x="347" y="-155" font-family="Arial" font-size="15.00" fill="#c7d2fe">flattened matrix for</text>
<text xml:space="preserve" text-anchor="start" x="272.5" y="-137" font-family="Arial" font-size="15.00" fill="#c7d2fe">storage, and a per&#45;case worst&#45;k that tells</text>
</g>
<!-- scoring -->
<g id="node6" class="node">
<title>scoring</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1411.06,-2214 1090.94,-2214 1090.94,-2034 1411.06,-2034 1411.06,-2214"/>
<text xml:space="preserve" text-anchor="start" x="1217.5" y="-2183" font-family="Arial" font-size="20.00" fill="#eff6ff">Scoring</text>
<text xml:space="preserve" text-anchor="start" x="1118" y="-2155" font-family="Arial" font-size="15.00" fill="#bfdbfe">Turns raw flows into the metric vector &#45;&#45;</text>
<text xml:space="preserve" text-anchor="start" x="1171.5" y="-2137" font-family="Arial" font-size="15.00" fill="#bfdbfe">overload energy, critical</text>
<text xml:space="preserve" text-anchor="start" x="1111" y="-2119" font-family="Arial" font-size="15.00" fill="#bfdbfe">branch counts under N&#45;0 and N&#45;1, busbar</text>
<text xml:space="preserve" text-anchor="start" x="1179.5" y="-2101" font-family="Arial" font-size="15.00" fill="#bfdbfe">outage penalty &#45;&#45; and</text>
<text xml:space="preserve" text-anchor="start" x="1111" y="-2083" font-family="Arial" font-size="15.00" fill="#bfdbfe">aggregates it into the scalar fitness that is</text>
</g>
<!-- repertoire -->
<g id="node7" class="node">
<title>repertoire</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1471.06,-1891.2 1132.94,-1891.2 1132.94,-1711.2 1471.06,-1711.2 1471.06,-1891.2"/>
<text xml:space="preserve" text-anchor="start" x="1169" y="-1860.2" font-family="Arial" font-size="20.00" fill="#f8fafc">Discrete MAP&#45;Elites repertoire</text>
<text xml:space="preserve" text-anchor="start" x="1169.5" y="-1832.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">Cells indexed by the switching&#45;distance</text>
<text xml:space="preserve" text-anchor="start" x="1208.5" y="-1814.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">descriptors: disconnections,</text>
<text xml:space="preserve" text-anchor="start" x="1153" y="-1796.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">split substations and reassignment distance.</text>
<text xml:space="preserve" text-anchor="start" x="1223.5" y="-1778.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">Each cell keeps its own</text>
<text xml:space="preserve" text-anchor="start" x="1179" y="-1760.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">elites (cell_depth), so a conservative</text>
</g>
<!-- mutation -->
<g id="node8" class="node">
<title>mutation</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="571.06,-1568.4 226.94,-1568.4 226.94,-1388.4 571.06,-1388.4 571.06,-1568.4"/>
<text xml:space="preserve" text-anchor="start" x="361.5" y="-1537.4" font-family="Arial" font-size="20.00" fill="#ffe0c2">Mutation</text>
<text xml:space="preserve" text-anchor="start" x="253.5" y="-1509.4" font-family="Arial" font-size="15.00" fill="#f9b27c">Per genome, a Poisson&#45;sampled number of</text>
<text xml:space="preserve" text-anchor="start" x="299.5" y="-1491.4" font-family="Arial" font-size="15.00" fill="#f9b27c">substation mutations followed</text>
<text xml:space="preserve" text-anchor="start" x="247" y="-1473.4" font-family="Arial" font-size="15.00" fill="#f9b27c">by one disconnection mutation, each drawing</text>
<text xml:space="preserve" text-anchor="start" x="300.5" y="-1455.4" font-family="Arial" font-size="15.00" fill="#f9b27c">ADD / CHANGE / REMOVE /</text>
<text xml:space="preserve" text-anchor="start" x="269.5" y="-1437.4" font-family="Arial" font-size="15.00" fill="#f9b27c">IDENTITY. Feasibility is enforced while</text>
</g>
<!-- crossover -->
<g id="node9" class="node">
<title>crossover</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1015.06,-1568.4 680.94,-1568.4 680.94,-1388.4 1015.06,-1388.4 1015.06,-1568.4"/>
<text xml:space="preserve" text-anchor="start" x="802.5" y="-1528.4" font-family="Arial" font-size="20.00" fill="#ffe0c2">Crossover</text>
<text xml:space="preserve" text-anchor="start" x="702.5" y="-1500.4" font-family="Arial" font-size="15.00" fill="#f9b27c">Builds an offspring by sampling actions and</text>
<text xml:space="preserve" text-anchor="start" x="768" y="-1482.4" font-family="Arial" font-size="15.00" fill="#f9b27c">disconnections from the</text>
<text xml:space="preserve" text-anchor="start" x="701" y="-1464.4" font-family="Arial" font-size="15.00" fill="#f9b27c">union of two parents, biased toward the first</text>
<text xml:space="preserve" text-anchor="start" x="824.5" y="-1446.4" font-family="Arial" font-size="15.00" fill="#f9b27c">parent.</text>
</g>
<!-- pusher -->
<g id="node10" class="node">
<title>pusher</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1479.06,-1568.4 1124.94,-1568.4 1124.94,-1388.4 1479.06,-1388.4 1479.06,-1568.4"/>
<text xml:space="preserve" text-anchor="start" x="1223" y="-1537.4" font-family="Arial" font-size="20.00" fill="#f8fafc">Epoch result push</text>
<text xml:space="preserve" text-anchor="start" x="1145" y="-1509.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">How topologies leave the DC stage. At the end</text>
<text xml:space="preserve" text-anchor="start" x="1225.5" y="-1491.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">of each epoch the new</text>
<text xml:space="preserve" text-anchor="start" x="1161" y="-1473.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">elites are pulled off the GPU, converted to</text>
<text xml:space="preserve" text-anchor="start" x="1233" y="-1455.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">TopologyPushResult</text>
<text xml:space="preserve" text-anchor="start" x="1150" y="-1437.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">messages and produced to the \`results\` topic</text>
</g>
<!-- results -->
<g id="node11" class="node">
<title>results</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1905.02,-1232.4 1558.98,-1232.4 1558.98,-1060.4 1905.02,-1060.4 1905.02,-1232.4"/>
<text xml:space="preserve" text-anchor="start" x="1703.5" y="-1196.4" font-family="Arial" font-size="20.00" fill="#f8fafc">results</text>
<text xml:space="preserve" text-anchor="start" x="1590.5" y="-1168.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">The one shared topic. Both stages publish</text>
<text xml:space="preserve" text-anchor="start" x="1653" y="-1150.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">topologies here and the</text>
<text xml:space="preserve" text-anchor="start" x="1583" y="-1132.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">AC&#45;Validator also consumes it to pick up DC</text>
<text xml:space="preserve" text-anchor="start" x="1693.5" y="-1114.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">candidates.</text>
</g>
<!-- branchactionset&#45;&gt;scoring -->
<g id="edge1" class="edge">
<title>branchactionset&#45;&gt;scoring</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1251,-2344.47C1251,-2325.89 1251,-2305.76 1251,-2285.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1253.63,-2285.73 1251,-2278.23 1248.38,-2285.73 1253.63,-2285.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1025,-2269.96 1025,-2309.56 1251,-2309.56 1251,-2269.96 1025,-2269.96"/>
<text xml:space="preserve" text-anchor="start" x="1028" y="-2306.56" font-family="Arial" font-size="14.00" fill="#c9c9c9">sampling space &#45;&#45; indices into these</text>
<text xml:space="preserve" text-anchor="start" x="1028" y="-2289.76" font-family="Arial" font-size="14.00" fill="#c9c9c9">arrays</text>
</g>
<!-- bsdfstage&#45;&gt;n0stage -->
<g id="edge9" class="edge">
<title>bsdfstage&#45;&gt;n0stage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M405.55,-1056.47C405.81,-1015.27 406.12,-966.16 406.38,-923.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="409.01,-923.97 406.43,-916.46 403.76,-923.94 409.01,-923.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="406.06,-973.6 406.06,-996.4 666.06,-996.4 666.06,-973.6 406.06,-973.6"/>
<text xml:space="preserve" text-anchor="start" x="409.06" y="-993.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">updated PTDF, LODF, MODF, static flows</text>
</g>
<!-- n0stage&#45;&gt;n1stage -->
<g id="edge10" class="edge">
<title>n0stage&#45;&gt;n1stage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M407.55,-733.67C407.81,-692.47 408.12,-643.36 408.38,-600.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="411.01,-601.17 408.43,-593.66 405.76,-601.14 411.01,-601.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="408.06,-650.8 408.06,-673.6 598.06,-673.6 598.06,-650.8 408.06,-650.8"/>
<text xml:space="preserve" text-anchor="start" x="411.06" y="-670.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">N&#45;0 flows and nodal injections</text>
</g>
<!-- n1stage&#45;&gt;resultextraction -->
<g id="edge11" class="edge">
<title>n1stage&#45;&gt;resultextraction</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M409.28,-410.87C409.41,-369.67 409.56,-320.56 409.69,-278.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="412.32,-278.36 409.71,-270.86 407.07,-278.35 412.32,-278.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="409.53,-328 409.53,-350.8 478.53,-350.8 478.53,-328 409.53,-328"/>
<text xml:space="preserve" text-anchor="start" x="412.53" y="-347.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">N&#45;1 matrix</text>
</g>
<!-- resultextraction&#45;&gt;scoring -->
<g id="edge12" class="edge">
<title>resultextraction&#45;&gt;scoring</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M187,-336.48C146.08,-382.83 116,-437.94 116,-499.8 116,-1802.2 116,-1802.2 116,-1802.2 116,-2000.27 764.2,-2082.18 1080.89,-2110.38"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1080.37,-2112.97 1088.07,-2111.02 1080.83,-2107.74 1080.37,-2112.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="0,-1639.4 0,-1662.2 116,-1662.2 116,-1639.4 0,-1639.4"/>
<text xml:space="preserve" text-anchor="start" x="3" y="-1659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">N&#45;0 and N&#45;1 flows</text>
</g>
<!-- scoring&#45;&gt;repertoire -->
<g id="edge2" class="edge">
<title>scoring&#45;&gt;repertoire</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1265.14,-2034.07C1271.69,-1992.87 1279.49,-1943.76 1286.23,-1901.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1288.82,-1901.84 1287.4,-1894.02 1283.63,-1901.01 1288.82,-1901.84"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1278.03,-1951.2 1278.03,-1974 1506.03,-1974 1506.03,-1951.2 1278.03,-1951.2"/>
<text xml:space="preserve" text-anchor="start" x="1281.03" y="-1971" font-family="Arial" font-size="14.00" fill="#c9c9c9">fitness and descriptors, sorted insert</text>
</g>
<!-- repertoire&#45;&gt;mutation -->
<g id="edge3" class="edge">
<title>repertoire&#45;&gt;mutation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1133.21,-1744.58C996.03,-1698.83 797.78,-1631.49 626,-1568.4 611.19,-1562.96 595.89,-1557.22 580.54,-1551.38"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="581.75,-1549.03 573.81,-1548.81 579.88,-1553.94 581.75,-1549.03"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="848.11,-1628.4 848.11,-1651.2 942.11,-1651.2 942.11,-1628.4 848.11,-1628.4"/>
<text xml:space="preserve" text-anchor="start" x="851.11" y="-1648.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">sampled elites</text>
</g>
<!-- repertoire&#45;&gt;crossover -->
<g id="edge4" class="edge">
<title>repertoire&#45;&gt;crossover</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1176.14,-1711.27C1115.86,-1668.67 1043.63,-1617.63 982.29,-1574.29"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="983.82,-1572.16 976.18,-1569.98 980.8,-1576.45 983.82,-1572.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1088.61,-1628.4 1088.61,-1651.2 1180.61,-1651.2 1180.61,-1628.4 1088.61,-1628.4"/>
<text xml:space="preserve" text-anchor="start" x="1091.61" y="-1648.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">sampled pairs</text>
</g>
<!-- repertoire&#45;&gt;pusher -->
<g id="edge5" class="edge">
<title>repertoire&#45;&gt;pusher</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1302,-1711.27C1302,-1670.07 1302,-1620.96 1302,-1578.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1304.63,-1578.76 1302,-1571.26 1299.38,-1578.76 1304.63,-1578.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1302,-1628.4 1302,-1651.2 1369,-1651.2 1369,-1628.4 1302,-1628.4"/>
<text xml:space="preserve" text-anchor="start" x="1305" y="-1648.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">new elites</text>
</g>
<!-- mutation&#45;&gt;bsdfstage -->
<g id="edge6" class="edge">
<title>mutation&#45;&gt;bsdfstage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M400.62,-1388.53C401.07,-1363.56 401.58,-1335.6 402.08,-1307.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="404.7,-1308.17 402.21,-1300.62 399.45,-1308.08 404.7,-1308.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="298.44,-1343.16 298.44,-1365.96 401.44,-1365.96 401.44,-1343.16 298.44,-1343.16"/>
<text xml:space="preserve" text-anchor="start" x="301.44" y="-1362.96" font-family="Arial" font-size="14.00" fill="#c9c9c9">candidate batch</text>
</g>
<!-- crossover&#45;&gt;bsdfstage -->
<g id="edge7" class="edge">
<title>crossover&#45;&gt;bsdfstage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M728.7,-1388.53C693.56,-1362.35 654.02,-1332.9 614.96,-1303.8"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="616.64,-1301.78 609.06,-1299.41 613.5,-1305.99 616.64,-1301.78"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="564.45,-1342.9 564.45,-1365.7 667.45,-1365.7 667.45,-1342.9 564.45,-1342.9"/>
<text xml:space="preserve" text-anchor="start" x="567.45" y="-1362.7" font-family="Arial" font-size="14.00" fill="#c9c9c9">candidate batch</text>
</g>
<!-- pusher&#45;&gt;results -->
<g id="edge8" class="edge">
<title>pusher&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1417.8,-1388.53C1478.38,-1342.04 1552.43,-1285.21 1613.42,-1238.4"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1614.8,-1240.65 1619.15,-1234 1611.61,-1236.49 1614.8,-1240.65"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1517,-1305.6 1517,-1328.4 1714,-1328.4 1714,-1305.6 1517,-1305.6"/>
<text xml:space="preserve" text-anchor="start" x="1520" y="-1325.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">TopologyPushResult per epoch</text>
</g>
</g>
</svg>`;case`acValidatorInternals`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1853pt" height="3179pt"
 viewBox="0.00 0.00 1853.00 3179.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3164.45)">
<g id="clust1" class="cluster">
<title>cluster_kafka</title>
<polygon fill="#3e4651" stroke="#2d333d" points="8,-2872.2 8,-3129.4 418,-3129.4 418,-2872.2 8,-2872.2"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-3115.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">KAFKA</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_processedgrid</title>
<polygon fill="#3e4651" stroke="#2d333d" points="456,-2860.2 456,-3141.4 1815,-3141.4 1815,-2860.2 456,-2860.2"/>
<text xml:space="preserve" text-anchor="start" x="464" y="-3127.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">PROCESSED GRID FOLDER</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_acvalidator</title>
<polygon fill="#3a404a" stroke="#292f37" points="340,-303 340,-2809.4 1061,-2809.4 1061,-303 340,-303"/>
<text xml:space="preserve" text-anchor="start" x="348" y="-2795.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">AC&#45;VALIDATOR</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_selectstrategy</title>
<polygon fill="#5a3620" stroke="#462a17" points="452,-1572.4 452,-2499.2 874,-2499.2 874,-1572.4 452,-1572.4"/>
<text xml:space="preserve" text-anchor="start" x="460" y="-2485.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#f9b27c" fill-opacity="0.701961">SELECT_STRATEGY</text>
</g>
<!-- results -->
<g id="node1" class="node">
<title>results</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="386.02,-3076.2 39.98,-3076.2 39.98,-2904.2 386.02,-2904.2 386.02,-3076.2"/>
<text xml:space="preserve" text-anchor="start" x="184.5" y="-3040.2" font-family="Arial" font-size="20.00" fill="#f8fafc">results</text>
<text xml:space="preserve" text-anchor="start" x="71.5" y="-3012.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The one shared topic. Both stages publish</text>
<text xml:space="preserve" text-anchor="start" x="134" y="-2994.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">topologies here and the</text>
<text xml:space="preserve" text-anchor="start" x="64" y="-2976.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">AC&#45;Validator also consumes it to pick up DC</text>
<text xml:space="preserve" text-anchor="start" x="174.5" y="-2958.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">candidates.</text>
</g>
<!-- gridsnapshot -->
<g id="node2" class="node">
<title>gridsnapshot</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="830.06,-3080.2 495.94,-3080.2 495.94,-2900.2 830.06,-2900.2 830.06,-3080.2"/>
<text xml:space="preserve" text-anchor="start" x="575" y="-3022.2" font-family="Arial" font-size="20.00" fill="#f8fafc">grid.xiidm / grid.json</text>
<text xml:space="preserve" text-anchor="start" x="516" y="-2994.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The normalized backend grid, written by the</text>
<text xml:space="preserve" text-anchor="start" x="633.5" y="-2976.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">importer.</text>
</g>
<!-- actionset -->
<g id="node3" class="node">
<title>actionset</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1317.56,-3080.2 940.44,-3080.2 940.44,-2900.2 1317.56,-2900.2 1317.56,-3080.2"/>
<text xml:space="preserve" text-anchor="start" x="960.5" y="-3049.2" font-family="Arial" font-size="20.00" fill="#f8fafc">action_set.json + action_set_diffs.hdf5</text>
<text xml:space="preserve" text-anchor="start" x="990" y="-3021.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The same action space in physical terms:</text>
<text xml:space="preserve" text-anchor="start" x="1032.5" y="-3003.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">station&#45;local reconfigurations</text>
<text xml:space="preserve" text-anchor="start" x="966.5" y="-2985.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">A and disconnectable branches D, expressed as</text>
<text xml:space="preserve" text-anchor="start" x="1049" y="-2967.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">switch positions against</text>
<text xml:space="preserve" text-anchor="start" x="1066" y="-2949.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the asset topology.</text>
</g>
<!-- snapshots -->
<g id="node4" class="node">
<title>snapshots</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1774.56,-3080.2 1427.44,-3080.2 1427.44,-2900.2 1774.56,-2900.2 1774.56,-3080.2"/>
<text xml:space="preserve" text-anchor="start" x="1496.5" y="-3022.2" font-family="Arial" font-size="20.00" fill="#f8fafc">optimizer_snapshots/ac</text>
<text xml:space="preserve" text-anchor="start" x="1450.5" y="-2994.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Repertoire, realized asset topologies, AC/DC</text>
<text xml:space="preserve" text-anchor="start" x="1447.5" y="-2976.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow tables, SLDs, OpenRAO summaries.</text>
</g>
<!-- discriminator -->
<g id="node5" class="node">
<title>discriminator</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="823.02,-2438 502.98,-2438 502.98,-2258 823.02,-2258 823.02,-2438"/>
<text xml:space="preserve" text-anchor="start" x="584" y="-2380" font-family="Arial" font-size="20.00" fill="#ffe0c2">Discriminator filter</text>
<text xml:space="preserve" text-anchor="start" x="530" y="-2352" font-family="Arial" font-size="15.00" fill="#f9b27c">Drop candidates too close to something</text>
<text xml:space="preserve" text-anchor="start" x="603.5" y="-2334" font-family="Arial" font-size="15.00" fill="#f9b27c">already validated.</text>
</g>
<!-- dominator -->
<g id="node6" class="node">
<title>dominator</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="834.06,-2115.2 491.94,-2115.2 491.94,-1935.2 834.06,-1935.2 834.06,-2115.2"/>
<text xml:space="preserve" text-anchor="start" x="596.5" y="-2066.2" font-family="Arial" font-size="20.00" fill="#ffe0c2">Dominator filter</text>
<text xml:space="preserve" text-anchor="start" x="512" y="-2038.2" font-family="Arial" font-size="15.00" fill="#f9b27c">Drop a candidate if another topology reaches</text>
<text xml:space="preserve" text-anchor="start" x="597.5" y="-2020.2" font-family="Arial" font-size="15.00" fill="#f9b27c">similar or better DC</text>
<text xml:space="preserve" text-anchor="start" x="540" y="-2002.2" font-family="Arial" font-size="15.00" fill="#f9b27c">fitness at a lower switching distance.</text>
</g>
<!-- median -->
<g id="node7" class="node">
<title>median</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="828.56,-1792.4 497.44,-1792.4 497.44,-1612.4 828.56,-1612.4 828.56,-1792.4"/>
<text xml:space="preserve" text-anchor="start" x="609.5" y="-1734.4" font-family="Arial" font-size="20.00" fill="#ffe0c2">Median filter</text>
<text xml:space="preserve" text-anchor="start" x="517.5" y="-1706.4" font-family="Arial" font-size="15.00" fill="#f9b27c">Drop candidates whose fitness is below the</text>
<text xml:space="preserve" text-anchor="start" x="562" y="-1688.4" font-family="Arial" font-size="15.00" fill="#f9b27c">median of their descriptor cell.</text>
</g>
<!-- resultlistener -->
<g id="node8" class="node">
<title>resultlistener</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="829.56,-2748.2 496.44,-2748.2 496.44,-2568.2 829.56,-2568.2 829.56,-2748.2"/>
<text xml:space="preserve" text-anchor="start" x="600.5" y="-2718" font-family="Arial" font-size="20.00" fill="#f8fafc">Result listener</text>
<text xml:space="preserve" text-anchor="start" x="572.5" y="-2690" font-family="Arial" font-size="13.00" fill="#cbd5e1">SQLite (in&#45;memory), SQLModel</text>
<text xml:space="preserve" text-anchor="start" x="546" y="-2670.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Spools the results topic into a local</text>
<text xml:space="preserve" text-anchor="start" x="581" y="-2652.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">database, at startup and</text>
<text xml:space="preserve" text-anchor="start" x="516.5" y="-2634.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">between epochs, so candidates are already</text>
<text xml:space="preserve" text-anchor="start" x="574" y="-2616.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">staged when a run begins.</text>
</g>
<!-- worstk -->
<g id="node9" class="node">
<title>worstk</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="832.56,-1491.4 493.44,-1491.4 493.44,-1311.4 832.56,-1311.4 832.56,-1491.4"/>
<text xml:space="preserve" text-anchor="start" x="599" y="-1470.2" font-family="Arial" font-size="20.00" fill="#fbd3cb">Worst&#45;k epoch</text>
<text xml:space="preserve" text-anchor="start" x="630" y="-1442.2" font-family="Arial" font-size="13.00" fill="#f5b2a3">PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="513.5" y="-1422.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">Reruns only the handful of contingencies the</text>
<text xml:space="preserve" text-anchor="start" x="574.5" y="-1404.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">DC stage flagged as worst</text>
<text xml:space="preserve" text-anchor="start" x="524.5" y="-1386.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">for this topology. A candidate that already</text>
<text xml:space="preserve" text-anchor="start" x="582" y="-1368.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">fails there, or converges</text>
<text xml:space="preserve" text-anchor="start" x="553.5" y="-1350.6" font-family="Arial" font-size="15.00" fill="#f5b2a3">poorly, is rejected without the full</text>
</g>
<!-- remainingca -->
<g id="node10" class="node">
<title>remainingca</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1020.56,-1168.6 695.44,-1168.6 695.44,-988.6 1020.56,-988.6 1020.56,-1168.6"/>
<text xml:space="preserve" text-anchor="start" x="747.5" y="-1129.4" font-family="Arial" font-size="20.00" fill="#fbd3cb">Remaining contingencies</text>
<text xml:space="preserve" text-anchor="start" x="735.5" y="-1101.4" font-family="Arial" font-size="13.00" fill="#f5b2a3">PyPowSyBl security analysis, multiprocess</text>
<text xml:space="preserve" text-anchor="start" x="715.5" y="-1081.8" font-family="Arial" font-size="15.00" fill="#f5b2a3">Full AC N&#45;1 on the survivors, batched over</text>
<text xml:space="preserve" text-anchor="start" x="762.5" y="-1063.8" font-family="Arial" font-size="15.00" fill="#f5b2a3">runner processes. Hundreds</text>
<text xml:space="preserve" text-anchor="start" x="729" y="-1045.8" font-family="Arial" font-size="15.00" fill="#f5b2a3">of contingencies rather than a handful.</text>
</g>
<!-- acceptance -->
<g id="node11" class="node">
<title>acceptance</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="772.56,-845.8 413.44,-845.8 413.44,-665.8 772.56,-665.8 772.56,-845.8"/>
<text xml:space="preserve" text-anchor="start" x="493" y="-824.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Acceptance evaluation</text>
<text xml:space="preserve" text-anchor="start" x="541.5" y="-796.6" font-family="Arial" font-size="13.00" fill="#bfdbfe">polars LazyFrame</text>
<text xml:space="preserve" text-anchor="start" x="461.5" y="-777" font-family="Arial" font-size="15.00" fill="#bfdbfe">Detects constraint violations across the</text>
<text xml:space="preserve" text-anchor="start" x="500" y="-759" font-family="Arial" font-size="15.00" fill="#bfdbfe">loadflow tables and decides</text>
<text xml:space="preserve" text-anchor="start" x="433.5" y="-741" font-family="Arial" font-size="15.00" fill="#bfdbfe">whether a topology passes. Polars because the</text>
<text xml:space="preserve" text-anchor="start" x="527.5" y="-723" font-family="Arial" font-size="15.00" fill="#bfdbfe">result volume is the</text>
<text xml:space="preserve" text-anchor="start" x="489" y="-705" font-family="Arial" font-size="15.00" fill="#bfdbfe">bottleneck, not the check itself.</text>
</g>
<!-- summarywriter -->
<g id="node12" class="node">
<title>summarywriter</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="755.56,-523 430.44,-523 430.44,-343 755.56,-343 755.56,-523"/>
<text xml:space="preserve" text-anchor="start" x="523.5" y="-465" font-family="Arial" font-size="20.00" fill="#f8fafc">Summary writer</text>
<text xml:space="preserve" text-anchor="start" x="450.5" y="-437" font-family="Arial" font-size="15.00" fill="#cbd5e1">Realized asset topologies, loadflow tables,</text>
<text xml:space="preserve" text-anchor="start" x="481.5" y="-419" font-family="Arial" font-size="15.00" fill="#cbd5e1">SLDs and OpenRAO summaries.</text>
</g>
<!-- contingency -->
<g id="node13" class="node">
<title>contingency</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1434.56,-845.8 1101.44,-845.8 1101.44,-665.8 1434.56,-665.8 1434.56,-845.8"/>
<text xml:space="preserve" text-anchor="start" x="1174" y="-824.6" font-family="Arial" font-size="20.00" fill="#f8fafc">Contingency analysis</text>
<text xml:space="preserve" text-anchor="start" x="1167.5" y="-796.6" font-family="Arial" font-size="13.00" fill="#cbd5e1">toop_engine_contingency_analysis</text>
<text xml:space="preserve" text-anchor="start" x="1133.5" y="-777" font-family="Arial" font-size="15.00" fill="#cbd5e1">Runs an N&#45;1 analysis against whichever</text>
<text xml:space="preserve" text-anchor="start" x="1175" y="-759" font-family="Arial" font-size="15.00" fill="#cbd5e1">backend holds the grid, and</text>
<text xml:space="preserve" text-anchor="start" x="1121.5" y="-741" font-family="Arial" font-size="15.00" fill="#cbd5e1">normalizes both into the same result object.</text>
<text xml:space="preserve" text-anchor="start" x="1181" y="-723" font-family="Arial" font-size="15.00" fill="#cbd5e1">The two backends are not</text>
<text xml:space="preserve" text-anchor="start" x="1129.5" y="-705" font-family="Arial" font-size="15.00" fill="#cbd5e1">at feature parity, so which one you import</text>
</g>
<!-- interfaces -->
<g id="node14" class="node">
<title>interfaces</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="770.56,-190.1 415.44,-190.1 415.44,-10.1 770.56,-10.1 770.56,-190.1"/>
<text xml:space="preserve" text-anchor="start" x="550" y="-159.9" font-family="Arial" font-size="20.00" fill="#f8fafc">Interfaces</text>
<text xml:space="preserve" text-anchor="start" x="526" y="-131.9" font-family="Arial" font-size="13.00" fill="#cbd5e1">toop_engine_interfaces</text>
<text xml:space="preserve" text-anchor="start" x="437.5" y="-112.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">The shared vocabulary. Everything here exists</text>
<text xml:space="preserve" text-anchor="start" x="507" y="-94.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">so that two packages can</text>
<text xml:space="preserve" text-anchor="start" x="435.5" y="-76.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">agree on a payload without depending on each</text>
<text xml:space="preserve" text-anchor="start" x="574" y="-58.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">other.</text>
</g>
<!-- loadflowstore -->
<g id="node15" class="node">
<title>loadflowstore</title>
<path fill="#64748b" stroke="#475569" stroke-width="2" d="M1350.06,-182C1350.06,-192.04 1268.9,-200.2 1169,-200.2 1069.1,-200.2 987.94,-192.04 987.94,-182 987.94,-182 987.94,-18.2 987.94,-18.2 987.94,-8.16 1069.1,0 1169,0 1268.9,0 1350.06,-8.16 1350.06,-18.2 1350.06,-18.2 1350.06,-182 1350.06,-182"/>
<path fill="none" stroke="#475569" stroke-width="2" d="M1350.06,-182C1350.06,-171.96 1268.9,-163.8 1169,-163.8 1069.1,-163.8 987.94,-171.96 987.94,-182"/>
<text xml:space="preserve" text-anchor="start" x="1078" y="-168.9" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="1103" y="-140.9" font-family="Arial" font-size="13.00" fill="#cbd5e1">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1063" y="-121.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="1008" y="-103.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="1024.5" y="-85.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="1018.5" y="-67.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="1083.5" y="-49.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology it evaluates gets</text>
</g>
<!-- results&#45;&gt;resultlistener -->
<g id="edge3" class="edge">
<title>results&#45;&gt;resultlistener</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M328.89,-2904.22C391.57,-2858.25 469.16,-2801.35 533.79,-2753.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="535.07,-2756.27 539.56,-2749.72 531.96,-2752.04 535.07,-2756.27"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="438,-2817.4 438,-2840.2 531,-2840.2 531,-2817.4 438,-2817.4"/>
<text xml:space="preserve" text-anchor="start" x="441" y="-2837.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">DC topologies</text>
</g>
<!-- gridsnapshot&#45;&gt;resultlistener -->
<g id="edge1" class="edge">
<title>gridsnapshot&#45;&gt;resultlistener</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M663,-2900.33C663,-2875.36 663,-2847.4 663,-2819.62"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="665.63,-2819.92 663,-2812.42 660.38,-2819.92 665.63,-2819.92"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="601,-2854.96 601,-2877.76 663,-2877.76 663,-2854.96 601,-2854.96"/>
<text xml:space="preserve" text-anchor="start" x="604" y="-2874.76" font-family="Arial" font-size="14.00" fill="#c9c9c9">base grid</text>
</g>
<!-- actionset&#45;&gt;resultlistener -->
<g id="edge2" class="edge">
<title>actionset&#45;&gt;resultlistener</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1003.51,-2900.33C966.46,-2874.1 924.77,-2844.57 883.6,-2815.42"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="885.2,-2813.34 877.57,-2811.15 882.17,-2817.62 885.2,-2813.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="810.17,-2854.77 810.17,-2877.57 939.17,-2877.57 939.17,-2854.77 810.17,-2854.77"/>
<text xml:space="preserve" text-anchor="start" x="813.17" y="-2874.57" font-family="Arial" font-size="14.00" fill="#c9c9c9">to realize topologies</text>
</g>
<!-- discriminator&#45;&gt;dominator -->
<g id="edge5" class="edge">
<title>discriminator&#45;&gt;dominator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M663,-2258.07C663,-2216.87 663,-2167.76 663,-2125.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="665.63,-2125.56 663,-2118.06 660.38,-2125.56 665.63,-2125.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="663,-2175.2 663,-2198 724,-2198 724,-2175.2 663,-2175.2"/>
<text xml:space="preserve" text-anchor="start" x="666" y="-2195" font-family="Arial" font-size="14.00" fill="#c9c9c9">survivors</text>
</g>
<!-- dominator&#45;&gt;median -->
<g id="edge9" class="edge">
<title>dominator&#45;&gt;median</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M663,-1935.27C663,-1894.07 663,-1844.96 663,-1802.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="665.63,-1802.76 663,-1795.26 660.38,-1802.76 665.63,-1802.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="663,-1852.4 663,-1875.2 724,-1875.2 724,-1852.4 663,-1852.4"/>
<text xml:space="preserve" text-anchor="start" x="666" y="-1872.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">survivors</text>
</g>
<!-- median&#45;&gt;worstk -->
<g id="edge12" class="edge">
<title>median&#45;&gt;worstk</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M663,-1572.4C663,-1548.71 663,-1524.33 663,-1501.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="665.63,-1501.74 663,-1494.24 660.38,-1501.74 665.63,-1501.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="568,-1531.65 568,-1554.45 663,-1554.45 663,-1531.65 568,-1531.65"/>
<text xml:space="preserve" text-anchor="start" x="571" y="-1551.45" font-family="Arial" font-size="14.00" fill="#c9c9c9">selected batch</text>
</g>
<!-- resultlistener&#45;&gt;discriminator -->
<g id="edge4" class="edge">
<title>resultlistener&#45;&gt;discriminator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M663,-2568.47C663,-2549.89 663,-2529.76 663,-2509.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="665.63,-2509.73 663,-2502.23 660.38,-2509.73 665.63,-2509.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="567,-2533.56 567,-2556.36 663,-2556.36 663,-2533.56 567,-2533.56"/>
<text xml:space="preserve" text-anchor="start" x="570" y="-2553.36" font-family="Arial" font-size="14.00" fill="#c9c9c9">candidate pool</text>
</g>
<!-- worstk&#45;&gt;remainingca -->
<g id="edge6" class="edge">
<title>worstk&#45;&gt;remainingca</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M717.06,-1311.47C742.42,-1269.74 772.71,-1219.91 798.7,-1177.16"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="800.78,-1178.79 802.43,-1171.02 796.29,-1176.06 800.78,-1178.79"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="766.35,-1228.6 766.35,-1251.4 827.35,-1251.4 827.35,-1228.6 766.35,-1228.6"/>
<text xml:space="preserve" text-anchor="start" x="769.35" y="-1248.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">survivors</text>
</g>
<!-- worstk&#45;&gt;acceptance -->
<g id="edge7" class="edge">
<title>worstk&#45;&gt;acceptance</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M605.29,-1311.49C581.61,-1269.88 556.98,-1218.55 545,-1168.6 519.75,-1063.34 540.73,-939.64 562.34,-855.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="564.85,-856.28 564.21,-848.36 559.77,-854.95 564.85,-856.28"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="545,-1067.2 545,-1090 640,-1090 640,-1067.2 545,-1067.2"/>
<text xml:space="preserve" text-anchor="start" x="548" y="-1087" font-family="Arial" font-size="14.00" fill="#c9c9c9">worst&#45;k results</text>
</g>
<!-- worstk&#45;&gt;contingency -->
<g id="edge8" class="edge">
<title>worstk&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M832.54,-1334.66C914.36,-1296.16 1009.04,-1240.79 1076,-1168.6 1159.5,-1078.57 1212.45,-945.77 1241.12,-855.5"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1243.6,-856.36 1243.34,-848.42 1238.59,-854.79 1243.6,-856.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1189.63,-1067.2 1189.63,-1090 1328.63,-1090 1328.63,-1067.2 1189.63,-1067.2"/>
<text xml:space="preserve" text-anchor="start" x="1192.63" y="-1087" font-family="Arial" font-size="14.00" fill="#c9c9c9">worst&#45;k contingencies</text>
</g>
<!-- remainingca&#45;&gt;acceptance -->
<g id="edge10" class="edge">
<title>remainingca&#45;&gt;acceptance</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M784.53,-988.67C749.85,-946.68 708.39,-896.49 672.93,-853.56"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="675.11,-852.07 668.31,-847.96 671.06,-855.42 675.11,-852.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="733.45,-905.8 733.45,-928.6 827.45,-928.6 827.45,-905.8 733.45,-905.8"/>
<text xml:space="preserve" text-anchor="start" x="736.45" y="-925.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">full N&#45;1 results</text>
</g>
<!-- remainingca&#45;&gt;contingency -->
<g id="edge11" class="edge">
<title>remainingca&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M971.66,-988.67C1025.99,-946.16 1091.07,-895.24 1146.38,-851.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1147.8,-854.18 1152.09,-847.49 1144.56,-850.05 1147.8,-854.18"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1075.29,-905.8 1075.29,-928.6 1124.29,-928.6 1124.29,-905.8 1075.29,-905.8"/>
<text xml:space="preserve" text-anchor="start" x="1078.29" y="-925.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">full N&#45;1</text>
</g>
<!-- acceptance&#45;&gt;summarywriter -->
<g id="edge13" class="edge">
<title>acceptance&#45;&gt;summarywriter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M593,-665.87C593,-624.67 593,-575.56 593,-533.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="595.63,-533.36 593,-525.86 590.38,-533.36 595.63,-533.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="593,-583 593,-605.8 722,-605.8 722,-583 593,-583"/>
<text xml:space="preserve" text-anchor="start" x="596" y="-602.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topologies</text>
</g>
<!-- summarywriter&#45;&gt;snapshots -->
<g id="edge16" class="edge">
<title>summarywriter&#45;&gt;snapshots</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M755.51,-451.69C1040.63,-487.28 1601,-579.53 1601,-754.8 1601,-2659.2 1601,-2659.2 1601,-2659.2 1601,-2736.69 1601,-2824.43 1601,-2889.94"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1598.38,-2889.73 1601,-2897.23 1603.63,-2889.73 1598.38,-2889.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1601,-1691 1601,-1713.8 1762,-1713.8 1762,-1691 1601,-1691"/>
<text xml:space="preserve" text-anchor="start" x="1604" y="-1710.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">summaries and diagrams</text>
</g>
<!-- summarywriter&#45;&gt;interfaces -->
<g id="edge14" class="edge">
<title>summarywriter&#45;&gt;interfaces</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M593,-343.32C593,-299.28 593,-245.91 593,-200.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="595.63,-200.55 593,-193.05 590.38,-200.55 595.63,-200.55"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="593,-260.2 593,-283 711,-283 711,-260.2 593,-260.2"/>
<text xml:space="preserve" text-anchor="start" x="596" y="-280" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topology</text>
</g>
<!-- summarywriter&#45;&gt;loadflowstore -->
<g id="edge15" class="edge">
<title>summarywriter&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M817.55,-303C877.11,-268.79 940.48,-232.38 996.77,-200.04"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="998.03,-202.35 1003.23,-196.33 995.42,-197.79 998.03,-202.35"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="699.94,-209.18 699.94,-248.78 911.94,-248.78 911.94,-209.18 699.94,-209.18"/>
<text xml:space="preserve" text-anchor="start" x="702.94" y="-245.78" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="702.94" y="-228.98" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- interfaces&#45;&gt;loadflowstore -->
<g id="edge17" class="edge">
<title>interfaces&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M770.52,-100.1C835.92,-100.1 910.27,-100.1 976.9,-100.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="976.57,-102.73 984.07,-100.1 976.57,-97.48 976.57,-102.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="825.25,-103.1 825.25,-125.9 933.25,-125.9 933.25,-103.1 825.25,-103.1"/>
<text xml:space="preserve" text-anchor="start" x="828.25" y="-122.9" font-family="Arial" font-size="14.00" fill="#c9c9c9">persisted per job</text>
</g>
<!-- loadflowstore&#45;&gt;resultlistener -->
<g id="edge18" class="edge">
<title>loadflowstore&#45;&gt;resultlistener</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M987.07,-155.76C935.2,-170.77 878.51,-186.64 826,-200.2 675.69,-239.01 214,-218.98 214,-432 214,-2349 214,-2349 214,-2349 214,-2430.05 264.35,-2491.38 331.32,-2537.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="329.79,-2539.24 337.48,-2541.21 332.7,-2534.87 329.79,-2539.24"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="48,-1540.28 48,-1563.08 214,-1563.08 214,-1540.28 48,-1540.28"/>
<text xml:space="preserve" text-anchor="start" x="51" y="-1560.08" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial loadflow as baseline</text>
</g>
</g>
</svg>`;case`contingencyAnalysis`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="2008pt" height="3075pt"
 viewBox="0.00 0.00 2008.00 3075.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3060.45)">
<g id="clust1" class="cluster">
<title>cluster_toop</title>
<polygon fill="#353b43" stroke="#262b32" points="8,-8 8,-3037.4 1576,-3037.4 1576,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-3023.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_contingency</title>
<polygon fill="#3a404a" stroke="#292f37" points="500,-1033.4 500,-2643.3 1458,-2643.3 1458,-1033.4 500,-1033.4"/>
<text xml:space="preserve" text-anchor="start" x="508" y="-2629.85" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">CONTINGENCY ANALYSIS</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_pwca</title>
<polygon fill="#2225aa" stroke="#2a2490" points="1014,-2049.8 1014,-2315 1418,-2315 1418,-2049.8 1014,-2049.8"/>
<text xml:space="preserve" text-anchor="start" x="1022" y="-2301.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c7d2fe" fill-opacity="0.701961">RUN_CONTINGENCY_ANALYSIS_POWSYBL</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_ppca</title>
<polygon fill="#5a3620" stroke="#462a17" points="540,-1073.4 540,-2323 974,-2323 974,-1073.4 540,-1073.4"/>
<text xml:space="preserve" text-anchor="start" x="548" y="-2309.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#f9b27c" fill-opacity="0.701961">RUN_CONTINGENCY_ANALYSIS_PANDAPOWER</text>
</g>
<g id="clust5" class="cluster">
<title>cluster_interfaces</title>
<polygon fill="#3a404a" stroke="#292f37" points="48,-48 48,-1016.4 1536,-1016.4 1536,-48 48,-48"/>
<text xml:space="preserve" text-anchor="start" x="56" y="-1002.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">INTERFACES</text>
</g>
<g id="clust6" class="cluster">
<title>cluster_lfresults</title>
<polygon fill="#3e4651" stroke="#2d333d" points="80,-80 80,-963.2 1504,-963.2 1504,-80 80,-80"/>
<text xml:space="preserve" text-anchor="start" x="88" y="-949.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">LOADFLOWRESULTS</text>
</g>
<g id="clust10" class="cluster">
<title>cluster_importer</title>
<polygon fill="#3e4651" stroke="#2d333d" points="48,-2370.1 48,-2635.3 460,-2635.3 460,-2370.1 48,-2370.1"/>
<text xml:space="preserve" text-anchor="start" x="56" y="-2621.85" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">IMPORTER</text>
</g>
<g id="clust11" class="cluster">
<title>cluster_acvalidator</title>
<polygon fill="#3e4651" stroke="#2d333d" points="620,-2695 620,-2976.2 1536,-2976.2 1536,-2695 620,-2695"/>
<text xml:space="preserve" text-anchor="start" x="628" y="-2962.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">AC&#45;VALIDATOR</text>
</g>
<!-- pwlimitcache -->
<g id="node1" class="node">
<title>pwlimitcache</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1385.56,-2261.8 1046.44,-2261.8 1046.44,-2081.8 1385.56,-2081.8 1385.56,-2261.8"/>
<text xml:space="preserve" text-anchor="start" x="1135.5" y="-2203.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Branch limit cache</text>
<text xml:space="preserve" text-anchor="start" x="1066.5" y="-2175.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Caches operational limits across runs on the</text>
<text xml:space="preserve" text-anchor="start" x="1167" y="-2157.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">same network.</text>
</g>
<!-- ppoutagegrouping -->
<g id="node2" class="node">
<title>ppoutagegrouping</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="934.06,-2261.8 579.94,-2261.8 579.94,-2081.8 934.06,-2081.8 934.06,-2261.8"/>
<text xml:space="preserve" text-anchor="start" x="683" y="-2221.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Outage grouping</text>
<text xml:space="preserve" text-anchor="start" x="600" y="-2193.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Expands a contingency into every element that</text>
<text xml:space="preserve" text-anchor="start" x="704" y="-2175.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">goes out with it.</text>
<text xml:space="preserve" text-anchor="start" x="614" y="-2157.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Off by default; then each contingency is its</text>
<text xml:space="preserve" text-anchor="start" x="719.5" y="-2139.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">own group.</text>
</g>
<!-- ppslack -->
<g id="node3" class="node">
<title>ppslack</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="917.02,-1939 596.98,-1939 596.98,-1759 917.02,-1759 917.02,-1939"/>
<text xml:space="preserve" text-anchor="start" x="688" y="-1899" font-family="Arial" font-size="20.00" fill="#f8fafc">Slack allocation</text>
<text xml:space="preserve" text-anchor="start" x="619.5" y="-1871" font-family="Arial" font-size="15.00" fill="#cbd5e1">Gives each surviving island its own slack</text>
<text xml:space="preserve" text-anchor="start" x="681" y="-1853" font-family="Arial" font-size="15.00" fill="#cbd5e1">bus, above a minimum</text>
<text xml:space="preserve" text-anchor="start" x="637" y="-1835" font-family="Arial" font-size="15.00" fill="#cbd5e1">island size. Without this an islanded</text>
<text xml:space="preserve" text-anchor="start" x="675" y="-1817" font-family="Arial" font-size="15.00" fill="#cbd5e1">contingency simply fails.</text>
</g>
<!-- ppspps -->
<g id="node4" class="node">
<title>ppspps</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="925.06,-1616.2 588.94,-1616.2 588.94,-1436.2 925.06,-1436.2 925.06,-1616.2"/>
<text xml:space="preserve" text-anchor="start" x="680" y="-1585.2" font-family="Arial" font-size="20.00" fill="#f8fafc">SpPS rule engine</text>
<text xml:space="preserve" text-anchor="start" x="646" y="-1557.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Special Protection Schemes as a</text>
<text xml:space="preserve" text-anchor="start" x="662.5" y="-1539.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">condition/action rule engine.</text>
<text xml:space="preserve" text-anchor="start" x="609" y="-1521.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">A scheme whose conditions pass applies its</text>
<text xml:space="preserve" text-anchor="start" x="683" y="-1503.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions to the network</text>
<text xml:space="preserve" text-anchor="start" x="634" y="-1485.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">and the loadflow re&#45;runs, so the next</text>
</g>
<!-- ppcascade -->
<g id="node5" class="node">
<title>ppcascade</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="917.02,-1293.4 596.98,-1293.4 596.98,-1113.4 917.02,-1113.4 917.02,-1293.4"/>
<text xml:space="preserve" text-anchor="start" x="670.5" y="-1262.4" font-family="Arial" font-size="20.00" fill="#f8fafc">Cascade simulation</text>
<text xml:space="preserve" text-anchor="start" x="634.5" y="-1234.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Iterative follow&#45;on outage simulation:</text>
<text xml:space="preserve" text-anchor="start" x="683" y="-1216.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">overload and distance</text>
<text xml:space="preserve" text-anchor="start" x="629.5" y="-1198.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">protection detection, outage grouping,</text>
<text xml:space="preserve" text-anchor="start" x="668" y="-1180.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">re&#45;solve, repeat. Produces</text>
<text xml:space="preserve" text-anchor="start" x="648.5" y="-1162.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">an event log rather than a single</text>
</g>
<!-- dispatcher -->
<g id="node6" class="node">
<title>dispatcher</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1385.56,-2582.1 1046.44,-2582.1 1046.44,-2402.1 1385.56,-2402.1 1385.56,-2582.1"/>
<text xml:space="preserve" text-anchor="start" x="1109.5" y="-2551.1" font-family="Arial" font-size="20.00" fill="#f8fafc">get_ac_loadflow_results</text>
<text xml:space="preserve" text-anchor="start" x="1078.5" y="-2523.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">The single entry point. Dispatches on the</text>
<text xml:space="preserve" text-anchor="start" x="1149" y="-2505.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">*type of the network</text>
<text xml:space="preserve" text-anchor="start" x="1066.5" y="-2487.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">object* &#45;&#45; a pandapowerNet goes one way, a</text>
<text xml:space="preserve" text-anchor="start" x="1134.5" y="-2469.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">PyPowSyBl Network the</text>
<text xml:space="preserve" text-anchor="start" x="1090" y="-2451.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">other &#45;&#45; and raises if it is neither. Both</text>
</g>
<!-- branchres -->
<g id="node7" class="node">
<title>branchres</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="479.56,-902 140.44,-902 140.44,-722 479.56,-722 479.56,-902"/>
<text xml:space="preserve" text-anchor="start" x="245.5" y="-844" font-family="Arial" font-size="20.00" fill="#f8fafc">branch_results</text>
<text xml:space="preserve" text-anchor="start" x="160.5" y="-816" font-family="Arial" font-size="15.00" fill="#c2f0c2">Flows and loading per monitored branch, per</text>
<text xml:space="preserve" text-anchor="start" x="222" y="-798" font-family="Arial" font-size="15.00" fill="#c2f0c2">contingency and timestep.</text>
</g>
<!-- noderes -->
<g id="node8" class="node">
<title>noderes</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="924.06,-902 589.94,-902 589.94,-722 924.06,-722 924.06,-902"/>
<text xml:space="preserve" text-anchor="start" x="700.5" y="-844" font-family="Arial" font-size="20.00" fill="#f8fafc">node_results</text>
<text xml:space="preserve" text-anchor="start" x="610" y="-816" font-family="Arial" font-size="15.00" fill="#c2f0c2">Voltage magnitude and angle per monitored</text>
<text xml:space="preserve" text-anchor="start" x="738.5" y="-798" font-family="Arial" font-size="15.00" fill="#c2f0c2">node.</text>
</g>
<!-- regres -->
<g id="node9" class="node">
<title>regres</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1464.02,-902 1143.98,-902 1143.98,-722 1464.02,-722 1464.02,-902"/>
<text xml:space="preserve" text-anchor="start" x="1185.5" y="-844" font-family="Arial" font-size="20.00" fill="#f8fafc">regulating_element_results</text>
<text xml:space="preserve" text-anchor="start" x="1168.5" y="-816" font-family="Arial" font-size="15.00" fill="#c2f0c2">Tap positions and setpoints of regulating</text>
<text xml:space="preserve" text-anchor="start" x="1271.5" y="-798" font-family="Arial" font-size="15.00" fill="#c2f0c2">elements.</text>
</g>
<!-- vadiffres -->
<g id="node10" class="node">
<title>vadiffres</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="468.06,-601 119.94,-601 119.94,-421 468.06,-421 468.06,-601"/>
<text xml:space="preserve" text-anchor="start" x="230" y="-570" font-family="Arial" font-size="20.00" fill="#f8fafc">va_diff_results</text>
<text xml:space="preserve" text-anchor="start" x="144.5" y="-542" font-family="Arial" font-size="15.00" fill="#c2f0c2">Voltage angle differences across the ends of</text>
<text xml:space="preserve" text-anchor="start" x="216" y="-524" font-family="Arial" font-size="15.00" fill="#c2f0c2">an outaged branch and</text>
<text xml:space="preserve" text-anchor="start" x="140" y="-506" font-family="Arial" font-size="15.00" fill="#c2f0c2">across open switches. What tells you whether</text>
<text xml:space="preserve" text-anchor="start" x="225" y="-488" font-family="Arial" font-size="15.00" fill="#c2f0c2">a split can be closed</text>
<text xml:space="preserve" text-anchor="start" x="273.5" y="-470" font-family="Arial" font-size="15.00" fill="#c2f0c2">again.</text>
</g>
<!-- convergedres -->
<g id="node11" class="node">
<title>convergedres</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="898.02,-601 577.98,-601 577.98,-421 898.02,-421 898.02,-601"/>
<text xml:space="preserve" text-anchor="start" x="691.5" y="-543" font-family="Arial" font-size="20.00" fill="#f8fafc">converged</text>
<text xml:space="preserve" text-anchor="start" x="600" y="-515" font-family="Arial" font-size="15.00" fill="#c2f0c2">Convergence status per contingency and</text>
<text xml:space="preserve" text-anchor="start" x="602" y="-497" font-family="Arial" font-size="15.00" fill="#c2f0c2">timestep. The index of what actually ran.</text>
</g>
<!-- switchres -->
<g id="node12" class="node">
<title>switchres</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1464.06,-601 1117.94,-601 1117.94,-421 1464.06,-421 1464.06,-601"/>
<text xml:space="preserve" text-anchor="start" x="1228.5" y="-552" font-family="Arial" font-size="20.00" fill="#ffe0c2">switch_results</text>
<text xml:space="preserve" text-anchor="start" x="1162.5" y="-524" font-family="Arial" font-size="15.00" fill="#f9b27c">Power through each monitored switch,</text>
<text xml:space="preserve" text-anchor="start" x="1138" y="-506" font-family="Arial" font-size="15.00" fill="#f9b27c">aggregated from everything connected to one</text>
<text xml:space="preserve" text-anchor="start" x="1275" y="-488" font-family="Arial" font-size="15.00" fill="#f9b27c">side.</text>
</g>
<!-- connectivityres -->
<g id="node13" class="node">
<title>connectivityres</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="464.06,-300 123.94,-300 123.94,-120 464.06,-120 464.06,-300"/>
<text xml:space="preserve" text-anchor="start" x="212.5" y="-242" font-family="Arial" font-size="20.00" fill="#ffe0c2">connectivity_result</text>
<text xml:space="preserve" text-anchor="start" x="144" y="-214" font-family="Arial" font-size="15.00" fill="#f9b27c">Which elements each contingency takes out.</text>
<text xml:space="preserve" text-anchor="start" x="191.5" y="-196" font-family="Arial" font-size="15.00" fill="#f9b27c">Populated by outage grouping.</text>
</g>
<!-- sppsres -->
<g id="node14" class="node">
<title>sppsres</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="934.02,-300 613.98,-300 613.98,-120 934.02,-120 934.02,-300"/>
<text xml:space="preserve" text-anchor="start" x="718" y="-233" font-family="Arial" font-size="20.00" fill="#ffe0c2">spps_results</text>
<text xml:space="preserve" text-anchor="start" x="668.5" y="-205" font-family="Arial" font-size="15.00" fill="#f9b27c">Per&#45;case SpPS run summaries.</text>
</g>
<!-- cascaderes -->
<g id="node15" class="node">
<title>cascaderes</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1463.56,-300 1084.44,-300 1084.44,-120 1463.56,-120 1463.56,-300"/>
<text xml:space="preserve" text-anchor="start" x="1202" y="-242" font-family="Arial" font-size="20.00" fill="#ffe0c2">cascade_results</text>
<text xml:space="preserve" text-anchor="start" x="1104.5" y="-214" font-family="Arial" font-size="15.00" fill="#f9b27c">One row per cascade event. Empty when cascade</text>
<text xml:space="preserve" text-anchor="start" x="1221" y="-196" font-family="Arial" font-size="15.00" fill="#f9b27c">screening is off.</text>
</g>
<!-- securityanalysis -->
<g id="node16" class="node">
<title>securityanalysis</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="427.56,-2582.1 80.44,-2582.1 80.44,-2402.1 427.56,-2402.1 427.56,-2582.1"/>
<text xml:space="preserve" text-anchor="start" x="100.5" y="-2560.9" font-family="Arial" font-size="20.00" fill="#f8fafc">import_grid_with_security_analysis</text>
<text xml:space="preserve" text-anchor="start" x="221" y="-2532.9" font-family="Arial" font-size="13.00" fill="#cbd5e1">PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="123.5" y="-2513.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">The only AC N&#45;1 of the import, with the</text>
<text xml:space="preserve" text-anchor="start" x="187" y="-2495.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">converging loadflow</text>
<text xml:space="preserve" text-anchor="start" x="109" y="-2477.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">parameters saved next to the grid. Threads</text>
<text xml:space="preserve" text-anchor="start" x="167" y="-2459.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">and batch size come from</text>
<text xml:space="preserve" text-anchor="start" x="163.5" y="-2441.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">n_contingency_worker and</text>
</g>
<!-- worstk -->
<g id="node17" class="node">
<title>worstk</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="999.56,-2915 660.44,-2915 660.44,-2735 999.56,-2735 999.56,-2915"/>
<text xml:space="preserve" text-anchor="start" x="766" y="-2893.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Worst&#45;k epoch</text>
<text xml:space="preserve" text-anchor="start" x="797" y="-2865.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="680.5" y="-2846.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Reruns only the handful of contingencies the</text>
<text xml:space="preserve" text-anchor="start" x="741.5" y="-2828.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">DC stage flagged as worst</text>
<text xml:space="preserve" text-anchor="start" x="691.5" y="-2810.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">for this topology. A candidate that already</text>
<text xml:space="preserve" text-anchor="start" x="749" y="-2792.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">fails there, or converges</text>
<text xml:space="preserve" text-anchor="start" x="720.5" y="-2774.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">poorly, is rejected without the full</text>
</g>
<!-- remainingca -->
<g id="node18" class="node">
<title>remainingca</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1495.56,-2915 1170.44,-2915 1170.44,-2735 1495.56,-2735 1495.56,-2915"/>
<text xml:space="preserve" text-anchor="start" x="1222.5" y="-2875.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Remaining contingencies</text>
<text xml:space="preserve" text-anchor="start" x="1210.5" y="-2847.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">PyPowSyBl security analysis, multiprocess</text>
<text xml:space="preserve" text-anchor="start" x="1190.5" y="-2828.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Full AC N&#45;1 on the survivors, batched over</text>
<text xml:space="preserve" text-anchor="start" x="1237.5" y="-2810.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">runner processes. Hundreds</text>
<text xml:space="preserve" text-anchor="start" x="1204" y="-2792.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">of contingencies rather than a handful.</text>
</g>
<!-- loadflowstore -->
<g id="node19" class="node">
<title>loadflowstore</title>
<path fill="#64748b" stroke="#475569" stroke-width="2" d="M1978.06,-2574C1978.06,-2584.04 1896.9,-2592.2 1797,-2592.2 1697.1,-2592.2 1615.94,-2584.04 1615.94,-2574 1615.94,-2574 1615.94,-2410.2 1615.94,-2410.2 1615.94,-2400.16 1697.1,-2392 1797,-2392 1896.9,-2392 1978.06,-2400.16 1978.06,-2410.2 1978.06,-2410.2 1978.06,-2574 1978.06,-2574"/>
<path fill="none" stroke="#475569" stroke-width="2" d="M1978.06,-2574C1978.06,-2563.96 1896.9,-2555.8 1797,-2555.8 1697.1,-2555.8 1615.94,-2563.96 1615.94,-2574"/>
<text xml:space="preserve" text-anchor="start" x="1706" y="-2560.9" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="1731" y="-2532.9" font-family="Arial" font-size="13.00" fill="#cbd5e1">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1691" y="-2513.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="1636" y="-2495.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="1652.5" y="-2477.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="1646.5" y="-2459.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="1711.5" y="-2441.3" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology it evaluates gets</text>
</g>
<!-- pwlimitcache&#45;&gt;branchres -->
<g id="edge3" class="edge">
<title>pwlimitcache&#45;&gt;branchres</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1165.27,-2049.8C1144.83,-1990.64 1126,-1917.74 1126,-1850 1126,-1850 1126,-1850 1126,-1202.4 1126,-1108.98 1083.64,-1076.97 1001,-1033.4 955.17,-1009.24 583.01,-1035.89 535,-1016.4 506.94,-1005 480.13,-988.46 455.42,-969.66"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="457.04,-967.59 449.51,-965.06 453.82,-971.74 457.04,-967.59"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="954,-1202.4 954,-1225.2 1126,-1225.2 1126,-1202.4 954,-1202.4"/>
<text xml:space="preserve" text-anchor="start" x="957" y="-1222.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">fills the five common tables</text>
</g>
<!-- ppoutagegrouping&#45;&gt;ppslack -->
<g id="edge15" class="edge">
<title>ppoutagegrouping&#45;&gt;ppslack</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M757,-2081.87C757,-2040.67 757,-1991.56 757,-1949.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="759.63,-1949.36 757,-1941.86 754.38,-1949.36 759.63,-1949.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="757,-1999 757,-2021.8 902,-2021.8 902,-1999 757,-1999"/>
<text xml:space="preserve" text-anchor="start" x="760" y="-2018.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">grouped contingencies</text>
</g>
<!-- ppslack&#45;&gt;ppspps -->
<g id="edge16" class="edge">
<title>ppslack&#45;&gt;ppspps</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M757,-1759.07C757,-1717.87 757,-1668.76 757,-1626.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="759.63,-1626.56 757,-1619.06 754.38,-1626.56 759.63,-1626.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="757,-1676.2 757,-1699 860,-1699 860,-1676.2 757,-1676.2"/>
<text xml:space="preserve" text-anchor="start" x="760" y="-1696" font-family="Arial" font-size="14.00" fill="#c9c9c9">solvable islands</text>
</g>
<!-- ppspps&#45;&gt;ppcascade -->
<g id="edge18" class="edge">
<title>ppspps&#45;&gt;ppcascade</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M757,-1436.27C757,-1395.07 757,-1345.96 757,-1303.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="759.63,-1303.76 757,-1296.26 754.38,-1303.76 759.63,-1303.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="757,-1353.4 757,-1376.2 876,-1376.2 876,-1353.4 757,-1353.4"/>
<text xml:space="preserve" text-anchor="start" x="760" y="-1373.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">post&#45;scheme state</text>
</g>
<!-- ppcascade&#45;&gt;branchres -->
<g id="edge19" class="edge">
<title>ppcascade&#45;&gt;branchres</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M601.52,-1073.4C579.08,-1054.42 556.37,-1035 535,-1016.4 517.82,-1001.45 500.02,-985.74 482.31,-969.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="484.33,-968.25 476.98,-965.21 480.83,-972.16 484.33,-968.25"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="420.95,-1016.36 420.95,-1039.16 534.95,-1039.16 534.95,-1016.36 420.95,-1016.36"/>
<text xml:space="preserve" text-anchor="start" x="423.95" y="-1036.16" font-family="Arial" font-size="14.00" fill="#c9c9c9">fills all nine tables</text>
</g>
<!-- dispatcher&#45;&gt;pwlimitcache -->
<g id="edge13" class="edge">
<title>dispatcher&#45;&gt;pwlimitcache</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1216,-2402.44C1216,-2378.49 1216,-2351.87 1216,-2325.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1218.63,-2325.53 1216,-2318.03 1213.38,-2325.53 1218.63,-2325.53"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1073,-2358.78 1073,-2381.58 1216,-2381.58 1216,-2358.78 1073,-2358.78"/>
<text xml:space="preserve" text-anchor="start" x="1076" y="-2378.58" font-family="Arial" font-size="14.00" fill="#c9c9c9">if PyPowSyBl Network</text>
</g>
<!-- dispatcher&#45;&gt;ppoutagegrouping -->
<g id="edge14" class="edge">
<title>dispatcher&#45;&gt;ppoutagegrouping</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1087.84,-2402.22C1054.79,-2379.31 1018.24,-2353.96 981.84,-2328.72"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="983.74,-2326.84 976.08,-2324.72 980.74,-2331.15 983.74,-2326.84"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="916.66,-2362.57 916.66,-2385.37 1030.66,-2385.37 1030.66,-2362.57 916.66,-2362.57"/>
<text xml:space="preserve" text-anchor="start" x="919.66" y="-2382.37" font-family="Arial" font-size="14.00" fill="#c9c9c9">if pandapowerNet</text>
</g>
<!-- branchres&#45;&gt;vadiffres -->
<!-- noderes&#45;&gt;regres -->
<!-- regres&#45;&gt;convergedres -->
<!-- vadiffres&#45;&gt;connectivityres -->
<!-- convergedres&#45;&gt;switchres -->
<!-- cascaderes&#45;&gt;loadflowstore -->
<g id="edge7" class="edge">
<title>cascaderes&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1504,-276.39C1610.14,-322.03 1713,-395.73 1713,-510 1713,-2172.8 1713,-2172.8 1713,-2172.8 1713,-2243.72 1732.74,-2320.68 1753.08,-2381.15"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1750.59,-2381.98 1755.5,-2388.23 1755.55,-2380.28 1750.59,-2381.98"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1605,-1471.32 1605,-1494.12 1713,-1494.12 1713,-1471.32 1605,-1471.32"/>
<text xml:space="preserve" text-anchor="start" x="1608" y="-1491.12" font-family="Arial" font-size="14.00" fill="#c9c9c9">persisted per job</text>
</g>
<!-- securityanalysis&#45;&gt;pwlimitcache -->
<g id="edge8" class="edge">
<title>securityanalysis&#45;&gt;pwlimitcache</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M365.23,-2402.19C407.2,-2373.83 456.94,-2345.95 507,-2331 559.6,-2315.29 948.77,-2339.9 1001,-2323 1004.97,-2321.72 1008.93,-2320.33 1012.88,-2318.85"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1013.83,-2321.3 1019.85,-2316.11 1011.91,-2316.41 1013.83,-2321.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="605.48,-2326.36 605.48,-2349.16 694.48,-2349.16 694.48,-2326.36 605.48,-2326.36"/>
<text xml:space="preserve" text-anchor="start" x="608.48" y="-2346.16" font-family="Arial" font-size="14.00" fill="#c9c9c9">base grid N&#45;1</text>
</g>
<!-- worstk&#45;&gt;dispatcher -->
<g id="edge10" class="edge">
<title>worstk&#45;&gt;dispatcher</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M933.69,-2735.11C986.48,-2689.86 1050.7,-2634.81 1104.54,-2588.65"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1106.12,-2590.75 1110.11,-2583.88 1102.7,-2586.77 1106.12,-2590.75"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1023,-2652.2 1023,-2675 1162,-2675 1162,-2652.2 1023,-2652.2"/>
<text xml:space="preserve" text-anchor="start" x="1026" y="-2672" font-family="Arial" font-size="14.00" fill="#c9c9c9">worst&#45;k contingencies</text>
</g>
<!-- worstk&#45;&gt;remainingca -->
<g id="edge9" class="edge">
<title>worstk&#45;&gt;remainingca</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M999.24,-2825C1051.13,-2825 1108.27,-2825 1160.38,-2825"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1160.21,-2827.63 1167.71,-2825 1160.21,-2822.38 1160.21,-2827.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1054.5,-2828 1054.5,-2850.8 1115.5,-2850.8 1115.5,-2828 1054.5,-2828"/>
<text xml:space="preserve" text-anchor="start" x="1057.5" y="-2847.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">survivors</text>
</g>
<!-- remainingca&#45;&gt;dispatcher -->
<g id="edge11" class="edge">
<title>remainingca&#45;&gt;dispatcher</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1301.65,-2735.32C1286.01,-2691.09 1267.04,-2637.46 1250.94,-2591.91"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1253.42,-2591.07 1248.45,-2584.87 1248.47,-2592.82 1253.42,-2591.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1278.01,-2652.2 1278.01,-2675 1327.01,-2675 1327.01,-2652.2 1278.01,-2652.2"/>
<text xml:space="preserve" text-anchor="start" x="1281.01" y="-2672" font-family="Arial" font-size="14.00" fill="#c9c9c9">full N&#45;1</text>
</g>
<!-- remainingca&#45;&gt;loadflowstore -->
<g id="edge12" class="edge">
<title>remainingca&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1536,-2734.65C1568.65,-2716.8 1601.16,-2696.77 1630,-2675 1658.34,-2653.61 1685.82,-2627.02 1710.04,-2600.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1711.95,-2602.64 1715.07,-2595.33 1708.07,-2599.1 1711.95,-2602.64"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1421.32,-2672.47 1421.32,-2712.07 1633.32,-2712.07 1633.32,-2672.47 1421.32,-2672.47"/>
<text xml:space="preserve" text-anchor="start" x="1424.32" y="-2709.07" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="1424.32" y="-2692.27" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- loadflowstore&#45;&gt;worstk -->
<g id="edge17" class="edge">
<title>loadflowstore&#45;&gt;worstk</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1621.58,-2585.53C1577.74,-2606.39 1530.29,-2627.12 1485,-2643.3 1428.59,-2663.45 1413.13,-2665.43 1354,-2675 1252.27,-2691.46 1220.28,-2669.73 1125.04,-2692.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1124.58,-2689.9 1117.94,-2694.27 1125.85,-2695 1124.58,-2689.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1187.99,-2675 1187.99,-2697.8 1353.99,-2697.8 1353.99,-2675 1187.99,-2675"/>
<text xml:space="preserve" text-anchor="start" x="1190.99" y="-2694.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial loadflow as baseline</text>
</g>
</g>
</svg>`;case`assetTopology`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1580pt" height="3686pt"
 viewBox="0.00 0.00 1580.00 3686.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 3671.25)">
<g id="clust1" class="cluster">
<title>cluster_topologymodel</title>
<polygon fill="#3e4651" stroke="#2d333d" points="90.96,-2825 90.96,-3407.2 1104.96,-3407.2 1104.96,-2825 90.96,-2825"/>
<text xml:space="preserve" text-anchor="start" x="98.96" y="-3393.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">GET_MASTER_ASSET_TOPOLOGY_ARTIFACT</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_materialize</title>
<polygon fill="#3e4651" stroke="#2d333d" points="338.96,-1891 338.96,-2172.2 1541.96,-2172.2 1541.96,-1891 338.96,-1891"/>
<text xml:space="preserve" text-anchor="start" x="346.96" y="-2158.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">GET_RUNTIME_ASSET_TOPOLOGY</text>
</g>
<g id="clust4" class="cluster">
<title>cluster_simplify</title>
<polygon fill="#3e4651" stroke="#2d333d" points="177.96,-1279.8 177.96,-1561 1125.96,-1561 1125.96,-1279.8 177.96,-1279.8"/>
<text xml:space="preserve" text-anchor="start" x="185.96" y="-1547.55" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">SIMPLIFY_ASSET_TOPOLOGY</text>
</g>
<!-- busbreakerextract -->
<g id="node1" class="node">
<title>busbreakerextract</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="537.01,-3346 130.9,-3346 130.9,-3166 537.01,-3166 537.01,-3346"/>
<text xml:space="preserve" text-anchor="start" x="150.96" y="-3288" font-family="Arial" font-size="20.00" fill="#f8fafc">get_bus_breaker_master_asset_topology</text>
<text xml:space="preserve" text-anchor="start" x="168.46" y="-3260" font-family="Arial" font-size="15.00" fill="#cbd5e1">UCTE. Bus&#45;breaker source, so bays and busbars</text>
<text xml:space="preserve" text-anchor="start" x="212.46" y="-3242" font-family="Arial" font-size="15.00" fill="#cbd5e1">have to be inferred rather than read.</text>
</g>
<!-- nodebreakerextract -->
<g id="node2" class="node">
<title>nodebreakerextract</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1065.01,-3346 646.9,-3346 646.9,-3166 1065.01,-3166 1065.01,-3346"/>
<text xml:space="preserve" text-anchor="start" x="666.96" y="-3297" font-family="Arial" font-size="20.00" fill="#f8fafc">get_node_breaker_master_asset_topology</text>
<text xml:space="preserve" text-anchor="start" x="708.46" y="-3269" font-family="Arial" font-size="15.00" fill="#cbd5e1">CGMES. Node&#45;breaker source, walked as a</text>
<text xml:space="preserve" text-anchor="start" x="719.46" y="-3251" font-family="Arial" font-size="15.00" fill="#cbd5e1">station graph &#45;&#45; the richest input, and the</text>
<text xml:space="preserve" text-anchor="start" x="754.46" y="-3233" font-family="Arial" font-size="15.00" fill="#cbd5e1">one the model is shaped after.</text>
</g>
<!-- ppextract -->
<g id="node3" class="node">
<title>ppextract</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="599.01,-3045 182.9,-3045 182.9,-2865 599.01,-2865 599.01,-3045"/>
<text xml:space="preserve" text-anchor="start" x="202.96" y="-2987" font-family="Arial" font-size="20.00" fill="#f8fafc">get_master_asset_topology_from_network</text>
<text xml:space="preserve" text-anchor="start" x="228.96" y="-2959" font-family="Arial" font-size="15.00" fill="#cbd5e1">pandapower nets, read through the pandapower</text>
<text xml:space="preserve" text-anchor="start" x="315.96" y="-2941" font-family="Arial" font-size="15.00" fill="#cbd5e1">switch and bus tables.</text>
</g>
<!-- pwmaterialize -->
<g id="node4" class="node">
<title>pwmaterialize</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="892.51,-2111 379.4,-2111 379.4,-1931 892.51,-1931 892.51,-2111"/>
<text xml:space="preserve" text-anchor="start" x="399.46" y="-2053" font-family="Arial" font-size="20.00" fill="#f8fafc">materialize_runtime_bus_groups_from_network_state</text>
<text xml:space="preserve" text-anchor="start" x="508.96" y="-2025" font-family="Arial" font-size="15.00" fill="#cbd5e1">Reads switch positions straight off the</text>
<text xml:space="preserve" text-anchor="start" x="498.46" y="-2007" font-family="Arial" font-size="15.00" fill="#cbd5e1">node&#45;breaker network, station by station.</text>
</g>
<!-- compactmaterialize -->
<g id="node5" class="node">
<title>compactmaterialize</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1501.51,-2111 1002.4,-2111 1002.4,-1931 1501.51,-1931 1501.51,-2111"/>
<text xml:space="preserve" text-anchor="start" x="1022.46" y="-2080" font-family="Arial" font-size="20.00" fill="#f8fafc">materialize_runtime_bus_group_from_runtime_state</text>
<text xml:space="preserve" text-anchor="start" x="1110.96" y="-2052" font-family="Arial" font-size="15.00" fill="#cbd5e1">The backend&#45;neutral half: a canonical bus</text>
<text xml:space="preserve" text-anchor="start" x="1179.46" y="-2034" font-family="Arial" font-size="15.00" fill="#cbd5e1">group plus a compact</text>
<text xml:space="preserve" text-anchor="start" x="1095.46" y="-2016" font-family="Arial" font-size="15.00" fill="#cbd5e1">RuntimeSwitchingState overlay in, one runtime</text>
<text xml:space="preserve" text-anchor="start" x="1188.46" y="-1998" font-family="Arial" font-size="15.00" fill="#cbd5e1">bus group out. The</text>
<text xml:space="preserve" text-anchor="start" x="1104.46" y="-1980" font-family="Arial" font-size="15.00" fill="#cbd5e1">pandapower path runs through here, and so</text>
</g>
<!-- prepareseparation -->
<g id="node6" class="node">
<title>prepareseparation</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="575.51,-1499.8 218.4,-1499.8 218.4,-1319.8 575.51,-1319.8 575.51,-1499.8"/>
<text xml:space="preserve" text-anchor="start" x="275.46" y="-1468.8" font-family="Arial" font-size="20.00" fill="#f8fafc">prepare_for_separation_set</text>
<text xml:space="preserve" text-anchor="start" x="238.46" y="-1440.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where the reduction actually happens, one bus</text>
<text xml:space="preserve" text-anchor="start" x="344.46" y="-1422.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">group at a time:</text>
<text xml:space="preserve" text-anchor="start" x="252.46" y="-1404.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">order assets to the solver index order, drop</text>
<text xml:space="preserve" text-anchor="start" x="326.96" y="-1386.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">out&#45;of&#45;service assets</text>
<text xml:space="preserve" text-anchor="start" x="246.46" y="-1368.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">and disconnected busbars, remove duplicate</text>
</g>
<!-- bbsimplify -->
<g id="node7" class="node">
<title>bbsimplify</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1086.01,-1499.8 685.9,-1499.8 685.9,-1319.8 1086.01,-1319.8 1086.01,-1499.8"/>
<text xml:space="preserve" text-anchor="start" x="705.96" y="-1468.8" font-family="Arial" font-size="20.00" fill="#f8fafc">simplify_asset_topology_for_bb_outages</text>
<text xml:space="preserve" text-anchor="start" x="748.96" y="-1440.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">The second reduction, for busbar&#45;outage</text>
<text xml:space="preserve" text-anchor="start" x="807.46" y="-1422.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">preprocessing, run with</text>
<text xml:space="preserve" text-anchor="start" x="747.46" y="-1404.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">couplers forced closed. Yields a separate</text>
<text xml:space="preserve" text-anchor="start" x="823.96" y="-1386.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">simplified topology</text>
<text xml:space="preserve" text-anchor="start" x="772.46" y="-1368.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">rather than replacing the first one.</text>
</g>
<!-- gridsnapshot -->
<g id="node8" class="node">
<title>gridsnapshot</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="949.01,-3656.2 614.9,-3656.2 614.9,-3476.2 949.01,-3476.2 949.01,-3656.2"/>
<text xml:space="preserve" text-anchor="start" x="693.96" y="-3598.2" font-family="Arial" font-size="20.00" fill="#f8fafc">grid.xiidm / grid.json</text>
<text xml:space="preserve" text-anchor="start" x="634.96" y="-3570.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The normalized backend grid, written by the</text>
<text xml:space="preserve" text-anchor="start" x="752.46" y="-3552.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">importer.</text>
</g>
<!-- master -->
<g id="node9" class="node">
<title>master</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="596.98,-2744 276.94,-2744 276.94,-2564 596.98,-2564 596.98,-2744"/>
<text xml:space="preserve" text-anchor="start" x="331.46" y="-2713" font-family="Arial" font-size="20.00" fill="#eef2ff">1. MasterAssetTopology</text>
<text xml:space="preserve" text-anchor="start" x="299.46" y="-2685" font-family="Arial" font-size="15.00" fill="#c7d2fe">Structure, no state. Bus groups with their</text>
<text xml:space="preserve" text-anchor="start" x="354.46" y="-2667" font-family="Arial" font-size="15.00" fill="#c7d2fe">busbars, couplers, asset</text>
<text xml:space="preserve" text-anchor="start" x="303.96" y="-2649" font-family="Arial" font-size="15.00" fill="#c7d2fe">bays and circuit groups, the branch and</text>
<text xml:space="preserve" text-anchor="start" x="368.46" y="-2631" font-family="Arial" font-size="15.00" fill="#c7d2fe">injection assets they</text>
<text xml:space="preserve" text-anchor="start" x="319.96" y="-2613" font-family="Arial" font-size="15.00" fill="#c7d2fe">connect, and branch_connectivity /</text>
</g>
<!-- assettopomaster -->
<g id="node10" class="node">
<title>assettopomaster</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="742.51,-2421.2 269.4,-2421.2 269.4,-2241.2 742.51,-2241.2 742.51,-2421.2"/>
<text xml:space="preserve" text-anchor="start" x="289.46" y="-2390.2" font-family="Arial" font-size="20.00" fill="#f8fafc">initial_topology/asset_topology_master_data.json</text>
<text xml:space="preserve" text-anchor="start" x="363.46" y="-2362.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">A serialized MasterAssetTopology, and the</text>
<text xml:space="preserve" text-anchor="start" x="433.46" y="-2344.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">only form of the asset</text>
<text xml:space="preserve" text-anchor="start" x="364.96" y="-2326.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology that gets a file of its own. Written</text>
<text xml:space="preserve" text-anchor="start" x="418.46" y="-2308.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">by the importer, read back</text>
<text xml:space="preserve" text-anchor="start" x="353.96" y="-2290.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">at the start of DC preprocessing. The runtime</text>
</g>
<!-- runtime -->
<g id="node11" class="node">
<title>runtime</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1423.51,-1810 1080.4,-1810 1080.4,-1630 1423.51,-1630 1423.51,-1810"/>
<text xml:space="preserve" text-anchor="start" x="1139.46" y="-1779" font-family="Arial" font-size="20.00" fill="#f8fafc">2. RuntimeAssetTopology</text>
<text xml:space="preserve" text-anchor="start" x="1100.46" y="-1751" font-family="Arial" font-size="15.00" fill="#c2f0c2">The same structure, materialized against one</text>
<text xml:space="preserve" text-anchor="start" x="1192.96" y="-1733" font-family="Arial" font-size="15.00" fill="#c2f0c2">grid state. What it</text>
<text xml:space="preserve" text-anchor="start" x="1138.96" y="-1715" font-family="Arial" font-size="15.00" fill="#c2f0c2">adds is a second pair of matrices:</text>
<text xml:space="preserve" text-anchor="start" x="1157.96" y="-1697" font-family="Arial" font-size="15.00" fill="#c2f0c2">branch_switching_table and</text>
<text xml:space="preserve" text-anchor="start" x="1103.96" y="-1679" font-family="Arial" font-size="15.00" fill="#c2f0c2">injection_switching_table say what is closed</text>
</g>
<!-- storedactionset -->
<g id="node12" class="node">
<title>storedactionset</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1143.51,-180 806.4,-180 806.4,0 1143.51,0 1143.51,-180"/>
<text xml:space="preserve" text-anchor="start" x="900.46" y="-149" font-family="Arial" font-size="20.00" fill="#f8fafc">Stored action set</text>
<text xml:space="preserve" text-anchor="start" x="835.46" y="-121" font-family="Arial" font-size="15.00" fill="#cbd5e1">The action set in physical terms, keyed to</text>
<text xml:space="preserve" text-anchor="start" x="901.46" y="-103" font-family="Arial" font-size="15.00" fill="#cbd5e1">the asset topology, as</text>
<text xml:space="preserve" text-anchor="start" x="826.46" y="-85" font-family="Arial" font-size="15.00" fill="#cbd5e1">opposed to the electrical index form the JAX</text>
<text xml:space="preserve" text-anchor="start" x="918.46" y="-67" font-family="Arial" font-size="15.00" fill="#cbd5e1">solver uses. Two</text>
<text xml:space="preserve" text-anchor="start" x="827.46" y="-49" font-family="Arial" font-size="15.00" fill="#cbd5e1">representations of one thing: the JAX one is</text>
</g>
<!-- simplified -->
<g id="node13" class="node">
<title>simplified</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1075.51,-1198.8 752.4,-1198.8 752.4,-1018.8 1075.51,-1018.8 1075.51,-1198.8"/>
<text xml:space="preserve" text-anchor="start" x="795.46" y="-1167.8" font-family="Arial" font-size="20.00" fill="#eff6ff">3. SimplifiedAssetTopology</text>
<text xml:space="preserve" text-anchor="start" x="774.46" y="-1139.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">The runtime form reduced to what the DC</text>
<text xml:space="preserve" text-anchor="start" x="826.46" y="-1121.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">solver can search &#45;&#45; and a</text>
<text xml:space="preserve" text-anchor="start" x="772.46" y="-1103.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">*subclass* of it, so the reduction is carried</text>
<text xml:space="preserve" text-anchor="start" x="844.46" y="-1085.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">in the type system: a</text>
<text xml:space="preserve" text-anchor="start" x="773.96" y="-1067.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">function that needs a simplified bus group</text>
</g>
<!-- electricalactions -->
<g id="node14" class="node">
<title>electricalactions</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="791.51,-859.2 462.4,-859.2 462.4,-679.2 791.51,-679.2 791.51,-859.2"/>
<text xml:space="preserve" text-anchor="start" x="506.96" y="-828.2" font-family="Arial" font-size="20.00" fill="#f8fafc">compute_electrical_actions</text>
<text xml:space="preserve" text-anchor="start" x="482.46" y="-800.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Stage one of action set enumeration: every</text>
<text xml:space="preserve" text-anchor="start" x="565.96" y="-782.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">electrically distinct</text>
<text xml:space="preserve" text-anchor="start" x="502.46" y="-764.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">two&#45;node split of a station, filtered for</text>
<text xml:space="preserve" text-anchor="start" x="582.46" y="-746.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">islanding and</text>
<text xml:space="preserve" text-anchor="start" x="491.96" y="-728.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">connectivity, clipped if a station exceeds</text>
</g>
<!-- stationrealisations -->
<g id="node15" class="node">
<title>stationrealisations</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="792.51,-502.8 455.4,-502.8 455.4,-322.8 792.51,-322.8 792.51,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="485.96" y="-471.8" font-family="Arial" font-size="20.00" fill="#f8fafc">enumerate_station_realisations</text>
<text xml:space="preserve" text-anchor="start" x="481.46" y="-443.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">Stage two: map each electrical split onto a</text>
<text xml:space="preserve" text-anchor="start" x="543.96" y="-425.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">reachable node&#45;breaker</text>
<text xml:space="preserve" text-anchor="start" x="475.46" y="-407.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">realization and precompute its reassignment</text>
<text xml:space="preserve" text-anchor="start" x="557.46" y="-389.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">distance. Splits with</text>
<text xml:space="preserve" text-anchor="start" x="510.96" y="-371.8" font-family="Arial" font-size="15.00" fill="#cbd5e1">no valid realization are discarded.</text>
</g>
<!-- bboutage -->
<g id="node16" class="node">
<title>bboutage</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="696.01,-180 339.9,-180 339.9,0 696.01,0 696.01,-180"/>
<text xml:space="preserve" text-anchor="start" x="416.96" y="-122" font-family="Arial" font-size="20.00" fill="#f8fafc">preprocess_bb_outage</text>
<text xml:space="preserve" text-anchor="start" x="359.96" y="-94" font-family="Arial" font-size="15.00" fill="#cbd5e1">Optional busbar outage contingencies, used by</text>
<text xml:space="preserve" text-anchor="start" x="402.46" y="-76" font-family="Arial" font-size="15.00" fill="#cbd5e1">the do&#45;not&#45;make&#45;it&#45;worse criterion.</text>
</g>
<!-- busbreakerextract&#45;&gt;ppextract -->
<!-- ppextract&#45;&gt;master -->
<g id="edge4" class="edge">
<title>ppextract&#45;&gt;master</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M410.8,-2825C414.47,-2801.16 418.25,-2776.6 421.75,-2753.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="424.29,-2754.62 422.83,-2746.81 419.1,-2753.82 424.29,-2754.62"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="212.07,-2784.25 212.07,-2823.85 417.07,-2823.85 417.07,-2784.25 212.07,-2784.25"/>
<text xml:space="preserve" text-anchor="start" x="215.07" y="-2820.85" font-family="Arial" font-size="14.00" fill="#c9c9c9">bus groups, bays, circuit groups,</text>
<text xml:space="preserve" text-anchor="start" x="215.07" y="-2804.05" font-family="Arial" font-size="14.00" fill="#c9c9c9">possible connectivity</text>
</g>
<!-- compactmaterialize&#45;&gt;runtime -->
<g id="edge7" class="edge">
<title>compactmaterialize&#45;&gt;runtime</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1251.96,-1891C1251.96,-1867.31 1251.96,-1842.93 1251.96,-1820.28"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1254.58,-1820.34 1251.96,-1812.84 1249.33,-1820.34 1254.58,-1820.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1059.96,-1850.25 1059.96,-1873.05 1251.96,-1873.05 1251.96,-1850.25 1059.96,-1850.25"/>
<text xml:space="preserve" text-anchor="start" x="1062.96" y="-1870.05" font-family="Arial" font-size="14.00" fill="#c9c9c9">structure + what is closed now</text>
</g>
<!-- bbsimplify&#45;&gt;simplified -->
<g id="edge10" class="edge">
<title>bbsimplify&#45;&gt;simplified</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M898.04,-1279.8C900.26,-1256.11 902.54,-1231.73 904.66,-1209.08"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="907.27,-1209.34 905.36,-1201.63 902.04,-1208.85 907.27,-1209.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="670.85,-1239.05 670.85,-1261.85 901.85,-1261.85 901.85,-1239.05 670.85,-1239.05"/>
<text xml:space="preserve" text-anchor="start" x="673.85" y="-1258.85" font-family="Arial" font-size="14.00" fill="#c9c9c9">one reduced slice per electrical node</text>
</g>
<!-- gridsnapshot&#45;&gt;busbreakerextract -->
<g id="edge2" class="edge">
<title>gridsnapshot&#45;&gt;busbreakerextract</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M652.68,-3476.26C623.78,-3456.38 592.3,-3434.72 560.74,-3413.02"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="562.44,-3411 554.78,-3408.91 559.47,-3415.33 562.44,-3411"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="476.63,-3441.83 476.63,-3464.63 602.63,-3464.63 602.63,-3441.83 476.63,-3441.83"/>
<text xml:space="preserve" text-anchor="start" x="479.63" y="-3461.63" font-family="Arial" font-size="14.00" fill="#c9c9c9">normalized network</text>
</g>
<!-- gridsnapshot&#45;&gt;pwmaterialize -->
<g id="edge3" class="edge">
<title>gridsnapshot&#45;&gt;pwmaterialize</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M948.83,-3523.49C1076.55,-3480.39 1230.96,-3398.94 1230.96,-3257 1230.96,-3257 1230.96,-3257 1230.96,-2330.2 1230.96,-2321.21 1092.13,-2249 948.6,-2176.73"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="950.16,-2174.58 942.28,-2173.56 947.8,-2179.27 950.16,-2174.58"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1001.96,-3141.15 1001.96,-3163.95 1230.96,-3163.95 1230.96,-3141.15 1001.96,-3141.15"/>
<text xml:space="preserve" text-anchor="start" x="1004.96" y="-3160.95" font-family="Arial" font-size="14.00" fill="#c9c9c9">live switch, coupler and busbar state</text>
</g>
<!-- master&#45;&gt;assettopomaster -->
<g id="edge5" class="edge">
<title>master&#45;&gt;assettopomaster</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M456.09,-2564.07C464.97,-2522.78 475.55,-2473.55 484.68,-2431.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="487.2,-2431.87 486.21,-2423.99 482.07,-2430.77 487.2,-2431.87"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="473.53,-2481.2 473.53,-2504 637.53,-2504 637.53,-2481.2 473.53,-2481.2"/>
<text xml:space="preserve" text-anchor="start" x="476.53" y="-2501" font-family="Arial" font-size="14.00" fill="#c9c9c9">serialized once per import</text>
</g>
<!-- assettopomaster&#45;&gt;pwmaterialize -->
<g id="edge6" class="edge">
<title>assettopomaster&#45;&gt;pwmaterialize</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M543.38,-2241.47C551.33,-2222.64 559.94,-2202.21 568.62,-2181.65"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="570.93,-2182.92 571.43,-2174.99 566.09,-2180.88 570.93,-2182.92"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="435.92,-2207.02 435.92,-2229.82 557.92,-2229.82 557.92,-2207.02 435.92,-2207.02"/>
<text xml:space="preserve" text-anchor="start" x="438.92" y="-2226.82" font-family="Arial" font-size="14.00" fill="#c9c9c9">canonical structure</text>
</g>
<!-- runtime&#45;&gt;prepareseparation -->
<g id="edge8" class="edge">
<title>runtime&#45;&gt;prepareseparation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1080.4,-1690.75C955.65,-1666.65 784.16,-1626.08 640.39,-1565.05"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="641.67,-1562.74 633.74,-1562.19 639.59,-1567.57 641.67,-1562.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="702.27,-1637.55 702.27,-1660.35 852.27,-1660.35 852.27,-1637.55 702.27,-1637.55"/>
<text xml:space="preserve" text-anchor="start" x="705.27" y="-1657.35" font-family="Arial" font-size="14.00" fill="#c9c9c9">full physical bus groups</text>
</g>
<!-- runtime&#45;&gt;storedactionset -->
<g id="edge9" class="edge">
<title>runtime&#45;&gt;storedactionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1251.96,-1630.32C1251.96,-1568.73 1251.96,-1484.77 1251.96,-1410.8 1251.96,-1410.8 1251.96,-1410.8 1251.96,-411.8 1251.96,-320.58 1187,-242.47 1119.99,-186.18"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1122.03,-184.45 1114.57,-181.7 1118.68,-188.5 1122.03,-184.45"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1251.96,-919.2 1251.96,-958.8 1498.96,-958.8 1498.96,-919.2 1251.96,-919.2"/>
<text xml:space="preserve" text-anchor="start" x="1254.96" y="-955.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">starting_bus_groups &#45;&#45; to reach the real</text>
<text xml:space="preserve" text-anchor="start" x="1254.96" y="-939" font-family="Arial" font-size="14.00" fill="#c9c9c9">switches</text>
</g>
<!-- simplified&#45;&gt;storedactionset -->
<g id="edge14" class="edge">
<title>simplified&#45;&gt;storedactionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M916.66,-1019.01C920.19,-911.1 927.1,-723.49 936.96,-562.8 944.87,-433.81 957.4,-285.24 965.89,-189.84"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="968.48,-190.34 966.53,-182.64 963.25,-189.88 968.48,-190.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="936.96,-562.8 936.96,-619.2 1168.96,-619.2 1168.96,-562.8 936.96,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="939.96" y="-616.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">simplified_starting_bus_groups &#45;&#45; the</text>
<text xml:space="preserve" text-anchor="start" x="939.96" y="-599.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">ordering local_actions is indexed</text>
<text xml:space="preserve" text-anchor="start" x="939.96" y="-582.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">against</text>
</g>
<!-- simplified&#45;&gt;electricalactions -->
<g id="edge11" class="edge">
<title>simplified&#45;&gt;electricalactions</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M752.4,-1054.73C708.08,-1032.19 664.87,-1001.06 637.96,-958.8 621.26,-932.58 615.42,-899.92 614.68,-869.18"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="617.31,-869.36 614.64,-861.87 612.06,-869.39 617.31,-869.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="637.96,-927.6 637.96,-950.4 875.96,-950.4 875.96,-927.6 637.96,-927.6"/>
<text xml:space="preserve" text-anchor="start" x="640.96" y="-947.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">the geometry splits are enumerated in</text>
</g>
<!-- simplified&#45;&gt;stationrealisations -->
<g id="edge12" class="edge">
<title>simplified&#45;&gt;stationrealisations</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M752.64,-1101.46C586.08,-1085.28 333.87,-1031.7 211.96,-859.2 165.79,-793.87 173.65,-749.43 211.96,-679.2 262.02,-587.4 358.97,-522.57 446.25,-479.98"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="447.13,-482.47 452.75,-476.85 444.86,-477.74 447.13,-482.47"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="211.96,-757.8 211.96,-780.6 406.96,-780.6 406.96,-757.8 211.96,-757.8"/>
<text xml:space="preserve" text-anchor="start" x="214.96" y="-777.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">station to realize a split against</text>
</g>
<!-- simplified&#45;&gt;bboutage -->
<g id="edge13" class="edge">
<title>simplified&#45;&gt;bboutage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M752.47,-1099.49C609.94,-1086 400.87,-1050.74 242.96,-958.8 84.56,-866.57 57.88,-795.84 8.96,-619.2 -45.15,-423.87 164.75,-267.26 331.3,-176.72"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="332.18,-179.23 337.54,-173.36 329.69,-174.61 332.18,-179.23"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="8.96,-579.6 8.96,-602.4 230.96,-602.4 230.96,-579.6 8.96,-579.6"/>
<text xml:space="preserve" text-anchor="start" x="11.96" y="-599.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">reduced again with couplers closed</text>
</g>
<!-- electricalactions&#45;&gt;stationrealisations -->
<g id="edge15" class="edge">
<title>electricalactions&#45;&gt;stationrealisations</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M626.2,-679.23C625.78,-628.58 625.24,-565.01 624.8,-512.79"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="627.42,-513.01 624.73,-505.53 622.17,-513.05 627.42,-513.01"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="625.64,-579.6 625.64,-602.4 721.64,-602.4 721.64,-579.6 625.64,-579.6"/>
<text xml:space="preserve" text-anchor="start" x="628.64" y="-599.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">electrical splits</text>
</g>
<!-- stationrealisations&#45;&gt;storedactionset -->
<g id="edge17" class="edge">
<title>stationrealisations&#45;&gt;storedactionset</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M643.38,-322.95C653.07,-294.01 667.29,-263.36 687.96,-240 717.87,-206.19 757.31,-178.91 797.34,-157.41"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="798.34,-159.85 803.75,-154.04 795.89,-155.2 798.34,-159.85"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="687.96,-240 687.96,-262.8 875.96,-262.8 875.96,-240 687.96,-240"/>
<text xml:space="preserve" text-anchor="start" x="690.96" y="-259.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">physical switchings per action</text>
</g>
<!-- stationrealisations&#45;&gt;bboutage -->
<g id="edge16" class="edge">
<title>stationrealisations&#45;&gt;bboutage</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M530.65,-323.11C516.45,-304.64 503.9,-284.23 495.96,-262.8 487.5,-239.98 486.59,-214.41 489.3,-190.12"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="491.89,-190.58 490.27,-182.8 486.68,-189.89 491.89,-190.58"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="495.96,-240 495.96,-262.8 572.96,-262.8 572.96,-240 495.96,-240"/>
<text xml:space="preserve" text-anchor="start" x="498.96" y="-259.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">action set A</text>
</g>
</g>
</svg>`;case`loadflowFormat`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="1562pt" height="1410pt"
 viewBox="0.00 0.00 1562.00 1410.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1395.05)">
<g id="clust1" class="cluster">
<title>cluster_loadflowstore</title>
<polygon fill="#3e4651" stroke="#2d333d" points="8,-260 8,-1141.2 1319,-1141.2 1319,-260 8,-260"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1127.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">LOADFLOW RESULT STORE</text>
</g>
<!-- lfmetadata -->
<g id="node1" class="node">
<title>lfmetadata</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1272.02,-1080 951.98,-1080 951.98,-900 1272.02,-900 1272.02,-1080"/>
<text xml:space="preserve" text-anchor="start" x="1050.5" y="-1040" font-family="Arial" font-size="20.00" fill="#f8fafc">metadata.json</text>
<text xml:space="preserve" text-anchor="start" x="975.5" y="-1012" font-family="Arial" font-size="15.00" fill="#cbd5e1">The only non&#45;Parquet file: job_id and the</text>
<text xml:space="preserve" text-anchor="start" x="1020" y="-994" font-family="Arial" font-size="15.00" fill="#cbd5e1">global warnings list. Written</text>
<text xml:space="preserve" text-anchor="start" x="977.5" y="-976" font-family="Arial" font-size="15.00" fill="#cbd5e1">first, so its presence marks the folder as</text>
<text xml:space="preserve" text-anchor="start" x="1087" y="-958" font-family="Arial" font-size="15.00" fill="#cbd5e1">started.</text>
</g>
<!-- lfbranch -->
<g id="node2" class="node">
<title>lfbranch</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="406.06,-1080 47.94,-1080 47.94,-900 406.06,-900 406.06,-1080"/>
<text xml:space="preserve" text-anchor="start" x="126" y="-1049" font-family="Arial" font-size="20.00" fill="#f8fafc">branch_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="82.5" y="-1021" font-family="Arial" font-size="15.00" fill="#c2f0c2">index: timestep, contingency, element, side</text>
<text xml:space="preserve" text-anchor="start" x="91" y="-1003" font-family="Arial" font-size="15.00" fill="#c2f0c2">columns: i, p, q, loading, element_name,</text>
<text xml:space="preserve" text-anchor="start" x="163.5" y="-985" font-family="Arial" font-size="15.00" fill="#c2f0c2">contingency_name</text>
<text xml:space="preserve" text-anchor="start" x="68" y="-967" font-family="Arial" font-size="15.00" fill="#c2f0c2">Indexed per branch *end*, so a branch appears</text>
<text xml:space="preserve" text-anchor="start" x="145" y="-949" font-family="Arial" font-size="15.00" fill="#c2f0c2">twice per case. \`loading\`</text>
</g>
<!-- lfnode -->
<g id="node3" class="node">
<title>lfnode</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="836.02,-1080 515.98,-1080 515.98,-900 836.02,-900 836.02,-1080"/>
<text xml:space="preserve" text-anchor="start" x="583.5" y="-1040" font-family="Arial" font-size="20.00" fill="#f8fafc">node_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="549.5" y="-1012" font-family="Arial" font-size="15.00" fill="#c2f0c2">index: timestep, contingency, element</text>
<text xml:space="preserve" text-anchor="start" x="559" y="-994" font-family="Arial" font-size="15.00" fill="#c2f0c2">columns: vm, vm_loading, va, p, q,</text>
<text xml:space="preserve" text-anchor="start" x="538.5" y="-976" font-family="Arial" font-size="15.00" fill="#c2f0c2">vm_basecase_deviation, element_name,</text>
<text xml:space="preserve" text-anchor="start" x="612.5" y="-958" font-family="Arial" font-size="15.00" fill="#c2f0c2">contingency_name</text>
</g>
<!-- lfconverged -->
<g id="node4" class="node">
<title>lfconverged</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="818.56,-780 491.44,-780 491.44,-600 818.56,-600 818.56,-780"/>
<text xml:space="preserve" text-anchor="start" x="572" y="-749" font-family="Arial" font-size="20.00" fill="#f8fafc">converged.parquet</text>
<text xml:space="preserve" text-anchor="start" x="559" y="-721" font-family="Arial" font-size="15.00" fill="#c2f0c2">index: timestep, contingency</text>
<text xml:space="preserve" text-anchor="start" x="511.5" y="-703" font-family="Arial" font-size="15.00" fill="#c2f0c2">columns: status, iteration_count, warnings,</text>
<text xml:space="preserve" text-anchor="start" x="591.5" y="-685" font-family="Arial" font-size="15.00" fill="#c2f0c2">contingency_name</text>
<text xml:space="preserve" text-anchor="start" x="517.5" y="-667" font-family="Arial" font-size="15.00" fill="#c2f0c2">The index of what actually ran. Read this</text>
<text xml:space="preserve" text-anchor="start" x="564.5" y="-649" font-family="Arial" font-size="15.00" fill="#c2f0c2">first: non&#45;converging cases</text>
</g>
<!-- lfvadiff -->
<g id="node5" class="node">
<title>lfvadiff</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="381.02,-780 60.98,-780 60.98,-600 381.02,-600 381.02,-780"/>
<text xml:space="preserve" text-anchor="start" x="120.5" y="-731" font-family="Arial" font-size="20.00" fill="#f8fafc">va_diff_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="94.5" y="-703" font-family="Arial" font-size="15.00" fill="#c2f0c2">index: timestep, contingency, element</text>
<text xml:space="preserve" text-anchor="start" x="110" y="-685" font-family="Arial" font-size="15.00" fill="#c2f0c2">columns: va_diff, element_name,</text>
<text xml:space="preserve" text-anchor="start" x="157.5" y="-667" font-family="Arial" font-size="15.00" fill="#c2f0c2">contingency_name</text>
</g>
<!-- lfreg -->
<g id="node6" class="node">
<title>lfreg</title>
<polygon fill="#428a4f" stroke="#2d5d39" stroke-width="0" points="1279.06,-780 928.94,-780 928.94,-600 1279.06,-600 1279.06,-780"/>
<text xml:space="preserve" text-anchor="start" x="949" y="-731" font-family="Arial" font-size="20.00" fill="#f8fafc">regulating_element_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="977.5" y="-703" font-family="Arial" font-size="15.00" fill="#c2f0c2">index: timestep, contingency, element</text>
<text xml:space="preserve" text-anchor="start" x="964" y="-685" font-family="Arial" font-size="15.00" fill="#c2f0c2">columns: value, regulating_element_type,</text>
<text xml:space="preserve" text-anchor="start" x="986.5" y="-667" font-family="Arial" font-size="15.00" fill="#c2f0c2">element_name, contingency_name</text>
</g>
<!-- lfswitch -->
<g id="node7" class="node">
<title>lfswitch</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="368.02,-480 47.98,-480 47.98,-300 368.02,-300 368.02,-480"/>
<text xml:space="preserve" text-anchor="start" x="109" y="-449" font-family="Arial" font-size="20.00" fill="#ffe0c2">switch_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="81.5" y="-421" font-family="Arial" font-size="15.00" fill="#f9b27c">index: timestep, contingency, element</text>
<text xml:space="preserve" text-anchor="start" x="86" y="-403" font-family="Arial" font-size="15.00" fill="#f9b27c">columns: p, q, vm, i, element_name,</text>
<text xml:space="preserve" text-anchor="start" x="126.5" y="-385" font-family="Arial" font-size="15.00" fill="#f9b27c">contingency_name, side</text>
<text xml:space="preserve" text-anchor="start" x="79" y="-367" font-family="Arial" font-size="15.00" fill="#f9b27c">Optional &#45;&#45; the file is absent unless the</text>
<text xml:space="preserve" text-anchor="start" x="109" y="-349" font-family="Arial" font-size="15.00" fill="#f9b27c">table was populated. The one</text>
</g>
<!-- lfspps -->
<g id="node8" class="node">
<title>lfspps</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="809.56,-480 478.44,-480 478.44,-300 809.56,-300 809.56,-480"/>
<text xml:space="preserve" text-anchor="start" x="551.5" y="-449" font-family="Arial" font-size="20.00" fill="#ffe0c2">spps_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="548" y="-421" font-family="Arial" font-size="15.00" fill="#f9b27c">index: timestep, contingency</text>
<text xml:space="preserve" text-anchor="start" x="578.5" y="-403" font-family="Arial" font-size="15.00" fill="#f9b27c">columns: iterations,</text>
<text xml:space="preserve" text-anchor="start" x="547.5" y="-385" font-family="Arial" font-size="15.00" fill="#f9b27c">activated_schemes_per_iter,</text>
<text xml:space="preserve" text-anchor="start" x="498.5" y="-367" font-family="Arial" font-size="15.00" fill="#f9b27c">max_iterations_reached, power_flow_failed</text>
<text xml:space="preserve" text-anchor="start" x="614" y="-349" font-family="Arial" font-size="15.00" fill="#f9b27c">Optional.</text>
</g>
<!-- lfcascade -->
<g id="node9" class="node">
<title>lfcascade</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1278.56,-480 919.44,-480 919.44,-300 1278.56,-300 1278.56,-480"/>
<text xml:space="preserve" text-anchor="start" x="990.5" y="-449" font-family="Arial" font-size="20.00" fill="#ffe0c2">cascade_results.parquet</text>
<text xml:space="preserve" text-anchor="start" x="939.5" y="-421" font-family="Arial" font-size="15.00" fill="#f9b27c">index: timestep, contingency, cascade_number,</text>
<text xml:space="preserve" text-anchor="start" x="1053.5" y="-403" font-family="Arial" font-size="15.00" fill="#f9b27c">element_mrid</text>
<text xml:space="preserve" text-anchor="start" x="945.5" y="-385" font-family="Arial" font-size="15.00" fill="#f9b27c">columns: element_id, contingency_outage_id,</text>
<text xml:space="preserve" text-anchor="start" x="1009" y="-367" font-family="Arial" font-size="15.00" fill="#f9b27c">element_outage_group_id,</text>
<text xml:space="preserve" text-anchor="start" x="979" y="-349" font-family="Arial" font-size="15.00" fill="#f9b27c">element_name, contingency_name,</text>
</g>
<!-- acvalidator -->
<g id="node10" class="node">
<title>acvalidator</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1441.06,-180 1102.94,-180 1102.94,0 1441.06,0 1441.06,-180"/>
<text xml:space="preserve" text-anchor="start" x="1216.5" y="-158.8" font-family="Arial" font-size="20.00" fill="#f8fafc">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="1170.5" y="-130.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="1138" y="-111.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="1186.5" y="-93.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="1123" y="-75.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="1197" y="-57.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="1143" y="-39.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">candidate it evaluates gets a full result</text>
</g>
<!-- lfresults -->
<g id="node11" class="node">
<title>lfresults</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1060.56,-1380 685.44,-1380 685.44,-1200 1060.56,-1200 1060.56,-1380"/>
<text xml:space="preserve" text-anchor="start" x="801" y="-1349" font-family="Arial" font-size="20.00" fill="#f8fafc">LoadflowResults</text>
<text xml:space="preserve" text-anchor="start" x="722" y="-1321" font-family="Arial" font-size="15.00" fill="#cbd5e1">One container per computation job, holding a</text>
<text xml:space="preserve" text-anchor="start" x="791" y="-1303" font-family="Arial" font-size="15.00" fill="#cbd5e1">pandera&#45;validated frame</text>
<text xml:space="preserve" text-anchor="start" x="727" y="-1285" font-family="Arial" font-size="15.00" fill="#cbd5e1">per result family, plus warnings. Mirrored by</text>
<text xml:space="preserve" text-anchor="start" x="794.5" y="-1267" font-family="Arial" font-size="15.00" fill="#cbd5e1">LoadflowResultsPolars,</text>
<text xml:space="preserve" text-anchor="start" x="705.5" y="-1249" font-family="Arial" font-size="15.00" fill="#cbd5e1">whose schemas subclass the pandas ones so the</text>
</g>
<!-- initialloadflow -->
<g id="node12" class="node">
<title>initialloadflow</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1531.56,-1380 1170.44,-1380 1170.44,-1200 1531.56,-1200 1531.56,-1380"/>
<text xml:space="preserve" text-anchor="start" x="1250" y="-1358.8" font-family="Arial" font-size="20.00" fill="#f8fafc">store_initial_ca_results</text>
<text xml:space="preserve" text-anchor="start" x="1328.5" y="-1330.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">Parquet</text>
<text xml:space="preserve" text-anchor="start" x="1227.5" y="-1311.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Stores the AC N&#45;1 of the import, with</text>
<text xml:space="preserve" text-anchor="start" x="1247" y="-1293.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">permanent and artificial branch</text>
<text xml:space="preserve" text-anchor="start" x="1220" y="-1275.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">limits, as the reference every proposed</text>
<text xml:space="preserve" text-anchor="start" x="1262" y="-1257.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">topology is later compared</text>
<text xml:space="preserve" text-anchor="start" x="1190.5" y="-1239.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">against. Runs before the DC preprocessing and</text>
</g>
<!-- lfmetadata&#45;&gt;lfconverged -->
<!-- lfbranch&#45;&gt;lfnode -->
<!-- lfnode&#45;&gt;lfvadiff -->
<!-- lfconverged&#45;&gt;lfswitch -->
<!-- lfvadiff&#45;&gt;lfreg -->
<!-- lfcascade&#45;&gt;acvalidator -->
<g id="edge6" class="edge">
<title>lfcascade&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1173.89,-260C1187.74,-236.14 1202,-211.57 1215.21,-188.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1217.45,-190.2 1218.94,-182.39 1212.91,-187.56 1217.45,-190.2"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1031.07,-220.07 1031.07,-242.87 1197.07,-242.87 1197.07,-220.07 1031.07,-220.07"/>
<text xml:space="preserve" text-anchor="start" x="1034.07" y="-239.87" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial loadflow as baseline</text>
</g>
<!-- acvalidator&#45;&gt;lfmetadata -->
<g id="edge9" class="edge">
<title>acvalidator&#45;&gt;lfmetadata</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1311.17,-179.71C1320.56,-205.14 1329.24,-233.23 1334,-260 1374.42,-487.55 1425.65,-567.84 1334,-780 1331.04,-786.86 1327.72,-793.61 1324.11,-800.22"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1321.94,-798.71 1320.5,-806.53 1326.5,-801.32 1321.94,-798.71"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1166.52,-484.48 1166.52,-524.08 1378.52,-524.08 1378.52,-484.48 1166.52,-484.48"/>
<text xml:space="preserve" text-anchor="start" x="1169.52" y="-521.08" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="1169.52" y="-504.28" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- lfresults&#45;&gt;lfmetadata -->
<g id="edge7" class="edge">
<title>lfresults&#45;&gt;lfmetadata</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M944.06,-1200.4C957.07,-1184.18 971,-1166.81 985.07,-1149.27"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="987,-1151.05 989.65,-1143.56 982.91,-1147.77 987,-1151.05"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="859.72,-1170.9 859.72,-1193.7 967.72,-1193.7 967.72,-1170.9 859.72,-1170.9"/>
<text xml:space="preserve" text-anchor="start" x="862.72" y="-1190.7" font-family="Arial" font-size="14.00" fill="#c9c9c9">persisted per job</text>
</g>
<!-- initialloadflow&#45;&gt;lfmetadata -->
<g id="edge8" class="edge">
<title>initialloadflow&#45;&gt;lfmetadata</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1279.94,-1200.4C1266.93,-1184.18 1253,-1166.81 1238.93,-1149.27"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1241.09,-1147.77 1234.35,-1143.56 1237,-1151.05 1241.09,-1147.77"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1125.28,-1170.9 1125.28,-1193.7 1256.28,-1193.7 1256.28,-1170.9 1125.28,-1170.9"/>
<text xml:space="preserve" text-anchor="start" x="1128.28" y="-1190.7" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial AC N&#45;1 results</text>
</g>
</g>
</svg>`;case`index`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="3887pt" height="1976pt"
 viewBox="0.00 0.00 3887.00 1976.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1961.45)">
<g id="clust1" class="cluster">
<title>cluster_toop</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="1040.41,-8 1040.41,-1675.6 2836.41,-1675.6 2836.41,-8 1040.41,-8"/>
<text xml:space="preserve" text-anchor="start" x="1048.41" y="-1662.15" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<!-- interfaces -->
<g id="node1" class="node">
<title>interfaces</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2580.97,-920.7 2225.86,-920.7 2225.86,-740.7 2580.97,-740.7 2580.97,-920.7"/>
<text xml:space="preserve" text-anchor="start" x="2360.41" y="-890.5" font-family="Arial" font-size="20.00" fill="#f8fafc">Interfaces</text>
<text xml:space="preserve" text-anchor="start" x="2336.41" y="-862.5" font-family="Arial" font-size="13.00" fill="#cbd5e1">toop_engine_interfaces</text>
<text xml:space="preserve" text-anchor="start" x="2247.91" y="-842.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">The shared vocabulary. Everything here exists</text>
<text xml:space="preserve" text-anchor="start" x="2317.41" y="-824.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">so that two packages can</text>
<text xml:space="preserve" text-anchor="start" x="2245.91" y="-806.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">agree on a payload without depending on each</text>
<text xml:space="preserve" text-anchor="start" x="2384.41" y="-788.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">other.</text>
</g>
<!-- postprocess -->
<g id="node2" class="node">
<title>postprocess</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2409.47,-560.9 2073.36,-560.9 2073.36,-380.9 2409.47,-380.9 2409.47,-560.9"/>
<text xml:space="preserve" text-anchor="start" x="2124.41" y="-512.7" font-family="Arial" font-size="20.00" fill="#f8fafc">Postprocessing and export</text>
<text xml:space="preserve" text-anchor="start" x="2136.41" y="-484.7" font-family="Arial" font-size="13.00" fill="#cbd5e1">toop_engine_dc_solver.postprocess,</text>
<text xml:space="preserve" text-anchor="start" x="2093.41" y="-465.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">Turns an action index back into something a</text>
<text xml:space="preserve" text-anchor="start" x="2178.91" y="-447.1" font-family="Arial" font-size="15.00" fill="#cbd5e1">grid tool can open.</text>
</g>
<!-- lfservice -->
<g id="node3" class="node">
<title>lfservice</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1403.97,-1593.2 1080.86,-1593.2 1080.86,-1413.2 1403.97,-1413.2 1403.97,-1593.2"/>
<text xml:space="preserve" text-anchor="start" x="1155.41" y="-1572" font-family="Arial" font-size="20.00" fill="#eff6ff">AC loadflow service</text>
<text xml:space="preserve" text-anchor="start" x="1185.41" y="-1544" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl</text>
<text xml:space="preserve" text-anchor="start" x="1121.41" y="-1524.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">A standalone N&#45;1 service on its own</text>
<text xml:space="preserve" text-anchor="start" x="1112.91" y="-1506.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">loadflow_commands / loadflow_results</text>
<text xml:space="preserve" text-anchor="start" x="1100.91" y="-1488.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">/ loadflow_heartbeat topics. Present in the</text>
<text xml:space="preserve" text-anchor="start" x="1155.41" y="-1470.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">codebase but off the main</text>
<text xml:space="preserve" text-anchor="start" x="1105.41" y="-1452.4" font-family="Arial" font-size="15.00" fill="#bfdbfe">path: dev&#45;deployment does not create its</text>
</g>
<!-- importerparams -->
<g id="node4" class="node">
<title>importerparams</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2318.47,-1593.2 1954.36,-1593.2 1954.36,-1413.2 2318.47,-1413.2 2318.47,-1593.2"/>
<text xml:space="preserve" text-anchor="start" x="2046.91" y="-1562.2" font-family="Arial" font-size="20.00" fill="#f8fafc">Importer parameters</text>
<text xml:space="preserve" text-anchor="start" x="1985.41" y="-1534.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Carried by the StartPreprocessingCommand.</text>
<text xml:space="preserve" text-anchor="start" x="2040.91" y="-1516.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Fixes the scope of the whole</text>
<text xml:space="preserve" text-anchor="start" x="1990.91" y="-1498.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">run before any search happens: which grid,</text>
<text xml:space="preserve" text-anchor="start" x="2047.91" y="-1480.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">which area, which stations</text>
<text xml:space="preserve" text-anchor="start" x="1974.41" y="-1462.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">may be switched, which contingencies, and how</text>
</g>
<!-- dcparams -->
<g id="node5" class="node">
<title>dcparams</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1844.47,-1593.2 1514.36,-1593.2 1514.36,-1413.2 1844.47,-1413.2 1844.47,-1593.2"/>
<text xml:space="preserve" text-anchor="start" x="1569.41" y="-1544.2" font-family="Arial" font-size="20.00" fill="#f8fafc">DC optimizer parameters</text>
<text xml:space="preserve" text-anchor="start" x="1534.41" y="-1516.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Carried by the StartOptimizationCommand.</text>
<text xml:space="preserve" text-anchor="start" x="1594.41" y="-1498.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Everything about how the</text>
<text xml:space="preserve" text-anchor="start" x="1539.91" y="-1480.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">search behaves and what it optimizes for.</text>
</g>
<!-- acparams -->
<g id="node6" class="node">
<title>acparams</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2795.97,-1593.2 2428.86,-1593.2 2428.86,-1413.2 2795.97,-1413.2 2795.97,-1593.2"/>
<text xml:space="preserve" text-anchor="start" x="2505.41" y="-1553.2" font-family="Arial" font-size="20.00" fill="#f8fafc">AC validator parameters</text>
<text xml:space="preserve" text-anchor="start" x="2448.91" y="-1525.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Carried by the same StartOptimizationCommand</text>
<text xml:space="preserve" text-anchor="start" x="2535.91" y="-1507.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">as the DC parameters.</text>
<text xml:space="preserve" text-anchor="start" x="2470.91" y="-1489.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Mostly about what to reject and how much</text>
<text xml:space="preserve" text-anchor="start" x="2549.91" y="-1471.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">compute to spend.</text>
</g>
<!-- importer -->
<g id="node7" class="node">
<title>importer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1500.97,-560.9 1169.86,-560.9 1169.86,-380.9 1500.97,-380.9 1500.97,-560.9"/>
<text xml:space="preserve" text-anchor="start" x="1298.91" y="-539.7" font-family="Arial" font-size="20.00" fill="#eff6ff">Importer</text>
<text xml:space="preserve" text-anchor="start" x="1223.91" y="-511.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl, pandapower, JAX</text>
<text xml:space="preserve" text-anchor="start" x="1189.91" y="-492.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">Normalizes a raw grid into a processed grid</text>
<text xml:space="preserve" text-anchor="start" x="1262.41" y="-474.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">folder and derives the</text>
<text xml:space="preserve" text-anchor="start" x="1193.41" y="-456.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">solver artifacts. Most of it depends only on</text>
<text xml:space="preserve" text-anchor="start" x="1258.41" y="-438.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">the initial grid topology,</text>
<text xml:space="preserve" text-anchor="start" x="1220.91" y="-420.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">so it can run before the forecast is</text>
</g>
<!-- dcoptimizer -->
<g id="node8" class="node">
<title>dcoptimizer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1435.97,-1253.6 1080.86,-1253.6 1080.86,-1073.6 1435.97,-1073.6 1435.97,-1253.6"/>
<text xml:space="preserve" text-anchor="start" x="1198.91" y="-1232.4" font-family="Arial" font-size="20.00" fill="#eff6ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="1205.41" y="-1204.4" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="1112.91" y="-1184.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="1199.41" y="-1166.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="1100.91" y="-1148.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="1184.41" y="-1130.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="1103.91" y="-1112.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acvalidator -->
<g id="node9" class="node">
<title>acvalidator</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2703.47,-1253.6 2365.36,-1253.6 2365.36,-1073.6 2703.47,-1073.6 2703.47,-1253.6"/>
<text xml:space="preserve" text-anchor="start" x="2478.91" y="-1232.4" font-family="Arial" font-size="20.00" fill="#eff6ff">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="2432.91" y="-1204.4" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="2400.41" y="-1184.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="2448.91" y="-1166.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="2385.41" y="-1148.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="2459.41" y="-1130.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="2405.41" y="-1112.8" font-family="Arial" font-size="15.00" fill="#bfdbfe">candidate it evaluates gets a full result</text>
</g>
<!-- contingency -->
<g id="node10" class="node">
<title>contingency</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2407.97,-228 2074.86,-228 2074.86,-48 2407.97,-48 2407.97,-228"/>
<text xml:space="preserve" text-anchor="start" x="2147.41" y="-206.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Contingency analysis</text>
<text xml:space="preserve" text-anchor="start" x="2140.91" y="-178.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">toop_engine_contingency_analysis</text>
<text xml:space="preserve" text-anchor="start" x="2106.91" y="-159.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Runs an N&#45;1 analysis against whichever</text>
<text xml:space="preserve" text-anchor="start" x="2148.41" y="-141.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">backend holds the grid, and</text>
<text xml:space="preserve" text-anchor="start" x="2094.91" y="-123.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">normalizes both into the same result object.</text>
<text xml:space="preserve" text-anchor="start" x="2154.41" y="-105.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The two backends are not</text>
<text xml:space="preserve" text-anchor="start" x="2102.91" y="-87.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">at feature parity, so which one you import</text>
</g>
<!-- client -->
<g id="node11" class="node">
<title>client</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="1848.47,-1946.4 1510.36,-1946.4 1510.36,-1766.4 1848.47,-1766.4 1848.47,-1946.4"/>
<text xml:space="preserve" text-anchor="start" x="1548.41" y="-1915.4" font-family="Arial" font-size="20.00" fill="#f0f9ff">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="1530.41" y="-1887.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="1605.91" y="-1869.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="1538.91" y="-1851.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="1607.41" y="-1833.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="1533.41" y="-1815.4" font-family="Arial" font-size="15.00" fill="#b6ecf7">In operational use the whole run must finish</text>
</g>
<!-- kafka -->
<g id="node12" class="node">
<title>kafka</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="548.93,-919.56 193.9,-919.56 193.9,-741.84 548.93,-741.84 548.93,-919.56"/>
<text xml:space="preserve" text-anchor="start" x="346.91" y="-899.5" font-family="Arial" font-size="20.00" fill="#ffe0c2">Kafka</text>
<text xml:space="preserve" text-anchor="start" x="327.91" y="-871.5" font-family="Arial" font-size="13.00" fill="#f9b27c">confluent&#45;kafka</text>
<text xml:space="preserve" text-anchor="start" x="299.41" y="-851.9" font-family="Arial" font-size="15.00" fill="#f9b27c">Six topics, created by</text>
<text xml:space="preserve" text-anchor="start" x="217.91" y="-833.9" font-family="Arial" font-size="15.00" fill="#f9b27c">dev&#45;deployment/docker&#45;compose.yaml. Every</text>
<text xml:space="preserve" text-anchor="start" x="344.91" y="-815.9" font-family="Arial" font-size="15.00" fill="#f9b27c">payload</text>
<text xml:space="preserve" text-anchor="start" x="231.41" y="-797.9" font-family="Arial" font-size="15.00" fill="#f9b27c">is a pydantic model dumped to JSON and</text>
<text xml:space="preserve" text-anchor="start" x="276.41" y="-779.9" font-family="Arial" font-size="15.00" fill="#f9b27c">wrapped in a single protobuf</text>
</g>
<!-- unprocessedgridstore -->
<g id="node13" class="node">
<title>unprocessedgridstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M3198.47,-912.6C3198.47,-922.64 3126.28,-930.8 3037.41,-930.8 2948.55,-930.8 2876.36,-922.64 2876.36,-912.6 2876.36,-912.6 2876.36,-748.8 2876.36,-748.8 2876.36,-738.76 2948.55,-730.6 3037.41,-730.6 3126.28,-730.6 3198.47,-738.76 3198.47,-748.8 3198.47,-748.8 3198.47,-912.6 3198.47,-912.6"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M3198.47,-912.6C3198.47,-902.56 3126.28,-894.4 3037.41,-894.4 2948.55,-894.4 2876.36,-902.56 2876.36,-912.6"/>
<text xml:space="preserve" text-anchor="start" x="2934.91" y="-899.5" font-family="Arial" font-size="20.00" fill="#f8fafc">Unprocessed grid store</text>
<text xml:space="preserve" text-anchor="start" x="2961.91" y="-871.5" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec AbstractFileSystem</text>
<text xml:space="preserve" text-anchor="start" x="2907.41" y="-851.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">Where the source grid files land before</text>
<text xml:space="preserve" text-anchor="start" x="2923.41" y="-833.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">anything touches them. The same</text>
<text xml:space="preserve" text-anchor="start" x="2896.41" y="-815.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">kind of thing as the loadflow result store &#45;&#45;</text>
<text xml:space="preserve" text-anchor="start" x="2956.41" y="-797.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">an fsspec filesystem the</text>
<text xml:space="preserve" text-anchor="start" x="2912.91" y="-779.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">worker is handed, local disk or object</text>
</g>
<!-- loadflowstore -->
<g id="node14" class="node">
<title>loadflowstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M3238.47,-552.8C3238.47,-562.84 3157.32,-571 3057.41,-571 2957.51,-571 2876.36,-562.84 2876.36,-552.8 2876.36,-552.8 2876.36,-389 2876.36,-389 2876.36,-378.96 2957.51,-370.8 3057.41,-370.8 3157.32,-370.8 3238.47,-378.96 3238.47,-389 3238.47,-389 3238.47,-552.8 3238.47,-552.8"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M3238.47,-552.8C3238.47,-542.76 3157.32,-534.6 3057.41,-534.6 2957.51,-534.6 2876.36,-542.76 2876.36,-552.8"/>
<text xml:space="preserve" text-anchor="start" x="2966.41" y="-539.7" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="2991.41" y="-511.7" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="2951.41" y="-492.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="2896.41" y="-474.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="2912.91" y="-456.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="2906.91" y="-438.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="2971.91" y="-420.1" font-family="Arial" font-size="15.00" fill="#c2f0c2">topology it evaluates gets</text>
</g>
<!-- processedgrid -->
<g id="node15" class="node">
<title>processedgrid</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M1000.43,-912.6C1000.43,-922.64 928.71,-930.8 840.41,-930.8 752.12,-930.8 680.39,-922.64 680.39,-912.6 680.39,-912.6 680.39,-748.8 680.39,-748.8 680.39,-738.76 752.12,-730.6 840.41,-730.6 928.71,-730.6 1000.43,-738.76 1000.43,-748.8 1000.43,-748.8 1000.43,-912.6 1000.43,-912.6"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M1000.43,-912.6C1000.43,-902.56 928.71,-894.4 840.41,-894.4 752.12,-894.4 680.39,-902.56 680.39,-912.6"/>
<text xml:space="preserve" text-anchor="start" x="746.91" y="-899.5" font-family="Arial" font-size="20.00" fill="#f8fafc">Processed grid folder</text>
<text xml:space="preserve" text-anchor="start" x="764.91" y="-871.5" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec AbstractFileSystem</text>
<text xml:space="preserve" text-anchor="start" x="709.41" y="-851.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">One folder per import job, shared by all</text>
<text xml:space="preserve" text-anchor="start" x="736.41" y="-833.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">three stages and the only large</text>
<text xml:space="preserve" text-anchor="start" x="701.91" y="-815.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">payload that never travels through Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="731.41" y="-797.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">fsspec keeps the backend open:</text>
<text xml:space="preserve" text-anchor="start" x="701.41" y="-779.9" font-family="Arial" font-size="15.00" fill="#c2f0c2">local disk in the dev setup, object storage</text>
</g>
<!-- downstream -->
<g id="node16" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="544.47,-560.9 222.36,-560.9 222.36,-380.9 544.47,-380.9 544.47,-560.9"/>
<text xml:space="preserve" text-anchor="start" x="242.41" y="-511.9" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="245.91" y="-483.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="271.91" y="-465.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="275.91" y="-447.9" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- interfaces&#45;&gt;postprocess -->
<g id="edge7" class="edge">
<title>interfaces&#45;&gt;postprocess</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2295.21,-740.85C2276.36,-719.89 2259.32,-696.13 2248.41,-670.6 2235.2,-639.68 2231.03,-603.51 2230.93,-570.58"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2233.56,-571.02 2231,-563.49 2228.31,-570.97 2233.56,-571.02"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2248.41,-639.4 2248.41,-662.2 2273.41,-662.2 2273.41,-639.4 2248.41,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="2251.41" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- interfaces&#45;&gt;importer -->
<g id="edge9" class="edge">
<title>interfaces&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2226,-786.73C2110.56,-757.35 1957.32,-715.69 1824.41,-670.6 1816.22,-667.82 1647.23,-599.05 1510.18,-543.19"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1511.52,-540.9 1503.59,-540.5 1509.54,-545.76 1511.52,-540.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1824.41,-639.4 1824.41,-662.2 1849.41,-662.2 1849.41,-639.4 1824.41,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="1827.41" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- interfaces&#45;&gt;contingency -->
<g id="edge10" class="edge">
<title>interfaces&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2297.69,-740.89C2240.89,-694.9 2175.82,-645.39 2142.41,-631 1991.01,-565.81 1883.13,-696.63 1776.41,-571 1718.81,-503.18 1729.59,-446.46 1776.41,-370.8 1839.3,-269.19 1961.62,-210.67 2065.03,-177.85"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2065.7,-180.39 2072.08,-175.65 2064.14,-175.38 2065.7,-180.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1776.41,-459.5 1776.41,-482.3 2018.41,-482.3 2018.41,-459.5 1776.41,-459.5"/>
<text xml:space="preserve" text-anchor="start" x="1779.41" y="-479.3" font-family="Arial" font-size="14.00" fill="#c9c9c9">monitored elements and contingencies</text>
</g>
<!-- interfaces&#45;&gt;loadflowstore -->
<g id="edge6" class="edge">
<title>interfaces&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2580.69,-767.88C2648.41,-741.49 2725.21,-708.1 2791.41,-670.6 2839.24,-643.51 2888.34,-609.03 2931.13,-576.43"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2932.57,-578.64 2936.93,-572 2929.38,-574.47 2932.57,-578.64"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2853.15,-639.4 2853.15,-662.2 2961.15,-662.2 2961.15,-639.4 2853.15,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="2856.15" y="-659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">persisted per job</text>
</g>
<!-- interfaces&#45;&gt;processedgrid -->
<g id="edge8" class="edge">
<title>interfaces&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2226.13,-830.7C1923.41,-830.7 1310.65,-830.7 1011.92,-830.7"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1011.96,-828.08 1004.46,-830.7 1011.96,-833.33 1011.96,-828.08"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1531.15,-833.7 1531.15,-856.5 1695.15,-856.5 1695.15,-833.7 1531.15,-833.7"/>
<text xml:space="preserve" text-anchor="start" x="1534.15" y="-853.5" font-family="Arial" font-size="14.00" fill="#c9c9c9">serialized once per import</text>
</g>
<!-- postprocess&#45;&gt;interfaces -->
<g id="edge16" class="edge">
<title>postprocess&#45;&gt;interfaces</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2281.6,-560.66C2305.06,-612.47 2334.7,-677.93 2358.87,-731.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2356.43,-732.29 2361.91,-738.04 2361.21,-730.12 2356.43,-732.29"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2327.27,-639.4 2327.27,-662.2 2476.27,-662.2 2476.27,-639.4 2327.27,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="2330.27" y="-659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">switch id and new state</text>
</g>
<!-- postprocess&#45;&gt;contingency -->
<g id="edge18" class="edge">
<title>postprocess&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2241.41,-381.22C2241.41,-337.18 2241.41,-283.81 2241.41,-238.38"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2244.04,-238.45 2241.41,-230.95 2238.79,-238.45 2244.04,-238.45"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2241.41,-288 2241.41,-310.8 2403.41,-310.8 2403.41,-288 2241.41,-288"/>
<text xml:space="preserve" text-anchor="start" x="2244.41" y="-307.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">grid with topology applied</text>
</g>
<!-- postprocess&#45;&gt;processedgrid -->
<g id="edge17" class="edge">
<title>postprocess&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2073.65,-543.83C2046.17,-553.96 2017.74,-563.47 1990.41,-571 1778.34,-629.46 1220.73,-657.02 1013.41,-730.6 1010.75,-731.54 1008.09,-732.53 1005.42,-733.55"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1004.55,-731.07 998.56,-736.28 1006.49,-735.95 1004.55,-731.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1614.6,-639.4 1614.6,-662.2 1639.6,-662.2 1639.6,-639.4 1614.6,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="1617.6" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- importerparams&#45;&gt;importer -->
<g id="edge23" class="edge">
<title>importerparams&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2109.29,-1413.49C2052.78,-1241.83 1907.29,-860.89 1666.41,-631 1622.27,-588.87 1564.86,-556 1510.28,-531.36"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1511.53,-529.04 1503.61,-528.39 1509.4,-533.84 1511.53,-529.04"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1937.67,-990.8 1937.67,-1013.6 2111.67,-1013.6 2111.67,-990.8 1937.67,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="1940.67" y="-1010.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">scope, limits, contingencies</text>
</g>
<!-- dcparams&#45;&gt;dcoptimizer -->
<g id="edge24" class="edge">
<title>dcparams&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1567.59,-1413.27C1542.82,-1393.52 1516.71,-1372.67 1492.41,-1353.2 1454.59,-1322.89 1413.41,-1289.74 1376.28,-1259.81"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1378.27,-1258.04 1370.78,-1255.37 1374.97,-1262.13 1378.27,-1258.04"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1492.41,-1313.6 1492.41,-1353.2 1695.41,-1353.2 1695.41,-1313.6 1492.41,-1313.6"/>
<text xml:space="preserve" text-anchor="start" x="1495.41" y="-1350.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">search bounds, fitness, operator</text>
<text xml:space="preserve" text-anchor="start" x="1495.41" y="-1333.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">probabilities</text>
</g>
<!-- acparams&#45;&gt;acvalidator -->
<g id="edge25" class="edge">
<title>acparams&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2591.92,-1413.5C2581.28,-1367.45 2568.25,-1311.04 2557.28,-1263.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2559.9,-1263.25 2555.65,-1256.54 2554.79,-1264.43 2559.9,-1263.25"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2575.75,-1313.6 2575.75,-1353.2 2793.75,-1353.2 2793.75,-1313.6 2575.75,-1313.6"/>
<text xml:space="preserve" text-anchor="start" x="2578.75" y="-1350.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">compute budget, pruning, rejection</text>
<text xml:space="preserve" text-anchor="start" x="2578.75" y="-1333.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">thresholds</text>
</g>
<!-- importer&#45;&gt;interfaces -->
<g id="edge26" class="edge">
<title>importer&#45;&gt;interfaces</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1500.67,-543.94C1528.27,-554.15 1556.88,-563.66 1584.41,-571 1765.76,-619.37 1823.19,-575.3 2002.41,-631 2048.01,-645.17 2142.24,-691.53 2227.91,-735.9"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2226.68,-738.22 2234.55,-739.35 2229.1,-733.56 2226.68,-738.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2090.23,-639.4 2090.23,-662.2 2115.23,-662.2 2115.23,-639.4 2090.23,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="2093.23" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- importer&#45;&gt;contingency -->
<g id="edge30" class="edge">
<title>importer&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1446.87,-381.09C1493.34,-347.74 1549.27,-312.11 1604.41,-288 1753,-223.04 1933.99,-184.19 2065.02,-162.53"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2065.19,-165.17 2072.17,-161.36 2064.35,-159.98 2065.19,-165.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1604.41,-288 1604.41,-310.8 1693.41,-310.8 1693.41,-288 1604.41,-288"/>
<text xml:space="preserve" text-anchor="start" x="1607.41" y="-307.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">base grid N&#45;1</text>
</g>
<!-- importer&#45;&gt;kafka -->
<g id="edge27" class="edge">
<title>importer&#45;&gt;kafka</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1169.86,-480.1C1009.76,-493.95 764.43,-531.14 576.41,-631 528.3,-656.55 483.7,-696.8 448.48,-734.4"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="446.78,-732.37 443.62,-739.66 450.64,-735.94 446.78,-732.37"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="576.41,-639.4 576.41,-662.2 601.41,-662.2 601.41,-639.4 576.41,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="579.41" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- importer&#45;&gt;loadflowstore -->
<g id="edge28" class="edge">
<title>importer&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1500.75,-541.94C1528.41,-552.49 1557.01,-562.62 1584.41,-571 1708.52,-608.94 1740.86,-618.54 1869.91,-632.22 2023.15,-648.46 2634.14,-635.34 2863.41,-571 2866.94,-570.01 2870.48,-568.95 2874.02,-567.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2874.85,-570.32 2881.14,-565.47 2873.19,-565.34 2874.85,-570.32"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1869.91,-639.4 1869.91,-662.2 2000.91,-662.2 2000.91,-639.4 1869.91,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="1872.91" y="-659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial AC N&#45;1 results</text>
</g>
<!-- importer&#45;&gt;processedgrid -->
<g id="edge29" class="edge">
<title>importer&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1268.7,-560.82C1237.19,-598.36 1197.13,-640.29 1154.41,-670.6 1098.87,-710.01 1074.4,-700.29 1013.41,-730.6 1011.82,-731.39 1010.21,-732.2 1008.6,-733.01"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1007.53,-730.61 1002.04,-736.35 1009.91,-735.29 1007.53,-730.61"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1199.83,-639.4 1199.83,-662.2 1224.83,-662.2 1224.83,-639.4 1199.83,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="1202.83" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- dcoptimizer&#45;&gt;kafka -->
<g id="edge31" class="edge">
<title>dcoptimizer&#45;&gt;kafka</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1080.99,-1144.45C944.11,-1125.34 752.85,-1087.32 599.41,-1013.6 553.07,-991.34 507.63,-958.12 469.66,-926.14"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="471.39,-924.17 463.97,-921.3 467.98,-928.17 471.39,-924.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="599.41,-990.8 599.41,-1013.6 624.41,-1013.6 624.41,-990.8 599.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="602.41" y="-998.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- acvalidator&#45;&gt;interfaces -->
<g id="edge32" class="edge">
<title>acvalidator&#45;&gt;interfaces</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2499.31,-1073.92C2481.76,-1029.6 2460.47,-975.83 2442.42,-930.22"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2444.93,-929.44 2439.73,-923.43 2440.05,-931.37 2444.93,-929.44"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2472.84,-990.8 2472.84,-1013.6 2590.84,-1013.6 2590.84,-990.8 2472.84,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="2475.84" y="-1010.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">accepted topology</text>
</g>
<!-- acvalidator&#45;&gt;contingency -->
<g id="edge36" class="edge">
<title>acvalidator&#45;&gt;contingency</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2589.13,-1073.97C2599.58,-1054.56 2609.66,-1033.78 2617.41,-1013.6 2630.96,-978.36 2632.16,-968.32 2636.41,-930.8 2664.63,-681.68 2697.37,-585.81 2568.41,-370.8 2532.33,-310.63 2473.94,-261.9 2416.42,-224.93"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2417.87,-222.74 2410.13,-220.95 2415.07,-227.18 2417.87,-222.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2662.13,-639.4 2662.13,-662.2 2687.13,-662.2 2687.13,-639.4 2662.13,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="2665.13" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- acvalidator&#45;&gt;kafka -->
<g id="edge33" class="edge">
<title>acvalidator&#45;&gt;kafka</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2365.47,-1150.9C2104.14,-1131.33 1586.71,-1086.75 1152.41,-1013.6 1107.89,-1006.1 1097.81,-999.01 1053.41,-990.8 864.53,-955.88 811.25,-979.37 625.41,-930.8 603.47,-925.07 580.85,-918.01 558.56,-910.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="559.5,-907.87 551.55,-907.87 557.77,-912.82 559.5,-907.87"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1152.41,-990.8 1152.41,-1013.6 1177.41,-1013.6 1177.41,-990.8 1152.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="1155.41" y="-998.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- acvalidator&#45;&gt;loadflowstore -->
<g id="edge34" class="edge">
<title>acvalidator&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2703.22,-1131.78C2894.42,-1093.03 3188.03,-1020.58 3253.41,-930.8 3332.37,-822.39 3244.16,-677.28 3162.09,-579.5"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3164.25,-578 3157.39,-573.98 3160.25,-581.39 3164.25,-578"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3284.23,-810.9 3284.23,-850.5 3496.23,-850.5 3496.23,-810.9 3284.23,-810.9"/>
<text xml:space="preserve" text-anchor="start" x="3287.23" y="-847.5" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="3287.23" y="-830.7" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- acvalidator&#45;&gt;processedgrid -->
<g id="edge35" class="edge">
<title>acvalidator&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2365.81,-1145.75C2147.64,-1122.63 1756.27,-1076.63 1425.41,-1013.6 1384.03,-1005.72 1374.58,-999.75 1333.41,-990.8 1192.02,-960.05 1149.2,-980.79 1013.41,-930.8 1010.93,-929.88 1008.43,-928.93 1005.94,-927.95"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1007.04,-925.57 999.1,-925.17 1005.06,-930.43 1007.04,-925.57"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1425.41,-990.8 1425.41,-1013.6 1586.41,-1013.6 1586.41,-990.8 1425.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="1428.41" y="-1010.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">summaries and diagrams</text>
</g>
<!-- contingency&#45;&gt;interfaces -->
<g id="edge37" class="edge">
<title>contingency&#45;&gt;interfaces</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2372.73,-227.86C2393.82,-246.04 2414.06,-266.32 2430.41,-288 2454.38,-319.76 2453.83,-332.45 2464.41,-370.8 2545.31,-663.99 2523.81,-599.83 2503.41,-670.6 2497.34,-691.67 2487.57,-712.64 2476.45,-732.11"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2474.33,-730.55 2472.79,-738.34 2478.85,-733.21 2474.33,-730.55"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2516.52,-459.5 2516.52,-482.3 2541.52,-482.3 2541.52,-459.5 2516.52,-459.5"/>
<text xml:space="preserve" text-anchor="start" x="2519.52" y="-467" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- client&#45;&gt;importerparams -->
<g id="edge2" class="edge">
<title>client&#45;&gt;importerparams</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1832.52,-1766.55C1862.06,-1747.7 1892.24,-1727.18 1919.41,-1706.4 1961.5,-1674.22 2004.81,-1635.24 2041.56,-1600.13"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2043.04,-1602.35 2046.64,-1595.26 2039.41,-1598.56 2043.04,-1602.35"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1944.63,-1683.6 1944.63,-1706.4 2168.63,-1706.4 2168.63,-1683.6 1944.63,-1683.6"/>
<text xml:space="preserve" text-anchor="start" x="1947.63" y="-1703.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in StartPreprocessingCommand</text>
</g>
<!-- client&#45;&gt;dcparams -->
<g id="edge3" class="edge">
<title>client&#45;&gt;dcparams</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1679.41,-1766.78C1679.41,-1717.06 1679.41,-1654.91 1679.41,-1603.55"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1682.04,-1603.63 1679.41,-1596.13 1676.79,-1603.63 1682.04,-1603.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1679.41,-1683.6 1679.41,-1706.4 1892.41,-1706.4 1892.41,-1683.6 1679.41,-1683.6"/>
<text xml:space="preserve" text-anchor="start" x="1682.41" y="-1703.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in StartOptimizationCommand</text>
</g>
<!-- client&#45;&gt;acparams -->
<g id="edge4" class="edge">
<title>client&#45;&gt;acparams</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1848.35,-1829.62C1992.68,-1803.44 2203.34,-1755.22 2373.41,-1675.6 2416.8,-1655.29 2460.49,-1626.84 2498.43,-1598.95"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2499.55,-1601.39 2504.01,-1594.82 2496.42,-1597.18 2499.55,-1601.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2338.26,-1683.6 2338.26,-1706.4 2611.26,-1706.4 2611.26,-1683.6 2338.26,-1683.6"/>
<text xml:space="preserve" text-anchor="start" x="2341.26" y="-1703.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in the same StartOptimizationCommand</text>
</g>
<!-- client&#45;&gt;kafka -->
<g id="edge1" class="edge">
<title>client&#45;&gt;kafka</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1510.58,-1840.36C1367.89,-1820.87 1163.35,-1776.44 1013.41,-1675.6 721.34,-1479.17 507.43,-1105.4 417.85,-928.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="420.25,-927.7 414.52,-922.18 415.56,-930.06 420.25,-927.7"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="682,-1322 682,-1344.8 707,-1344.8 707,-1322 682,-1322"/>
<text xml:space="preserve" text-anchor="start" x="685" y="-1329.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- kafka&#45;&gt;importer -->
<g id="edge12" class="edge">
<title>kafka&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M317.79,-742.06C302.03,-704.22 295.17,-661.68 321.41,-631 334.99,-615.12 873.14,-537.09 1159.78,-496.51"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1159.82,-499.16 1166.88,-495.51 1159.08,-493.96 1159.82,-499.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="321.41,-639.4 321.41,-662.2 454.41,-662.2 454.41,-639.4 321.41,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="324.41" y="-659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">consumes command</text>
</g>
<!-- kafka&#45;&gt;dcoptimizer -->
<g id="edge13" class="edge">
<title>kafka&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M511.29,-919.54C567.84,-952.36 634.71,-987.87 698.41,-1013.6 818.84,-1062.24 960.21,-1099.71 1070.79,-1124.91"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1070.2,-1127.47 1078.1,-1126.57 1071.36,-1122.35 1070.2,-1127.47"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="698.41,-990.8 698.41,-1013.6 831.41,-1013.6 831.41,-990.8 698.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="701.41" y="-1010.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">consumes command</text>
</g>
<!-- kafka&#45;&gt;acvalidator -->
<g id="edge14" class="edge">
<title>kafka&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M548.78,-905.21C574.26,-914.51 600.34,-923.35 625.41,-930.8 750.75,-968.02 785.7,-962.79 913.41,-990.8 957.52,-1000.47 967.85,-1006.32 1012.41,-1013.6 1266.71,-1055.15 2010.34,-1119.35 2355.09,-1147.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2354.72,-1150.57 2362.41,-1148.57 2355.15,-1145.34 2354.72,-1150.57"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1012.41,-990.8 1012.41,-1013.6 1037.41,-1013.6 1037.41,-990.8 1012.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="1015.41" y="-998.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- kafka&#45;&gt;downstream -->
<g id="edge11" class="edge">
<title>kafka&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M193.97,-797.88C78.25,-768.54 -41.72,-716.33 14.41,-631 58.81,-563.51 138.55,-524.52 212.57,-502.06"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="213.14,-504.63 219.59,-499.99 211.66,-499.59 213.14,-504.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="14.41,-639.4 14.41,-662.2 206.41,-662.2 206.41,-639.4 14.41,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="17.41" y="-659.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">validated topologies for review</text>
</g>
<!-- unprocessedgridstore&#45;&gt;importer -->
<g id="edge5" class="edge">
<title>unprocessedgridstore&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3036.72,-729.81C3031.03,-693.2 3017.63,-654.91 2988.41,-631 2867.58,-532.11 1737.1,-603.66 1584.41,-571 1560.01,-565.78 1534.91,-558.4 1510.46,-549.99"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1511.57,-547.6 1503.62,-547.59 1509.83,-552.55 1511.57,-547.6"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3018.16,-639.4 3018.16,-662.2 3043.16,-662.2 3043.16,-639.4 3018.16,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="3021.16" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- loadflowstore&#45;&gt;acvalidator -->
<g id="edge15" class="edge">
<title>loadflowstore&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3239.14,-491.55C3384.03,-518.31 3575.71,-581.5 3657.41,-730.6 3700.17,-808.63 3703.27,-854.55 3657.41,-930.8 3561.41,-1090.45 3005.56,-1140.56 2713.55,-1155.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2713.65,-1153.33 2706.3,-1156.34 2713.92,-1158.57 2713.65,-1153.33"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3690.65,-819.3 3690.65,-842.1 3856.65,-842.1 3856.65,-819.3 3690.65,-819.3"/>
<text xml:space="preserve" text-anchor="start" x="3693.65" y="-839.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial loadflow as baseline</text>
</g>
<!-- processedgrid&#45;&gt;importer -->
<g id="edge20" class="edge">
<title>processedgrid&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M971.77,-734.76C1043.61,-682.83 1132.34,-618.69 1204.16,-566.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1205.4,-569.12 1209.94,-562.6 1202.32,-564.86 1205.4,-569.12"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1102.76,-639.4 1102.76,-662.2 1127.76,-662.2 1127.76,-639.4 1102.76,-639.4"/>
<text xml:space="preserve" text-anchor="start" x="1105.76" y="-646.9" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- processedgrid&#45;&gt;dcoptimizer -->
<g id="edge21" class="edge">
<title>processedgrid&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M839.84,-931.48C844.17,-960.59 853.57,-990.69 872.41,-1013.6 921.98,-1073.86 998.97,-1109.81 1071.2,-1131.23"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1070.06,-1133.63 1077.99,-1133.19 1071.51,-1128.59 1070.06,-1133.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="872.41,-990.8 872.41,-1013.6 897.41,-1013.6 897.41,-990.8 872.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="875.41" y="-998.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- processedgrid&#45;&gt;acvalidator -->
<g id="edge22" class="edge">
<title>processedgrid&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M997.56,-923.72C1002.87,-926.18 1008.16,-928.55 1013.41,-930.8 1072.87,-956.24 1229.08,-1000.5 1292.41,-1013.6 1666.26,-1090.92 2112.31,-1132.31 2355.13,-1150.66"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2354.77,-1153.26 2362.45,-1151.2 2355.17,-1148.03 2354.77,-1153.26"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1292.41,-990.8 1292.41,-1013.6 1317.41,-1013.6 1317.41,-990.8 1292.41,-990.8"/>
<text xml:space="preserve" text-anchor="start" x="1295.41" y="-998.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- processedgrid&#45;&gt;downstream -->
<g id="edge19" class="edge">
<title>processedgrid&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M735.94,-731.53C699.34,-698.51 657.38,-662.18 617.41,-631 589.53,-609.25 558.83,-587.15 529.1,-566.62"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="530.9,-564.68 523.23,-562.59 527.93,-569 530.9,-564.68"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="661.82,-631 661.82,-670.6 930.82,-670.6 930.82,-631 661.82,-631"/>
<text xml:space="preserve" text-anchor="start" x="664.82" y="-667.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">UCTE, DGS, OpenRAO summaries, single</text>
<text xml:space="preserve" text-anchor="start" x="664.82" y="-650.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">line diagrams</text>
</g>
</g>
</svg>`;case`overview`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="4727pt" height="1303pt"
 viewBox="0.00 0.00 4727.00 1303.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1288.29)">
<!-- client -->
<g id="node1" class="node">
<title>client</title>
<polygon fill="#0284c7" stroke="#0369a1" stroke-width="0" points="338.11,-751 0,-751 0,-571 338.11,-571 338.11,-751"/>
<text xml:space="preserve" text-anchor="start" x="38.06" y="-720" font-family="Arial" font-size="20.00" fill="#f0f9ff">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="20.06" y="-692" font-family="Arial" font-size="15.00" fill="#b6ecf7">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="95.56" y="-674" font-family="Arial" font-size="15.00" fill="#b6ecf7">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="28.56" y="-656" font-family="Arial" font-size="15.00" fill="#b6ecf7">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="97.06" y="-638" font-family="Arial" font-size="15.00" fill="#b6ecf7">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="23.06" y="-620" font-family="Arial" font-size="15.00" fill="#b6ecf7">In operational use the whole run must finish</text>
</g>
<!-- importercommands -->
<g id="node2" class="node">
<title>importercommands</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="1074.14,-747.01 678.11,-747.01 678.11,-575 1074.14,-575 1074.14,-747.01"/>
<text xml:space="preserve" text-anchor="start" x="786.13" y="-693" font-family="Arial" font-size="20.00" fill="#ffe0c2">importer_commands</text>
<text xml:space="preserve" text-anchor="start" x="702.13" y="-665" font-family="Arial" font-size="15.00" fill="#f9b27c">StartPreprocessingCommand, ShutdownCommand.</text>
<text xml:space="preserve" text-anchor="start" x="832.63" y="-647" font-family="Arial" font-size="15.00" fill="#f9b27c">24 partitions.</text>
</g>
<!-- unprocessedgridstore -->
<g id="node3" class="node">
<title>unprocessedgridstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M1037.18,-446.9C1037.18,-456.95 965,-465.1 876.13,-465.1 787.26,-465.1 715.07,-456.95 715.07,-446.9 715.07,-446.9 715.07,-283.1 715.07,-283.1 715.07,-273.06 787.26,-264.9 876.13,-264.9 965,-264.9 1037.18,-273.06 1037.18,-283.1 1037.18,-283.1 1037.18,-446.9 1037.18,-446.9"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M1037.18,-446.9C1037.18,-436.86 965,-428.7 876.13,-428.7 787.26,-428.7 715.07,-436.86 715.07,-446.9"/>
<text xml:space="preserve" text-anchor="start" x="773.63" y="-433.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Unprocessed grid store</text>
<text xml:space="preserve" text-anchor="start" x="800.63" y="-405.8" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec AbstractFileSystem</text>
<text xml:space="preserve" text-anchor="start" x="746.13" y="-386.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">Where the source grid files land before</text>
<text xml:space="preserve" text-anchor="start" x="762.13" y="-368.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">anything touches them. The same</text>
<text xml:space="preserve" text-anchor="start" x="735.13" y="-350.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">kind of thing as the loadflow result store &#45;&#45;</text>
<text xml:space="preserve" text-anchor="start" x="795.13" y="-332.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">an fsspec filesystem the</text>
<text xml:space="preserve" text-anchor="start" x="751.63" y="-314.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">worker is handed, local disk or object</text>
</g>
<!-- importer -->
<g id="node4" class="node">
<title>importer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1783.26,-562 1452.14,-562 1452.14,-382 1783.26,-382 1783.26,-562"/>
<text xml:space="preserve" text-anchor="start" x="1581.2" y="-540.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Importer</text>
<text xml:space="preserve" text-anchor="start" x="1506.2" y="-512.8" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl, pandapower, JAX</text>
<text xml:space="preserve" text-anchor="start" x="1472.2" y="-493.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Normalizes a raw grid into a processed grid</text>
<text xml:space="preserve" text-anchor="start" x="1544.7" y="-475.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">folder and derives the</text>
<text xml:space="preserve" text-anchor="start" x="1475.7" y="-457.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">solver artifacts. Most of it depends only on</text>
<text xml:space="preserve" text-anchor="start" x="1540.7" y="-439.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">the initial grid topology,</text>
<text xml:space="preserve" text-anchor="start" x="1503.2" y="-421.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">so it can run before the forecast is</text>
</g>
<!-- processedgrid -->
<g id="node5" class="node">
<title>processedgrid</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M2517.29,-477.9C2517.29,-487.95 2445.57,-496.1 2357.27,-496.1 2268.98,-496.1 2197.25,-487.95 2197.25,-477.9 2197.25,-477.9 2197.25,-314.1 2197.25,-314.1 2197.25,-304.06 2268.98,-295.9 2357.27,-295.9 2445.57,-295.9 2517.29,-304.06 2517.29,-314.1 2517.29,-314.1 2517.29,-477.9 2517.29,-477.9"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M2517.29,-477.9C2517.29,-467.86 2445.57,-459.7 2357.27,-459.7 2268.98,-459.7 2197.25,-467.86 2197.25,-477.9"/>
<text xml:space="preserve" text-anchor="start" x="2263.77" y="-464.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Processed grid folder</text>
<text xml:space="preserve" text-anchor="start" x="2281.77" y="-436.8" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec AbstractFileSystem</text>
<text xml:space="preserve" text-anchor="start" x="2226.27" y="-417.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">One folder per import job, shared by all</text>
<text xml:space="preserve" text-anchor="start" x="2253.27" y="-399.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">three stages and the only large</text>
<text xml:space="preserve" text-anchor="start" x="2218.77" y="-381.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">payload that never travels through Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="2248.27" y="-363.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">fsspec keeps the backend open:</text>
<text xml:space="preserve" text-anchor="start" x="2218.27" y="-345.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">local disk in the dev setup, object storage</text>
</g>
<!-- loadflowstore -->
<g id="node6" class="node">
<title>loadflowstore</title>
<path fill="#428a4f" stroke="#2d5d39" stroke-width="2" d="M2538.33,-789.9C2538.33,-799.95 2457.18,-808.1 2357.27,-808.1 2257.37,-808.1 2176.22,-799.95 2176.22,-789.9 2176.22,-789.9 2176.22,-626.1 2176.22,-626.1 2176.22,-616.06 2257.37,-607.9 2357.27,-607.9 2457.18,-607.9 2538.33,-616.06 2538.33,-626.1 2538.33,-626.1 2538.33,-789.9 2538.33,-789.9"/>
<path fill="none" stroke="#2d5d39" stroke-width="2" d="M2538.33,-789.9C2538.33,-779.86 2457.18,-771.7 2357.27,-771.7 2257.37,-771.7 2176.22,-779.86 2176.22,-789.9"/>
<text xml:space="preserve" text-anchor="start" x="2266.27" y="-776.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Loadflow result store</text>
<text xml:space="preserve" text-anchor="start" x="2291.27" y="-748.8" font-family="Arial" font-size="13.00" fill="#c2f0c2">fsspec, polars, Parquet</text>
<text xml:space="preserve" text-anchor="start" x="2251.27" y="-729.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">Loadflow tables addressed by a</text>
<text xml:space="preserve" text-anchor="start" x="2196.27" y="-711.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">StoredLoadflowReference passed in messages,</text>
<text xml:space="preserve" text-anchor="start" x="2212.77" y="-693.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">so the tables themselves stay out of Kafka.</text>
<text xml:space="preserve" text-anchor="start" x="2206.77" y="-675.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">The AC&#45;Validator is the main producer: every</text>
<text xml:space="preserve" text-anchor="start" x="2271.77" y="-657.2" font-family="Arial" font-size="15.00" fill="#c2f0c2">topology it evaluates gets</text>
</g>
<!-- importerresults -->
<g id="node7" class="node">
<title>importerresults</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="2522.29,-172.01 2192.26,-172.01 2192.26,0 2522.29,0 2522.29,-172.01"/>
<text xml:space="preserve" text-anchor="start" x="2286.77" y="-118" font-family="Arial" font-size="20.00" fill="#ffe0c2">importer_results</text>
<text xml:space="preserve" text-anchor="start" x="2261.27" y="-90" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessingStartedResult,</text>
<text xml:space="preserve" text-anchor="start" x="2216.27" y="-72" font-family="Arial" font-size="15.00" fill="#f9b27c">PreprocessingSuccessResult, ErrorResult</text>
</g>
<!-- commands -->
<g id="node8" class="node">
<title>commands</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="2555.29,-1102.01 2159.26,-1102.01 2159.26,-930 2555.29,-930 2555.29,-1102.01"/>
<text xml:space="preserve" text-anchor="start" x="2308.77" y="-1048" font-family="Arial" font-size="20.00" fill="#ffe0c2">commands</text>
<text xml:space="preserve" text-anchor="start" x="2183.27" y="-1020" font-family="Arial" font-size="15.00" fill="#f9b27c">StartOptimizationCommand, ShutdownCommand. 4</text>
<text xml:space="preserve" text-anchor="start" x="2324.27" y="-1002" font-family="Arial" font-size="15.00" fill="#f9b27c">partitions.</text>
</g>
<!-- dcoptimizer -->
<g id="node9" class="node">
<title>dcoptimizer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3307.4,-1260 2952.29,-1260 2952.29,-1080 3307.4,-1080 3307.4,-1260"/>
<text xml:space="preserve" text-anchor="start" x="3070.34" y="-1238.8" font-family="Arial" font-size="20.00" fill="#eff6ff">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="3076.84" y="-1210.8" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="2984.34" y="-1191.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="3070.84" y="-1173.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="2972.34" y="-1155.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="3055.84" y="-1137.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="2975.34" y="-1119.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acvalidator -->
<g id="node10" class="node">
<title>acvalidator</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3298.9,-798 2960.79,-798 2960.79,-618 3298.9,-618 3298.9,-798"/>
<text xml:space="preserve" text-anchor="start" x="3074.34" y="-776.8" font-family="Arial" font-size="20.00" fill="#eff6ff">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="3028.34" y="-748.8" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="2995.84" y="-729.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="3044.34" y="-711.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="2980.84" y="-693.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="3054.84" y="-675.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="3000.84" y="-657.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">candidate it evaluates gets a full result</text>
</g>
<!-- results -->
<g id="node11" class="node">
<title>results</title>
<polygon fill="#a35829" stroke="#7e451d" stroke-width="0" points="4082.43,-790.01 3736.4,-790.01 3736.4,-618 4082.43,-618 4082.43,-790.01"/>
<text xml:space="preserve" text-anchor="start" x="3880.92" y="-754" font-family="Arial" font-size="20.00" fill="#ffe0c2">results</text>
<text xml:space="preserve" text-anchor="start" x="3767.92" y="-726" font-family="Arial" font-size="15.00" fill="#f9b27c">The one shared topic. Both stages publish</text>
<text xml:space="preserve" text-anchor="start" x="3830.42" y="-708" font-family="Arial" font-size="15.00" fill="#f9b27c">topologies here and the</text>
<text xml:space="preserve" text-anchor="start" x="3760.42" y="-690" font-family="Arial" font-size="15.00" fill="#f9b27c">AC&#45;Validator also consumes it to pick up DC</text>
<text xml:space="preserve" text-anchor="start" x="3870.92" y="-672" font-family="Arial" font-size="15.00" fill="#f9b27c">candidates.</text>
</g>
<!-- downstream -->
<g id="node12" class="node">
<title>downstream</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="4696.54,-397 4374.43,-397 4374.43,-217 4696.54,-217 4696.54,-397"/>
<text xml:space="preserve" text-anchor="start" x="4394.49" y="-348" font-family="Arial" font-size="20.00" fill="#f8fafc">Frontend / downstream systems</text>
<text xml:space="preserve" text-anchor="start" x="4397.99" y="-320" font-family="Arial" font-size="15.00" fill="#cbd5e1">Where an operator reviews the proposed</text>
<text xml:space="preserve" text-anchor="start" x="4423.99" y="-302" font-family="Arial" font-size="15.00" fill="#cbd5e1">actions and exports the accepted</text>
<text xml:space="preserve" text-anchor="start" x="4427.99" y="-284" font-family="Arial" font-size="15.00" fill="#cbd5e1">ones. Not part of this repository.</text>
</g>
<!-- client&#45;&gt;importercommands -->
<g id="edge1" class="edge">
<title>client&#45;&gt;importercommands</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M338.04,-661C436.98,-661 562.94,-661 668,-661"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="667.89,-663.63 675.39,-661 667.89,-658.38 667.89,-663.63"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="401.11,-664 401.11,-696.8 425.11,-696.8 425.11,-664 401.11,-664"/>
<text xml:space="preserve" text-anchor="start" x="409.61" y="-676.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">0</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="428.11,-664 428.11,-696.8 615.11,-696.8 615.11,-664 428.11,-664"/>
<text xml:space="preserve" text-anchor="start" x="431.11" y="-688.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">StartPreprocessingCommand</text>
</g>
<!-- client&#45;&gt;importerresults -->
<g id="edge8" class="edge">
<title>client&#45;&gt;importerresults</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M245.68,-563.31C335.63,-455.38 496.61,-286.71 678.11,-210 1195.62,8.71 1882.64,-31.2 2192.45,-64.6"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="243.76,-561.52 240.99,-568.97 247.8,-564.87 243.76,-561.52"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1188.14,-85.3 1188.14,-118.1 1212.14,-118.1 1212.14,-85.3 1188.14,-85.3"/>
<text xml:space="preserve" text-anchor="start" x="1196.64" y="-97.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">7</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1215.14,-85.3 1215.14,-118.1 1338.14,-118.1 1338.14,-85.3 1215.14,-85.3"/>
<text xml:space="preserve" text-anchor="start" x="1218.14" y="-110.1" font-family="Arial" font-size="14.00" fill="#c9c9c9">data folder is ready</text>
</g>
<!-- client&#45;&gt;commands -->
<g id="edge9" class="edge">
<title>client&#45;&gt;commands</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M337.86,-714.54C436.13,-744.13 563.15,-779.38 678.11,-802 1203.2,-905.32 1830.97,-969.98 2149.24,-998.67"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2148.78,-1001.26 2156.49,-999.32 2149.25,-996.03 2148.78,-1001.26"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1161.64,-917.54 1161.64,-950.34 1185.64,-950.34 1185.64,-917.54 1161.64,-917.54"/>
<text xml:space="preserve" text-anchor="start" x="1170.14" y="-930.04" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">8</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1188.64,-917.54 1188.64,-950.34 1364.64,-950.34 1364.64,-917.54 1188.64,-917.54"/>
<text xml:space="preserve" text-anchor="start" x="1191.64" y="-942.34" font-family="Arial" font-size="14.00" fill="#c9c9c9">StartOptimizationCommand</text>
</g>
<!-- importercommands&#45;&gt;importer -->
<g id="edge2" class="edge">
<title>importercommands&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1073.86,-610.73C1188.51,-581.43 1331.81,-544.81 1442.38,-516.55"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1442.75,-519.17 1449.37,-514.77 1441.45,-514.08 1442.75,-519.17"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1198.64,-597.48 1198.64,-630.28 1222.64,-630.28 1222.64,-597.48 1198.64,-597.48"/>
<text xml:space="preserve" text-anchor="start" x="1207.14" y="-609.98" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">1</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1225.64,-597.48 1225.64,-630.28 1327.64,-630.28 1327.64,-597.48 1225.64,-597.48"/>
<text xml:space="preserve" text-anchor="start" x="1228.64" y="-622.28" font-family="Arial" font-size="14.00" fill="#c9c9c9">picks up the job</text>
</g>
<!-- unprocessedgridstore&#45;&gt;importer -->
<g id="edge3" class="edge">
<title>unprocessedgridstore&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1038.07,-373.87C1140.07,-381.1 1274.48,-393.57 1392.14,-414.2 1408.55,-417.08 1425.52,-420.54 1442.44,-424.31"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1441.55,-426.8 1449.44,-425.9 1442.71,-421.68 1441.55,-426.8"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1137.14,-417.2 1137.14,-450 1161.14,-450 1161.14,-417.2 1137.14,-417.2"/>
<text xml:space="preserve" text-anchor="start" x="1145.64" y="-429.7" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">2</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1164.14,-417.2 1164.14,-450 1389.14,-450 1389.14,-417.2 1164.14,-417.2"/>
<text xml:space="preserve" text-anchor="start" x="1167.14" y="-442" font-family="Arial" font-size="14.00" fill="#c9c9c9">UCTE / CGMES / PowerFactory file</text>
</g>
<!-- importer&#45;&gt;processedgrid -->
<g id="edge4" class="edge">
<title>importer&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1783.02,-462.36C1875.98,-456.15 1994.3,-446.99 2099.26,-435 2127.43,-431.79 2157.26,-427.8 2186.24,-423.61"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2186.41,-426.24 2193.45,-422.56 2185.65,-421.04 2186.41,-426.24"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1846.76,-460.62 1846.76,-500.22 1870.76,-500.22 1870.76,-460.62 1846.76,-460.62"/>
<text xml:space="preserve" text-anchor="start" x="1855.26" y="-476.52" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">3</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1873.76,-460.62 1873.76,-500.22 2095.76,-500.22 2095.76,-460.62 1873.76,-460.62"/>
<text xml:space="preserve" text-anchor="start" x="1876.76" y="-497.22" font-family="Arial" font-size="14.00" fill="#c9c9c9">normalized snapshot, masks, asset</text>
<text xml:space="preserve" text-anchor="start" x="1958.76" y="-480.42" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- importer&#45;&gt;processedgrid -->
<g id="edge5" class="edge">
<title>importer&#45;&gt;processedgrid</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1783.02,-387.26C1802.9,-379.88 1823.25,-373.57 1843.26,-369.2 1955.83,-344.66 2085.43,-351.71 2186.21,-364.72"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2185.7,-367.3 2193.48,-365.69 2186.39,-362.1 2185.7,-367.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1852.26,-372.2 1852.26,-405 1876.26,-405 1876.26,-372.2 1852.26,-372.2"/>
<text xml:space="preserve" text-anchor="start" x="1860.76" y="-384.7" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">4</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1879.26,-372.2 1879.26,-405 2090.26,-405 2090.26,-372.2 1879.26,-372.2"/>
<text xml:space="preserve" text-anchor="start" x="1882.26" y="-397" font-family="Arial" font-size="14.00" fill="#c9c9c9">PTDF, action set, contingency set</text>
</g>
<!-- importer&#45;&gt;loadflowstore -->
<g id="edge6" class="edge">
<title>importer&#45;&gt;loadflowstore</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1782.85,-524.53C1896.08,-560.76 2046.98,-609.04 2165.55,-646.98"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2164.65,-649.45 2172.59,-649.23 2166.25,-644.45 2164.65,-649.45"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1846.26,-627.93 1846.26,-660.73 1870.26,-660.73 1870.26,-627.93 1846.26,-627.93"/>
<text xml:space="preserve" text-anchor="start" x="1854.76" y="-640.43" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">5</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1873.26,-627.93 1873.26,-660.73 2096.26,-660.73 2096.26,-627.93 1873.26,-627.93"/>
<text xml:space="preserve" text-anchor="start" x="1876.26" y="-652.73" font-family="Arial" font-size="14.00" fill="#c9c9c9">initial AC N&#45;1 and reference metrics</text>
</g>
<!-- importer&#45;&gt;importerresults -->
<g id="edge7" class="edge">
<title>importer&#45;&gt;importerresults</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1714.3,-382.05C1752.43,-349.29 1797.98,-313.8 1843.26,-287.2 1950.14,-224.41 2080.25,-173.64 2182.69,-138.81"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2183.38,-141.34 2189.65,-136.46 2181.7,-136.37 2183.38,-141.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1864.76,-290.2 1864.76,-323 1888.76,-323 1888.76,-290.2 1864.76,-290.2"/>
<text xml:space="preserve" text-anchor="start" x="1873.26" y="-302.7" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">6</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1891.76,-290.2 1891.76,-323 2077.76,-323 2077.76,-290.2 1891.76,-290.2"/>
<text xml:space="preserve" text-anchor="start" x="1894.76" y="-315" font-family="Arial" font-size="14.00" fill="#c9c9c9">PreprocessingSuccessResult</text>
</g>
<!-- processedgrid&#45;&gt;dcoptimizer -->
<g id="edge12" class="edge">
<title>processedgrid&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2511.79,-497.06C2528.72,-513.91 2543.91,-532.62 2555.29,-553 2663.91,-747.68 2466.37,-885.1 2615.29,-1051 2660.15,-1100.98 2822.33,-1132.76 2952.59,-1150.79"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2513.83,-495.38 2506.61,-492.06 2510.18,-499.16 2513.83,-495.38"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2619.29,-1143.88 2619.29,-1176.68 2650.29,-1176.68 2650.29,-1143.88 2619.29,-1143.88"/>
<text xml:space="preserve" text-anchor="start" x="2627.29" y="-1156.38" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">11</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2653.29,-1143.88 2653.29,-1176.68 2888.29,-1176.68 2888.29,-1143.88 2653.29,-1143.88"/>
<text xml:space="preserve" text-anchor="start" x="2656.29" y="-1168.68" font-family="Arial" font-size="14.00" fill="#c9c9c9">loads static information onto the GPU</text>
</g>
<!-- processedgrid&#45;&gt;acvalidator -->
<g id="edge13" class="edge">
<title>processedgrid&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2506.69,-497.45C2524.81,-514.42 2541.7,-533.06 2555.29,-553 2606.36,-627.93 2542.35,-694.13 2615.29,-748 2713.21,-820.33 2852.2,-804.43 2960.9,-773.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2508.94,-495.96 2501.64,-492.83 2505.4,-499.83 2508.94,-495.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2642.29,-802.58 2642.29,-835.38 2673.29,-835.38 2673.29,-802.58 2642.29,-802.58"/>
<text xml:space="preserve" text-anchor="start" x="2650.29" y="-815.08" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">12</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2676.29,-802.58 2676.29,-835.38 2865.29,-835.38 2865.29,-802.58 2676.29,-802.58"/>
<text xml:space="preserve" text-anchor="start" x="2679.29" y="-827.38" font-family="Arial" font-size="14.00" fill="#c9c9c9">loads base grid and action set</text>
</g>
<!-- processedgrid&#45;&gt;acvalidator -->
<g id="edge20" class="edge">
<title>processedgrid&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2528.43,-418.94C2636.48,-437.92 2777.16,-470.99 2892.29,-526.2 2941.68,-549.89 2990.63,-585.03 3031.07,-618.12"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2528.89,-416.35 2521.06,-417.66 2528,-421.53 2528.89,-416.35"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2618.29,-529.2 2618.29,-562 2649.29,-562 2649.29,-529.2 2618.29,-529.2"/>
<text xml:space="preserve" text-anchor="start" x="2626.29" y="-541.7" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">19</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2652.29,-529.2 2652.29,-562 2889.29,-562 2889.29,-529.2 2652.29,-529.2"/>
<text xml:space="preserve" text-anchor="start" x="2655.29" y="-554" font-family="Arial" font-size="14.00" fill="#c9c9c9">summaries, diagrams, loadflow tables</text>
</g>
<!-- processedgrid&#45;&gt;downstream -->
<g id="edge22" class="edge">
<title>processedgrid&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2517.85,-389.48C2916.63,-373.17 3953.94,-330.75 4364.05,-313.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4364.1,-316.6 4371.49,-313.67 4363.89,-311.35 4364.1,-316.6"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3370.4,-357.58 3370.4,-397.18 3401.4,-397.18 3401.4,-357.58 3370.4,-357.58"/>
<text xml:space="preserve" text-anchor="start" x="3378.4" y="-373.48" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">21</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3404.4,-357.58 3404.4,-397.18 3673.4,-397.18 3673.4,-357.58 3404.4,-357.58"/>
<text xml:space="preserve" text-anchor="start" x="3407.4" y="-394.18" font-family="Arial" font-size="14.00" fill="#c9c9c9">UCTE, DGS, OpenRAO summaries, single</text>
<text xml:space="preserve" text-anchor="start" x="3497.9" y="-377.38" font-family="Arial" font-size="14.00" fill="#c9c9c9">line diagrams</text>
</g>
<!-- loadflowstore&#45;&gt;acvalidator -->
<g id="edge14" class="edge">
<title>loadflowstore&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2539.29,-690.72C2564.78,-688.85 2590.68,-687.25 2615.29,-686.2 2738.29,-681 2769.31,-680.55 2892.29,-686.2 2911.23,-687.08 2931,-688.34 2950.69,-689.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2950.28,-692.42 2957.96,-690.38 2950.69,-687.19 2950.28,-692.42"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2623.29,-689.2 2623.29,-722 2654.29,-722 2654.29,-689.2 2623.29,-689.2"/>
<text xml:space="preserve" text-anchor="start" x="2631.29" y="-701.7" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">13</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2657.29,-689.2 2657.29,-722 2884.29,-722 2884.29,-689.2 2657.29,-689.2"/>
<text xml:space="preserve" text-anchor="start" x="2660.29" y="-714" font-family="Arial" font-size="14.00" fill="#c9c9c9">reads the initial loadflow as baseline</text>
</g>
<!-- loadflowstore&#45;&gt;acvalidator -->
<g id="edge18" class="edge">
<title>loadflowstore&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2548.88,-626.16C2570.94,-619.31 2593.39,-613.46 2615.29,-609.4 2736.34,-586.98 2771.57,-585.25 2892.29,-609.4 2914.98,-613.94 2938.21,-620.71 2960.82,-628.6"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2548.35,-623.57 2542,-628.35 2549.94,-628.58 2548.35,-623.57"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2630.79,-612.4 2630.79,-652 2661.79,-652 2661.79,-612.4 2630.79,-612.4"/>
<text xml:space="preserve" text-anchor="start" x="2638.79" y="-628.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">17</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2664.79,-612.4 2664.79,-652 2876.79,-652 2876.79,-612.4 2664.79,-612.4"/>
<text xml:space="preserve" text-anchor="start" x="2667.79" y="-649" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC loadflow results per evaluated</text>
<text xml:space="preserve" text-anchor="start" x="2744.79" y="-632.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">topology</text>
</g>
<!-- commands&#45;&gt;dcoptimizer -->
<g id="edge10" class="edge">
<title>commands&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2442.37,-1101.76C2488.96,-1143.08 2550.48,-1188.29 2615.29,-1210 2719.92,-1245.06 2842.99,-1237.03 2942.42,-1219.3"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2942.7,-1221.92 2949.6,-1217.99 2941.75,-1216.76 2942.7,-1221.92"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2685.29,-1237.44 2685.29,-1270.24 2709.29,-1270.24 2709.29,-1237.44 2685.29,-1237.44"/>
<text xml:space="preserve" text-anchor="start" x="2693.79" y="-1249.94" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">9</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2712.29,-1237.44 2712.29,-1270.24 2822.29,-1270.24 2822.29,-1237.44 2712.29,-1237.44"/>
<text xml:space="preserve" text-anchor="start" x="2715.29" y="-1262.24" font-family="Arial" font-size="14.00" fill="#c9c9c9">starts the DC run</text>
</g>
<!-- commands&#45;&gt;acvalidator -->
<g id="edge11" class="edge">
<title>commands&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2555.21,-996.71C2659.32,-980.98 2787.01,-952.68 2892.29,-902 2942.98,-877.6 2992.22,-840.18 3032.53,-804.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3034.1,-806.82 3037.95,-799.87 3030.61,-802.9 3034.1,-806.82"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2682.29,-988 2682.29,-1020.8 2713.29,-1020.8 2713.29,-988 2682.29,-988"/>
<text xml:space="preserve" text-anchor="start" x="2690.29" y="-1000.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">10</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2716.29,-988 2716.29,-1020.8 2825.29,-1020.8 2825.29,-988 2716.29,-988"/>
<text xml:space="preserve" text-anchor="start" x="2719.29" y="-1012.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">starts the AC run</text>
</g>
<!-- dcoptimizer&#45;&gt;results -->
<g id="edge15" class="edge">
<title>dcoptimizer&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3280.77,-1080.15C3417.55,-998.18 3618.74,-877.61 3756.23,-795.21"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3757.51,-797.5 3762.59,-791.4 3754.81,-793 3757.51,-797.5"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3425.4,-1022.27 3425.4,-1055.07 3456.4,-1055.07 3456.4,-1022.27 3425.4,-1022.27"/>
<text xml:space="preserve" text-anchor="start" x="3433.4" y="-1034.77" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">14</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3459.4,-1022.27 3459.4,-1055.07 3618.4,-1055.07 3618.4,-1022.27 3459.4,-1022.27"/>
<text xml:space="preserve" text-anchor="start" x="3462.4" y="-1047.07" font-family="Arial" font-size="14.00" fill="#c9c9c9">Strategy, once per epoch</text>
</g>
<!-- acvalidator&#45;&gt;acvalidator -->
<g id="edge17" class="edge">
<title>acvalidator&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3064.46,-797.99C3048.37,-854.28 3070.17,-908 3129.84,-908 3185.91,-908 3208.54,-860.59 3197.74,-808.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3200.3,-807.58 3195.96,-800.93 3195.2,-808.84 3200.3,-807.58"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3025.34,-911 3025.34,-943.8 3056.34,-943.8 3056.34,-911 3025.34,-911"/>
<text xml:space="preserve" text-anchor="start" x="3033.34" y="-923.5" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">16</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3059.34,-911 3059.34,-943.8 3234.34,-943.8 3234.34,-911 3059.34,-911"/>
<text xml:space="preserve" text-anchor="start" x="3062.34" y="-935.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">prune, worst&#45;k, then full N&#45;1</text>
</g>
<!-- acvalidator&#45;&gt;results -->
<g id="edge16" class="edge">
<title>acvalidator&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3308.94,-707.09C3437.33,-706.43 3609.65,-705.54 3736.58,-704.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3309.12,-704.46 3301.64,-707.13 3309.15,-709.71 3309.12,-704.46"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3424.9,-709.71 3424.9,-742.51 3455.9,-742.51 3455.9,-709.71 3424.9,-709.71"/>
<text xml:space="preserve" text-anchor="start" x="3432.9" y="-722.21" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">15</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3458.9,-709.71 3458.9,-742.51 3618.9,-742.51 3618.9,-709.71 3458.9,-709.71"/>
<text xml:space="preserve" text-anchor="start" x="3461.9" y="-734.51" font-family="Arial" font-size="14.00" fill="#c9c9c9">DC topologies to validate</text>
</g>
<!-- acvalidator&#45;&gt;results -->
<g id="edge19" class="edge">
<title>acvalidator&#45;&gt;results</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3298.59,-647.42C3321.45,-641.23 3344.82,-635.92 3367.4,-632.4 3503.09,-611.25 3540.6,-611.93 3676.4,-632.4 3692.86,-634.89 3709.77,-638.31 3726.59,-642.32"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3725.63,-644.79 3733.54,-644.02 3726.88,-639.69 3725.63,-644.79"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3387.9,-635.4 3387.9,-675 3418.9,-675 3418.9,-635.4 3387.9,-635.4"/>
<text xml:space="preserve" text-anchor="start" x="3395.9" y="-651.3" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">18</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3421.9,-635.4 3421.9,-675 3655.9,-675 3655.9,-635.4 3421.9,-635.4"/>
<text xml:space="preserve" text-anchor="start" x="3424.9" y="-672" font-family="Arial" font-size="14.00" fill="#c9c9c9">AC&#45;validated Strategy, referencing its</text>
<text xml:space="preserve" text-anchor="start" x="3513.9" y="-655.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">loadflow</text>
</g>
<!-- results&#45;&gt;downstream -->
<g id="edge21" class="edge">
<title>results&#45;&gt;downstream</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4045.45,-618.11C4145.32,-554.57 4281.35,-468.04 4384.59,-402.36"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4385.74,-404.74 4390.66,-398.5 4382.92,-400.31 4385.74,-404.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4145.43,-555.74 4145.43,-588.54 4176.43,-588.54 4176.43,-555.74 4145.43,-555.74"/>
<text xml:space="preserve" text-anchor="start" x="4153.43" y="-568.24" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">20</text>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4179.43,-555.74 4179.43,-588.54 4311.43,-588.54 4311.43,-555.74 4179.43,-555.74"/>
<text xml:space="preserve" text-anchor="start" x="4182.43" y="-580.54" font-family="Arial" font-size="14.00" fill="#c9c9c9">topologies for review</text>
</g>
</g>
</svg>`;case`parameters`:return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 15.1.0 (20260618.0150)
 -->
<!-- Pages: 1 -->
<svg width="2522pt" height="1768pt"
 viewBox="0.00 0.00 2522.00 1768.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1753.45)">
<g id="clust1" class="cluster">
<title>cluster_toop</title>
<polygon fill="#3a404a" stroke="#292f37" points="8,-8 8,-1550.4 2384,-1550.4 2384,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1536.95" font-family="Arial" font-weight="bold" font-size="11.00" fill="#cbd5e1" fill-opacity="0.701961">TOOP ENGINE</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_importerparams</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="48,-908 48,-1489.2 1408,-1489.2 1408,-908 48,-908"/>
<text xml:space="preserve" text-anchor="start" x="56" y="-1475.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">IMPORTER PARAMETERS</text>
</g>
<g id="clust5" class="cluster">
<title>cluster_dcparams</title>
<polygon fill="#2225aa" stroke="#2a2490" points="1455,-308 1455,-889.2 2344,-889.2 2344,-308 1455,-308"/>
<text xml:space="preserve" text-anchor="start" x="1463" y="-875.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#c7d2fe" fill-opacity="0.701961">DC OPTIMIZER PARAMETERS</text>
</g>
<g id="clust7" class="cluster">
<title>cluster_acparams</title>
<polygon fill="#603329" stroke="#4b2720" points="1448,-908 1448,-1489.2 2344,-1489.2 2344,-908 1448,-908"/>
<text xml:space="preserve" text-anchor="start" x="1456" y="-1475.75" font-family="Arial" font-weight="bold" font-size="11.00" fill="#f5b2a3" fill-opacity="0.701961">AC VALIDATOR PARAMETERS</text>
</g>
<!-- pareasettings -->
<g id="node1" class="node">
<title>pareasettings</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="451.02,-1428 130.98,-1428 130.98,-1248 451.02,-1248 451.02,-1428"/>
<text xml:space="preserve" text-anchor="start" x="234.5" y="-1397" font-family="Arial" font-size="20.00" fill="#eff6ff">AreaSettings</text>
<text xml:space="preserve" text-anchor="start" x="179.5" y="-1369" font-family="Arial" font-size="15.00" fill="#bfdbfe">control_area and view_area, plus</text>
<text xml:space="preserve" text-anchor="start" x="176" y="-1351" font-family="Arial" font-size="15.00" fill="#bfdbfe">cutoff_voltage (220 kV by default).</text>
<text xml:space="preserve" text-anchor="start" x="166.5" y="-1333" font-family="Arial" font-size="15.00" fill="#bfdbfe">Also where the limit adjustments live:</text>
<text xml:space="preserve" text-anchor="start" x="217.5" y="-1315" font-family="Arial" font-size="15.00" fill="#bfdbfe">dso_trafo_factors and</text>
<text xml:space="preserve" text-anchor="start" x="199" y="-1297" font-family="Arial" font-size="15.00" fill="#bfdbfe">border_line_factors, each a</text>
</g>
<!-- pstationrules -->
<g id="node2" class="node">
<title>pstationrules</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="910.06,-1428 581.94,-1428 581.94,-1248 910.06,-1248 910.06,-1428"/>
<text xml:space="preserve" text-anchor="start" x="651" y="-1397" font-family="Arial" font-size="20.00" fill="#eff6ff">RelevantStationRules</text>
<text xml:space="preserve" text-anchor="start" x="617" y="-1369" font-family="Arial" font-size="15.00" fill="#bfdbfe">What makes a station worth switching:</text>
<text xml:space="preserve" text-anchor="start" x="689.5" y="-1351" font-family="Arial" font-size="15.00" fill="#bfdbfe">min_busbars (2),</text>
<text xml:space="preserve" text-anchor="start" x="646" y="-1333" font-family="Arial" font-size="15.00" fill="#bfdbfe">min_connected_branches (4),</text>
<text xml:space="preserve" text-anchor="start" x="602" y="-1315" font-family="Arial" font-size="15.00" fill="#bfdbfe">min_connected_elements (4). This decides</text>
<text xml:space="preserve" text-anchor="start" x="634.5" y="-1297" font-family="Arial" font-size="15.00" fill="#bfdbfe">the set of switchable substations.</text>
</g>
<!-- plists -->
<g id="node3" class="node">
<title>plists</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1340.02,-1428 1019.98,-1428 1019.98,-1248 1340.02,-1248 1340.02,-1428"/>
<text xml:space="preserve" text-anchor="start" x="1067.5" y="-1397" font-family="Arial" font-size="20.00" fill="#eff6ff">White / black / ignore lists</text>
<text xml:space="preserve" text-anchor="start" x="1083.5" y="-1369" font-family="Arial" font-size="15.00" fill="#bfdbfe">white_list_file, black_list_file,</text>
<text xml:space="preserve" text-anchor="start" x="1117" y="-1351" font-family="Arial" font-size="15.00" fill="#bfdbfe">ignore_list_file and</text>
<text xml:space="preserve" text-anchor="start" x="1042" y="-1333" font-family="Arial" font-size="15.00" fill="#bfdbfe">select_by_voltage_level_id_list. Operator</text>
<text xml:space="preserve" text-anchor="start" x="1104.5" y="-1315" font-family="Arial" font-size="15.00" fill="#bfdbfe">overrides on top of the</text>
<text xml:space="preserve" text-anchor="start" x="1065" y="-1297" font-family="Arial" font-size="15.00" fill="#bfdbfe">area rules, applied during importer</text>
</g>
<!-- pcontingencies -->
<g id="node4" class="node">
<title>pcontingencies</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="451.56,-1128 88.44,-1128 88.44,-948 451.56,-948 451.56,-1128"/>
<text xml:space="preserve" text-anchor="start" x="200" y="-1097" font-family="Arial" font-size="20.00" fill="#eff6ff">Contingency list</text>
<text xml:space="preserve" text-anchor="start" x="120" y="-1069" font-family="Arial" font-size="15.00" fill="#bfdbfe">contingency_list_file plus its schema_format,</text>
<text xml:space="preserve" text-anchor="start" x="189.5" y="-1051" font-family="Arial" font-size="15.00" fill="#bfdbfe">either the PowerFactory</text>
<text xml:space="preserve" text-anchor="start" x="108.5" y="-1033" font-family="Arial" font-size="15.00" fill="#bfdbfe">import schema or the generic one. Becomes the</text>
<text xml:space="preserve" text-anchor="start" x="196" y="-1015" font-family="Arial" font-size="15.00" fill="#bfdbfe">N&#45;1 definition. Without</text>
<text xml:space="preserve" text-anchor="start" x="145.5" y="-997" font-family="Arial" font-size="15.00" fill="#bfdbfe">it, contingencies are derived from the</text>
</g>
<!-- psecurityanalysis -->
<g id="node5" class="node">
<title>psecurityanalysis</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1368.06,-1128 991.94,-1128 991.94,-948 1368.06,-948 1368.06,-1128"/>
<text xml:space="preserve" text-anchor="start" x="1076.5" y="-1097" font-family="Arial" font-size="20.00" fill="#eff6ff">Import security analysis</text>
<text xml:space="preserve" text-anchor="start" x="1012" y="-1069" font-family="Arial" font-size="15.00" fill="#bfdbfe">n_contingency_worker (8 OpenLoadFlow threads)</text>
<text xml:space="preserve" text-anchor="start" x="1167.5" y="-1051" font-family="Arial" font-size="15.00" fill="#bfdbfe">and</text>
<text xml:space="preserve" text-anchor="start" x="1030.5" y="-1033" font-family="Arial" font-size="15.00" fill="#bfdbfe">contingency_batch_size (all outages at once</text>
<text xml:space="preserve" text-anchor="start" x="1101.5" y="-1015" font-family="Arial" font-size="15.00" fill="#bfdbfe">by default) size the one</text>
<text xml:space="preserve" text-anchor="start" x="1048.5" y="-997" font-family="Arial" font-size="15.00" fill="#bfdbfe">AC N&#45;1 of the import. Its results set the</text>
</g>
<!-- ppreprocess -->
<g id="node6" class="node">
<title>ppreprocess</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="882.06,-1128 561.94,-1128 561.94,-948 882.06,-948 882.06,-1128"/>
<text xml:space="preserve" text-anchor="start" x="620.5" y="-1097" font-family="Arial" font-size="20.00" fill="#eff6ff">PreprocessParameters</text>
<text xml:space="preserve" text-anchor="start" x="593" y="-1069" font-family="Arial" font-size="15.00" fill="#bfdbfe">How hard to work on the action space.</text>
<text xml:space="preserve" text-anchor="start" x="623" y="-1051" font-family="Arial" font-size="15.00" fill="#bfdbfe">action_set_clip caps a station</text>
<text xml:space="preserve" text-anchor="start" x="646" y="-1033" font-family="Arial" font-size="15.00" fill="#bfdbfe">at 2^23 configurations;</text>
<text xml:space="preserve" text-anchor="start" x="602.5" y="-1015" font-family="Arial" font-size="15.00" fill="#bfdbfe">action_set_filter_bridge_lookup and</text>
<text xml:space="preserve" text-anchor="start" x="582" y="-997" font-family="Arial" font-size="15.00" fill="#bfdbfe">action_set_filter_bsdf_lodf drop splits that</text>
</g>
<!-- pme -->
<g id="node7" class="node">
<title>pme</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="2303.56,-828 1954.44,-828 1954.44,-648 2303.56,-648 2303.56,-828"/>
<text xml:space="preserve" text-anchor="start" x="1974.5" y="-797" font-family="Arial" font-size="20.00" fill="#eef2ff">BatchedMEParameters (ga_config)</text>
<text xml:space="preserve" text-anchor="start" x="1996.5" y="-769" font-family="Arial" font-size="15.00" fill="#c7d2fe">The search itself. runtime_seconds and</text>
<text xml:space="preserve" text-anchor="start" x="2033.5" y="-751" font-family="Arial" font-size="15.00" fill="#c7d2fe">iterations_per_epoch set the</text>
<text xml:space="preserve" text-anchor="start" x="1989.5" y="-733" font-family="Arial" font-size="15.00" fill="#c7d2fe">budget and how often results are pushed.</text>
<text xml:space="preserve" text-anchor="start" x="2066.5" y="-715" font-family="Arial" font-size="15.00" fill="#c7d2fe">target_metrics and</text>
<text xml:space="preserve" text-anchor="start" x="2002.5" y="-697" font-family="Arial" font-size="15.00" fill="#c7d2fe">observed_metrics define the fitness &#45;&#45;</text>
</g>
<!-- psolver -->
<g id="node8" class="node">
<title>psolver</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="1844.56,-828 1495.44,-828 1495.44,-648 1844.56,-648 1844.56,-828"/>
<text xml:space="preserve" text-anchor="start" x="1551.5" y="-797" font-family="Arial" font-size="20.00" fill="#eef2ff">LoadflowSolverParameters</text>
<text xml:space="preserve" text-anchor="start" x="1515.5" y="-769" font-family="Arial" font-size="15.00" fill="#c7d2fe">The shape of the search space and the batch.</text>
<text xml:space="preserve" text-anchor="start" x="1589.5" y="-751" font-family="Arial" font-size="15.00" fill="#c7d2fe">max_num_splits (4) and</text>
<text xml:space="preserve" text-anchor="start" x="1524" y="-733" font-family="Arial" font-size="15.00" fill="#c7d2fe">max_num_disconnections cap the genome;</text>
<text xml:space="preserve" text-anchor="start" x="1581" y="-715" font-family="Arial" font-size="15.00" fill="#c7d2fe">batch_size sets how many</text>
<text xml:space="preserve" text-anchor="start" x="1540.5" y="-697" font-family="Arial" font-size="15.00" fill="#c7d2fe">topologies the GPU evaluates at once;</text>
</g>
<!-- pdoublelimits -->
<g id="node9" class="node">
<title>pdoublelimits</title>
<polygon fill="#6366f1" stroke="#4f46e5" stroke-width="0" points="2301.56,-528 1956.44,-528 1956.44,-348 2301.56,-348 2301.56,-528"/>
<text xml:space="preserve" text-anchor="start" x="2072.5" y="-479" font-family="Arial" font-size="20.00" fill="#eef2ff">DoubleLimits</text>
<text xml:space="preserve" text-anchor="start" x="1976.5" y="-451" font-family="Arial" font-size="15.00" fill="#c7d2fe">Optional. Separate permanent and temporary</text>
<text xml:space="preserve" text-anchor="start" x="2045" y="-433" font-family="Arial" font-size="15.00" fill="#c7d2fe">branch limits, so N&#45;0 and</text>
<text xml:space="preserve" text-anchor="start" x="1984.5" y="-415" font-family="Arial" font-size="15.00" fill="#c7d2fe">N&#45;1 can be judged against different ratings.</text>
</g>
<!-- pacga -->
<g id="node10" class="node">
<title>pacga</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1846.06,-1428 1487.94,-1428 1487.94,-1248 1846.06,-1248 1846.06,-1428"/>
<text xml:space="preserve" text-anchor="start" x="1535" y="-1397" font-family="Arial" font-size="20.00" fill="#fbd3cb">ACGAParameters (ga_config)</text>
<text xml:space="preserve" text-anchor="start" x="1575" y="-1369" font-family="Arial" font-size="15.00" fill="#f5b2a3">runtime_seconds (180) and</text>
<text xml:space="preserve" text-anchor="start" x="1530" y="-1351" font-family="Arial" font-size="15.00" fill="#f5b2a3">max_initial_wait_seconds bound the run;</text>
<text xml:space="preserve" text-anchor="start" x="1508" y="-1333" font-family="Arial" font-size="15.00" fill="#f5b2a3">runner_processes, contingency_processes and</text>
<text xml:space="preserve" text-anchor="start" x="1624" y="-1315" font-family="Arial" font-size="15.00" fill="#f5b2a3">their worst_k</text>
<text xml:space="preserve" text-anchor="start" x="1541" y="-1297" font-family="Arial" font-size="15.00" fill="#f5b2a3">counterparts set the CPU parallelism,</text>
</g>
<!-- prejection -->
<g id="node11" class="node">
<title>prejection</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="2303.56,-1428 1956.44,-1428 1956.44,-1248 2303.56,-1248 2303.56,-1428"/>
<text xml:space="preserve" text-anchor="start" x="2040.5" y="-1397" font-family="Arial" font-size="20.00" fill="#fbd3cb">Rejection thresholds</text>
<text xml:space="preserve" text-anchor="start" x="1976.5" y="-1369" font-family="Arial" font-size="15.00" fill="#f5b2a3">What counts as a failure. enable_ac_rejection</text>
<text xml:space="preserve" text-anchor="start" x="2058.5" y="-1351" font-family="Arial" font-size="15.00" fill="#f5b2a3">switches the gate on,</text>
<text xml:space="preserve" text-anchor="start" x="2002" y="-1333" font-family="Arial" font-size="15.00" fill="#f5b2a3">then reject_overload_threshold (0.95),</text>
<text xml:space="preserve" text-anchor="start" x="2023" y="-1315" font-family="Arial" font-size="15.00" fill="#f5b2a3">reject_critical_branch_threshold</text>
<text xml:space="preserve" text-anchor="start" x="2006.5" y="-1297" font-family="Arial" font-size="15.00" fill="#f5b2a3">(1.1), reject_convergence_threshold,</text>
</g>
<!-- pinitialloadflow -->
<g id="node12" class="node">
<title>pinitialloadflow</title>
<polygon fill="#ac4d39" stroke="#853a2d" stroke-width="0" points="1826.06,-1128 1487.94,-1128 1487.94,-948 1826.06,-948 1826.06,-1128"/>
<text xml:space="preserve" text-anchor="start" x="1548.5" y="-1088" font-family="Arial" font-size="20.00" fill="#fbd3cb">initial_loadflow reference</text>
<text xml:space="preserve" text-anchor="start" x="1508" y="-1060" font-family="Arial" font-size="15.00" fill="#f5b2a3">An optional StoredLoadflowReference to the</text>
<text xml:space="preserve" text-anchor="start" x="1582.5" y="-1042" font-family="Arial" font-size="15.00" fill="#f5b2a3">baseline. If absent the</text>
<text xml:space="preserve" text-anchor="start" x="1511.5" y="-1024" font-family="Arial" font-size="15.00" fill="#f5b2a3">validator computes and stores the initial AC</text>
<text xml:space="preserve" text-anchor="start" x="1625.5" y="-1006" font-family="Arial" font-size="15.00" fill="#f5b2a3">N&#45;1 itself.</text>
</g>
<!-- importer -->
<g id="node13" class="node">
<title>importer</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="887.56,-828 556.44,-828 556.44,-648 887.56,-648 887.56,-828"/>
<text xml:space="preserve" text-anchor="start" x="685.5" y="-806.8" font-family="Arial" font-size="20.00" fill="#f8fafc">Importer</text>
<text xml:space="preserve" text-anchor="start" x="610.5" y="-778.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl, pandapower, JAX</text>
<text xml:space="preserve" text-anchor="start" x="576.5" y="-759.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Normalizes a raw grid into a processed grid</text>
<text xml:space="preserve" text-anchor="start" x="649" y="-741.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">folder and derives the</text>
<text xml:space="preserve" text-anchor="start" x="580" y="-723.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">solver artifacts. Most of it depends only on</text>
<text xml:space="preserve" text-anchor="start" x="645" y="-705.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the initial grid topology,</text>
<text xml:space="preserve" text-anchor="start" x="607.5" y="-687.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">so it can run before the forecast is</text>
</g>
<!-- dcoptimizer -->
<g id="node14" class="node">
<title>dcoptimizer</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="2306.56,-228 1951.44,-228 1951.44,-48 2306.56,-48 2306.56,-228"/>
<text xml:space="preserve" text-anchor="start" x="2069.5" y="-206.8" font-family="Arial" font-size="20.00" fill="#f8fafc">DC&#45;Optimizer</text>
<text xml:space="preserve" text-anchor="start" x="2076" y="-178.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, JAX / XLA</text>
<text xml:space="preserve" text-anchor="start" x="1983.5" y="-159.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Quality&#45;diversity search over the action set.</text>
<text xml:space="preserve" text-anchor="start" x="2070" y="-141.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">The whole loop is</text>
<text xml:space="preserve" text-anchor="start" x="1971.5" y="-123.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">GPU&#45;resident, so no host transfer happens per</text>
<text xml:space="preserve" text-anchor="start" x="2055" y="-105.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">iteration; results leave</text>
<text xml:space="preserve" text-anchor="start" x="1974.5" y="-87.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">only once per epoch. JAX JIT costs about 13s</text>
</g>
<!-- acvalidator -->
<g id="node15" class="node">
<title>acvalidator</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1385.06,-828 1046.94,-828 1046.94,-648 1385.06,-648 1385.06,-828"/>
<text xml:space="preserve" text-anchor="start" x="1160.5" y="-806.8" font-family="Arial" font-size="20.00" fill="#f8fafc">AC&#45;Validator</text>
<text xml:space="preserve" text-anchor="start" x="1114.5" y="-778.8" font-family="Arial" font-size="13.00" fill="#cbd5e1">Python, PyPowSyBl, polars, SQLite</text>
<text xml:space="preserve" text-anchor="start" x="1082" y="-759.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">Proposes no topologies of its own &#45;&#45; it is</text>
<text xml:space="preserve" text-anchor="start" x="1130.5" y="-741.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the quality gate in front of</text>
<text xml:space="preserve" text-anchor="start" x="1067" y="-723.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">the operator. What it does produce is the AC</text>
<text xml:space="preserve" text-anchor="start" x="1141" y="-705.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">loadflow results: every</text>
<text xml:space="preserve" text-anchor="start" x="1087" y="-687.2" font-family="Arial" font-size="15.00" fill="#cbd5e1">candidate it evaluates gets a full result</text>
</g>
<!-- client -->
<g id="node16" class="node">
<title>client</title>
<polygon fill="#64748b" stroke="#475569" stroke-width="0" points="1836.06,-1738.4 1497.94,-1738.4 1497.94,-1558.4 1836.06,-1558.4 1836.06,-1738.4"/>
<text xml:space="preserve" text-anchor="start" x="1536" y="-1707.4" font-family="Arial" font-size="20.00" fill="#f8fafc">Operator / orchestration client</text>
<text xml:space="preserve" text-anchor="start" x="1518" y="-1679.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">Drives the engine either directly from Python</text>
<text xml:space="preserve" text-anchor="start" x="1593.5" y="-1661.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">or by producing Kafka</text>
<text xml:space="preserve" text-anchor="start" x="1526.5" y="-1643.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">commands. ToOp ships no GUI or system</text>
<text xml:space="preserve" text-anchor="start" x="1595" y="-1625.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">integration of its own.</text>
<text xml:space="preserve" text-anchor="start" x="1521" y="-1607.4" font-family="Arial" font-size="15.00" fill="#cbd5e1">In operational use the whole run must finish</text>
</g>
<!-- pareasettings&#45;&gt;pcontingencies -->
<!-- pstationrules&#45;&gt;plists -->
<!-- plists&#45;&gt;psecurityanalysis -->
<!-- psecurityanalysis&#45;&gt;psolver -->
<!-- ppreprocess&#45;&gt;importer -->
<g id="edge10" class="edge">
<title>ppreprocess&#45;&gt;importer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M722,-908C722,-884.62 722,-860.56 722,-838.19"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="724.63,-838.3 722,-830.8 719.38,-838.3 724.63,-838.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="548,-867.77 548,-890.57 722,-890.57 722,-867.77 548,-867.77"/>
<text xml:space="preserve" text-anchor="start" x="551" y="-887.57" font-family="Arial" font-size="14.00" fill="#c9c9c9">scope, limits, contingencies</text>
</g>
<!-- pme&#45;&gt;pdoublelimits -->
<!-- pdoublelimits&#45;&gt;dcoptimizer -->
<g id="edge11" class="edge">
<title>pdoublelimits&#45;&gt;dcoptimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2129,-308C2129,-284.62 2129,-260.56 2129,-238.19"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2131.63,-238.3 2129,-230.8 2126.38,-238.3 2131.63,-238.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1926,-267.77 1926,-307.37 2129,-307.37 2129,-267.77 1926,-267.77"/>
<text xml:space="preserve" text-anchor="start" x="1929" y="-304.37" font-family="Arial" font-size="14.00" fill="#c9c9c9">search bounds, fitness, operator</text>
<text xml:space="preserve" text-anchor="start" x="1929" y="-287.57" font-family="Arial" font-size="14.00" fill="#c9c9c9">probabilities</text>
</g>
<!-- pacga&#45;&gt;pinitialloadflow -->
<!-- pinitialloadflow&#45;&gt;acvalidator -->
<g id="edge12" class="edge">
<title>pinitialloadflow&#45;&gt;acvalidator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1456.04,-908C1446.54,-901.68 1437.15,-895.39 1428,-889.2 1401.96,-871.58 1374.38,-852.42 1348.03,-833.87"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1349.64,-831.79 1342,-829.61 1346.61,-836.08 1349.64,-831.79"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1179.57,-868.41 1179.57,-908.01 1397.57,-908.01 1397.57,-868.41 1179.57,-868.41"/>
<text xml:space="preserve" text-anchor="start" x="1182.57" y="-905.01" font-family="Arial" font-size="14.00" fill="#c9c9c9">compute budget, pruning, rejection</text>
<text xml:space="preserve" text-anchor="start" x="1182.57" y="-888.21" font-family="Arial" font-size="14.00" fill="#c9c9c9">thresholds</text>
</g>
<!-- client&#45;&gt;pareasettings -->
<g id="edge4" class="edge">
<title>client&#45;&gt;pareasettings</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1498.15,-1646.07C1224.37,-1641.76 697.04,-1623.84 527,-1550.4 495.58,-1536.83 465.49,-1517.37 437.94,-1495.68"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="439.72,-1493.74 432.22,-1491.09 436.43,-1497.83 439.72,-1493.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="735.43,-1623.06 735.43,-1645.86 959.43,-1645.86 959.43,-1623.06 735.43,-1623.06"/>
<text xml:space="preserve" text-anchor="start" x="738.43" y="-1642.86" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in StartPreprocessingCommand</text>
</g>
<!-- client&#45;&gt;pme -->
<g id="edge5" class="edge">
<title>client&#45;&gt;pme</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1835.97,-1635.64C2032.99,-1620.17 2336.29,-1590.22 2371,-1550.4 2558.6,-1335.18 2502.99,-1161.17 2371,-908 2365.17,-896.81 2358.22,-886.22 2350.43,-876.2"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2352.67,-874.8 2345.91,-870.64 2348.6,-878.11 2352.67,-874.8"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1951.36,-1602.15 1951.36,-1624.95 2164.36,-1624.95 2164.36,-1602.15 1951.36,-1602.15"/>
<text xml:space="preserve" text-anchor="start" x="1954.36" y="-1621.95" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in StartOptimizationCommand</text>
</g>
<!-- client&#45;&gt;pacga -->
<g id="edge6" class="edge">
<title>client&#45;&gt;pacga</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1667,-1558.61C1667,-1540 1667,-1519.82 1667,-1499.5"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1669.63,-1499.73 1667,-1492.23 1664.38,-1499.73 1669.63,-1499.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1394,-1523.63 1394,-1546.43 1667,-1546.43 1667,-1523.63 1394,-1523.63"/>
<text xml:space="preserve" text-anchor="start" x="1397" y="-1543.43" font-family="Arial" font-size="14.00" fill="#c9c9c9">set in the same StartOptimizationCommand</text>
</g>
</g>
</svg>`;default:throw Error(`Unknown viewId: `+e)}};export{e as dotSource,t as svgSource};