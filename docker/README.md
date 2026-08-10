# Running ToOp on a Windows laptop with Docker

This directory contains a runnable ToOp image and a `docker-compose.yaml` (at the repository root)
for driving it from a Windows 11 machine — via Jupyter, an interactive shell, or the VS Code dev
container. The Python environment is created and executed entirely through `uv`.

> This is separate from `dev-deployment/docker-compose.yaml`, which starts the Kafka broker used by
> the worker-based deployment. That file is unchanged and still works standalone.

## Prerequisites

- **Docker Desktop** with the **WSL2** backend (not Hyper-V). Start it before any command below.
  Verify with `docker version` — it must report a Linux server engine.
- ~15 GB free disk. The CPU image holds about **4.5 GB** of content, 3.0 GB of which is the Python
  environment (jaxlib 383M, pypowsybl 223M, ray 220M, polars 204M, plotly 188M). The CUDA variant is
  substantially larger, and BuildKit's layer cache needs headroom beyond the image itself.

  Docker's own size reporting is unreliable here: under the containerd image store both
  `docker images` and `docker image inspect --format '{{.Size}}'` report ~1.0 GB — the *compressed*
  total — immediately after a build, and `docker images` was observed switching to 4.49 GB only
  after a daemon restart. To measure for real, look inside (prefix with `MSYS_NO_PATHCONV=1` in Git
  Bash, see below):
  `docker run --rm --entrypoint sh toop-toop:latest -c "du -sh /opt/venv"`.
- No local Python needed — everything runs inside the container.

## Quick start

```powershell
cd C:\Users\<you>\...\ToOp
docker compose build toop        # ~9 min cold (6.5 min of it is `uv sync`)
docker compose up -d
```

Subsequent container starts take ~2 s: the entrypoint's `uv sync` is a no-op check
(`Checked 312 packages in 62ms`) unless `uv.lock` actually changed.

Open `http://localhost:8888` and run `notebooks/example1_dc_loadflow_example.ipynb`.

For a shell instead:

```powershell
docker compose exec toop bash
# then, inside:
uv run python -c "import jax; print(jax.devices())"
uv run pytest packages/interfaces_pkg/tests -q
uv run python -m toop_engine_topology_optimizer.dc.main --help
```

Stop with `docker compose down`. Add `-v` to also discard the environment and cache volumes.

### If you use Git Bash instead of PowerShell

Git Bash (MSYS2) rewrites arguments that look like absolute POSIX paths into Windows paths before
`docker.exe` ever sees them, so a container path such as `/app` silently becomes
`C:/Program Files/Git/app`:

```
docker: Error response from daemon: the working directory 'C:/Program Files/Git/app'
is invalid, it needs to be an absolute path
```

Prefix the command with `MSYS_NO_PATHCONV=1` whenever it contains a container-absolute path
(`-w /app`, `-v <host>:/app`, or a bare `/opt/...` argument):

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/app" -w /app toop-toop:latest uv run pytest -q
```

The plain `docker compose` commands in this file are unaffected — their paths are relative — so
PowerShell users and anyone staying on `docker compose` can ignore this. Doubling the leading slash
(`//app`) works too, but is easier to forget.

## Running tests

```powershell
docker compose run --rm toop uv run pytest packages/interfaces_pkg/tests -q
docker compose run --rm toop uv run pytest packages/dc_solver_pkg/tests -q -n 4 --dist loadgroup
```

**Prefer an explicit `-n <N>` over `-n auto` here.** CI uses `-n auto` on a dedicated runner; inside
WSL2 on a laptop it starts one worker per core, each loading its own JAX runtime, which can easily
outgrow the VM's memory allowance. During this setup's first run the Docker engine became
unresponsive about four minutes into an `-n auto` invocation of this suite — memory pressure is the
plausible explanation, though a concurrent Docker Desktop self-update muddied the evidence and the
cause was never conclusively pinned down. Either way, `-n 4` is a safer starting point on a laptop,
and raising the WSL2 memory limit (below) is worthwhile.

`--dist loadgroup` is mandatory whenever `-n` is used: Kafka, Ray and performance tests are
serialized through `@pytest.mark.xdist_group` and will interfere with each other otherwise.

Measured result of the command above on a 13.58 GB VM: **491 passed, 1 failed, 8 skipped, 2 xfailed
in 13:41**. The one failure is `jax/benchmarks/test_bench_postprocessing.py::test_main`, which dies
with `ray.exceptions.OutOfMemoryError` — Ray sees the node at 95 % and kills a worker. It passes
when the file is run on its own:

```powershell
docker compose run --rm toop uv run pytest packages/dc_solver_pkg/tests/jax/benchmarks/test_bench_postprocessing.py -q
# 3 passed in 211.42s
```

So it is a VM memory-ceiling artefact, not a defect in the image. Raise the WSL2 memory limit
(below) or run the Ray benchmarks separately from the rest of the suite.

## Where to put your grid data

The repository is bind-mounted at `/app`, so the `data/` folder you see in Explorer is the same one
the pipeline reads and writes.

1. Create `data\<iteration_name>\` in Explorer.
2. Drop your grid file in it — `.xiidm`, `.json` (pandapower), or `.zip` (CGMES).
3. Set `iteration_name` and `file_name` in the notebook config cell.

The pipeline then creates `data\<iteration_name>\<timestamp>\` and fills it with the processed grid
folder: `grid.xiidm`/`grid.json`, `masks\`, `loadflow_parameters.json`, `initial_topology\`,
`static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `nminus1_definition.json`,
and `optimizer_snapshots\` with the DC and AC results. Filenames come from
`packages/interfaces_pkg/src/toop_engine_interfaces/folder_structure.py`.

Everything appears in Explorer as it is written, so you can inspect results without entering the
container.

> Grid data may be confidential. A pre-commit hook blocks committing `.uct`, `.xml`, `.zip`,
> `.json`, `.xiidm`, `.veragrid` and `.hdf5` files unless whitelisted in
> `.pre-commit-hooks/sensitive-files-whitelist.txt`.

## Why the virtualenv is not in the repository folder

`uv` creates the environment at **`/opt/venv`**, not `/app/.venv`, via `UV_PROJECT_ENVIRONMENT`.
`/app` is a bind mount from the Windows filesystem; a Linux virtualenv there would collide with any
natively-created `.venv` and would push thousands of small files across the WSL2 filesystem bridge
on every import.

`uv sync` and `uv run` behave exactly as documented — only the environment's location moves. The six
`packages/*` are installed **editable**, so editing a `.py` file in Windows takes effect on the next
import with no rebuild.

The environment lives on the `toop-venv` named volume, so it survives `docker compose down`. If it
ever gets into a bad state: `docker compose down -v && docker compose up -d`.

## Enabling the NVIDIA GPU

Requires an NVIDIA GPU, a current Windows driver with WSL2 CUDA support, and Docker Desktop's GPU
support enabled. AMD and Intel integrated GPUs are not usable — JAX has no backend for them here.

```powershell
docker compose --profile gpu build toop-gpu
docker compose --profile gpu up -d toop-gpu
docker compose exec toop-gpu uv run python -c "import jax; print(jax.devices())"
```

Jupyter for the GPU service is on `http://localhost:8889` so it can coexist with the CPU one. If
`jax.devices()` still reports CPU, the container is not seeing the GPU — check
`docker compose exec toop-gpu nvidia-smi` first.

## Performance notes for Windows

**Bind-mount speed.** `C:\...\ToOp` → `/app` crosses the WSL2 9p/virtiofs bridge. Imports are
unaffected (the environment is on a Linux-native volume), but random access into
`static_information.hdf5` and `action_set_diffs.hdf5` is measurably slower than native. If a large
grid becomes I/O-bound, move the checkout into the WSL2 filesystem
(`\\wsl$\<distro>\home\<you>\ToOp`) and run the same compose file from there — no changes needed.

**Memory.** JAX and pypowsybl's native runtime are memory-hungry, and with no explicit setting WSL2
takes about half the host's RAM — measured here, a 27.9 GB host produced a 13.58 GB VM. That is not
enough for the whole `dc_solver_pkg` suite at `-n 4`: Ray killed a worker at 95 % utilisation
(12.91 GB of 13.58 GB) in `test_bench_postprocessing.py::test_main`, a test that passes when run on
its own. A plain container OOM instead shows up as exit code 137.

Raise it in `%UserProfile%\.wslconfig` (create the `[wsl2]` section if absent):

```ini
[wsl2]
memory=16GB      # on a 32GB host; leave the OS several GB
processors=8
swap=4GB
```

then `wsl --shutdown` followed by restarting Docker Desktop. Ray's own kill threshold can be moved
with `RAY_memory_usage_threshold`, but raising the VM's memory is the better fix.

**Shared memory.** The compose file sets `shm_size: 2gb`; Ray and JAX allocate through `/dev/shm`
and Docker's 64 MB default produces confusing crashes.

## VS Code dev container

`.devcontainer/devcontainer.json` builds this same image (with `DEV_TOOLS=true`, adding `vim`,
`htop`, `less` and friends). Open the folder and choose **Reopen in Container**.

Two details differ from the compose flow:

- VS Code mounts the workspace at `/workspaces/ToOp` rather than `/app`, so `TOOP_SKIP_SYNC=1` is
  set and `.devcontainer/onCreateCommand.sh` performs the `uv sync` from the correct directory.
- The interpreter is `/opt/venv/bin/python`, preconfigured in `python.defaultInterpreterPath`.

The `${localEnv:HOME}/.ssh` bind mount was removed from `devcontainer.json`: `HOME` is not set on
Windows hosts, so that mount failed there.

## Regenerating protobuf schemas

Works here, and everywhere else. `protoc` comes from the `grpcio-tools` wheel rather than a system
binary, so the compiler version is pinned by `uv.lock` and needs no image-specific tooling:

```powershell
docker compose run --rm toop uv run bash packages/interfaces_pkg/src/compile_proto.sh
```

Regeneration is rare — the `*_pb2.py` / `.pyi` files are committed, and there is exactly one schema
(`message_wrapper.proto`, a single-field envelope marked temporary in its own comment).

**Two things to know before you regenerate.**

*The output is equivalent but not identical.* Comparing generated code across `grpcio-tools`
1.66–1.83 by AST shows **all 13 top-level statements identical** except the
`ValidateProtobufRuntimeVersion(...)` call, which merely embeds the generator's own version. The
serialized descriptor, field numbers and wire format never change. The committed file declares
gencode 5.29.3 (standalone `protoc` v29.3); the current toolchain emits 7.35.1. Older gencode loads
fine on a newer runtime, so the committed file works as-is — but the reverse does not, so
regenerating raises the effective minimum protobuf runtime to 7.x while the packages still declare
`protobuf>=5,<8`. Tighten that constraint in the same change if you commit regenerated output.

*Do not try to pin `grpcio-tools` low to reproduce the 5.29.x gencode.* Releases before 1.72 cap
protobuf at `<6`, which drags the entire environment's runtime down to 5.29.x — and `mypy-protobuf`
ships gencode built against 6.32.1, so `--mypy_out` then fails with
`Runtime version cannot be older than the linked gencode version`. Hence the `>=1.72` floor.

`protoc` does not emit the MPL header, so re-add it after regenerating; CI's `nwa` check enforces it.

## What is not included

Kafka worker services. `docs/usage.md` describes running
`packages/topology_optimizer_pkg/.../dc/worker/worker.py` directly, but that file (and the AC
equivalent) has no `__main__` block and no CLI: its `main()` takes already-constructed Kafka
producers, consumers and `fsspec` filesystems. Containerizing that path requires writing launcher
shims first. The directly runnable CLIs are
`toop_engine_topology_optimizer.dc.main` and
`toop_engine_contingency_analysis.ac_loadflow_service.lf_worker`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `exec /usr/local/bin/entrypoint.sh: no such file or directory` | `entrypoint.sh` was checked out with CRLF endings. `.gitattributes` forces LF; re-clone or run `git add --renormalize .`. |
| `the working directory 'C:/Program Files/Git/app' is invalid` (or any container path prefixed with a Git install dir) | Git Bash rewrote a POSIX path. Prefix the command with `MSYS_NO_PATHCONV=1`, or run it from PowerShell. |
| Container exits with code 137 | Out of memory — see the `.wslconfig` section above. |
| `docker version` returns `500 Internal Server Error` for *every* call, while `wsl -l -v` shows `docker-desktop` Running and `docker desktop status` says `running` | The Desktop backend is wedged; the "check if the server supports the requested API version" wording is boilerplate appended to any 500, not a real version problem (pinning `DOCKER_API_VERSION` does not help). Confirm by checking whether `%LOCALAPPDATA%\Docker\log\vm\init.log` has stopped growing, and whether `docker desktop status` still reports the *same* `SessionID` after a restart — an unchanged SessionID means the backend never actually restarted. Fix: quit Docker Desktop fully from the tray, `wsl --shutdown`, then relaunch and let any pending update finish. |
| `detected dubious ownership in repository at '/app'` | The entrypoint sets `safe.directory`; this means it was bypassed. Run `git config --global --add safe.directory /app` inside the container. |
| `uv sync` fails offline at startup | `docker compose run -e TOOP_SKIP_SYNC=1 toop bash`. |
| Rebuild does not pick up dependency changes | `docker compose build --no-cache toop`, then `docker compose down -v` to reset the environment volume. |
