# Contributing to ToOp

Thank you for your interest in contributing to ToOp! This guide will help you understand our development workflow and contribution process.

## Local Development Setup

Furthermore, we recommend the use of Visual Studio Code as you IDE including its Python Extension for an integrated test environment.
If you want to run all tests locally, you also need `Docker`.

### Getting Started

Clone the repository
```bash
   git clone https://github.com/eliagroup/ToOp.git
   cd ToOp
```
and install dependencies by running
```bash
  uv sync --all-groups
```

### Test installation

VS Code should automatically detect all tests on the Test tab.
Alternatively, you can run all tests via the terminal by running
```bash
   uv run pytest
```
or the test of some package via
```bash
   uv run pytest packages/<package_name>_pkg/tests
```

## Contributing via Pull Requests

We use [trunk-based development](https://trunkbaseddevelopment.com/), where **`main`** refers to the stable trunk branch containing all development and releases.

The contribution guidelines differ slightly between internal and external developers.
As an external developer, you can fork our repository and contribute to code via pull requests.
Make sure to name pull request according to our [commit message standard](./contribution_guide.md#pr-commit-message-standards) as CI validation will fail otherwise.

### Contributing with AI/LLM tools

You may use AI coding assistants, but:

- You are fully responsible for all contributed code, whether written manually or generated.
- You must understand and be able to explain every submitted change.
- A human must always review, validate, and approve the final contribution.
- Do not use AI to interact with maintainers on your behalf. Improving grammar or clarity of your message is permitted.
- Fully autonomous AI agents that open PRs without human review are not allowed and PRs will be rejected.

### Acknowledgements
This AI contribution guide is mainly based on SymPy and SciPy developers' AI policies. We thank them for their contribution.

### Branching for Development
You are advised to name branches in accordance with your goal.
For that, we provide [commit message types](./contribution_guide.md#commit-types). For example, if you develop a new feature you can name your branch:

**Feature Development**: Create feature branches from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/your-feature-name
   ```

**Bug Fixes**: Create fix branches from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b fix/fix-description
   ```

### Standard PR workflow for external developers

1. **Fork and install**:
   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   git clone https://github.com/<your-username>/ToOp.git
   cd ToOp
   uv sync
   pre-commit install
   ```

2. **Add our repo to merge updates**:
   ```bash
   git remote add upstream https://github.com/eliagroup/ToOp.git
   ```

3. **Create a branch from `main`**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feat/your-change
   ```

4. **Implement and validate** (tests + checks):
   ```bash
   uv run pytest
   pre-commit run --all-files
   ```

5. **Sync with upstream before pushing to your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

6. **Commit your change**:
We require [Conventional Commits](https://www.conventionalcommits.org/) and [Developer Certificate of Origin (DCO)](./contribution_guide.md#developer-certificate-of-origin) for each commit to the main branch.
Note that these commit messages are based on all commit messages within your PR, which will be squashed before a merge into main.
You will make your life easier by adhering to conventional commits throughout all your commits.
   You can do so by using the `-s` flag when committing.
      ```bash
      git add <files>
      git commit -s
      ```

6. **Push to your fork and open a PR to `main`**:
   ```bash
   git push --set-upstream origin feat/your-change
   ```

### Acceptance Criteria for PRs

- Single purpose: Each PR should contain one type of change - either a feature, a bugfix, or a refactor. Avoid mixing different types of changes in a single PR
- PR description should be meaningful but concise. Use the following template:

- Title of the pull request must conform to the [conventional commit spec](./contribution_guide.md#pr-commit-message-standards). It will be used for the squashed commit message of a PR.
- Pull request squash commit message has to include **Developer Certificate of Origin**. Under some circumstances GitHub automatically adds author's signature, but in some cases it has to be added manually.
- Code must pass all pre-commit hooks `pre-commit run --all-files`.
- Tests must pass.
   - Code coverage must be over 90% and aimed for 100%.
   - Test locally by running `uv run pytest`
- Documentation should be updated if needed

### PR Commit Message Standards

We require [Conventional Commits](https://www.conventionalcommits.org/) and Developer Certificate of Origin (DCO) for each commit in the main branch. These commits are created when a pull request is squashed into the main branch.

Such a pull request's commit message must follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

Signed-off-by: FirstName LastName <something@example.org>
```

##### Commit Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Developer Certificate of Origin

The last line of the commit message certifies the origin of the code that will be committed.
This means that you certify you have the rights to submit this work under this project's open source license (see [Git docs](https://git-scm.com/docs/git-config#Documentation/git-config.txt-formatsignOff)).

##### Examples

```bash
feat: add user authentication system

Signed-off-by: FirstName LastName <something@example.org>
---------------------------------------------------------

fix(api): resolve memory leak in data processing

Signed-off-by: FirstName LastName <something@example.org>
---------------------------------------------------------

docs: update installation instructions

Signed-off-by: FirstName LastName <something@example.org>
```

### Commit Message Validation

Commit messages are *not* validated but the final squashed commit of your PR will be. For that, we use [Commitizen](https://commitizen-tools.github.io/commitizen/) with the conventional commits standard.

## Writing tests

These rules apply to humans and coding agents alike. Ruff does not lint `tests/`, so reviewers check them by hand.

### Structure

- **Mirror `src/`.** The test for `toop_engine_x/a/b.py` lives in `packages/x_pkg/tests/a/test_b.py`. Look for an existing module before creating a new one.
- **One behaviour per test.** Arrange, act, assert. If a test needs a paragraph to explain, or grows past a screen, split it.
- **Name the behaviour, not the function.** Use `test_<unit>_<does_what>_<when>`, so a failing name reads as a bug report.

  | Good | Not good |
  | --- | --- |
  | `test_screening_is_skipped_when_the_basecase_did_not_converge` | `test_main` |
  | `test_update_static_information_removes_busbar_data_when_disabled` | `test_preprocess_net_step1` |
  | | `test_switching_tables_V2` |

  See `contingency_analysis_pkg/tests/pandapower/cascade/test_basecase_screen.py` and
  `topology_optimizer_pkg/tests/dc/genetic_functions/test_initialization.py` for the style we want.
- **Assert values.** A test must fail when the behaviour breaks. `assert True`, `assert result is not None` on its own,
  or "it did not raise" are not tests. Compare against expected numbers, shapes, ids or a reference implementation
  (e.g. `dc_solver_pkg/tests/numpy_reference.py`).
- **Do not weaken tests to make them pass.** Changing an expected value, widening a tolerance, adding `skip`/`xfail` or deleting
  a test needs a reason in the PR description.

### Fixtures

- **Reuse before you create.** Each package's `tests/conftest.py` already provides grids, preprocessed folders, `StaticInformation`,
  N-1 definitions and Kafka setups. Search it before writing a new fixture, and extend an existing one rather than copying it.
- **Expensive data once, copies per test.** A session-scoped `_name` fixture builds the data; a function-scoped `name` fixture
  returns a copy, so tests can mutate it safely:

  ```python
  @pytest.fixture(scope="session")
  def _case14_data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
      tmp_path = tmp_path_factory.mktemp("case14")
      case14_pandapower(tmp_path)
      return tmp_path


  @pytest.fixture
  def case14_data_folder(_case14_data_folder: Path, tmp_path: Path) -> Path:
      shutil.copytree(_case14_data_folder, tmp_path, dirs_exist_ok=True)
      return tmp_path
  ```

- **Smallest grid that shows the behaviour.** Use case14 or a synthetic network before case57, Oberrhein or real UCTE/CGMES data.
- Fixture names describe content (`case14_data_folder`, `jax_inputs_oberrhein`), never `test_*`, `data1` or `..._V2`.

### Cost tiers and the development loop

A fast loop lets you, and coding agents, iterate in seconds instead of minutes:

1. Run the single test you are working on: `uv run pytest packages/<pkg>/tests/<path>::<test_name>`.
2. Run its module, then the package: `uv run pytest packages/<pkg>/tests`.
3. Run the full suite (`uv run pytest -n auto --dist loadgroup`) only before opening the PR, or leave it to CI.

Know what a test costs before you add it:

- **Pure unit tests** (plain functions, small arrays, case14) should run in well under a second. Prefer this tier.
- **JAX tests** pay compilation on first call. Reuse compiled shapes and avoid parametrizing over many shapes.
  Set `JAX_PLATFORMS=cpu` without a GPU.
- **Service tests** need Docker (Kafka, Ray). Group Kafka tests with `@pytest.mark.xdist_group("kafka")`.
- Give slow tests an explicit `@pytest.mark.timeout(...)`.

Coverage must stay above 90% (`.coveragerc`). Treat it as a signal for untested behaviour, not as a target to hit with
assertion-free tests.

## Naming & documentation

Good names make most review comments unnecessary. Before you open a PR, read your diff once only for names.

### Names

- **Say what it is in domain terms, not how or when it was made.** Use the vocabulary from the [glossary](./glossary.md);
  do not invent synonyms. Add a new term to the glossary in the same PR.
- **Avoid these patterns** (all taken from the current code base):

  | Pattern | Example | Better |
  | --- | --- | --- |
  | `_helper`, `_utils`, `do_`, `process_` with no object | `graph_creation_nodes_helper` | `add_nodes_to_graph` |
  | Version or step suffixes | `basic_node_breaker_network_powsybl_v2`, `preprocess_net_step1` | describe the content: `node_breaker_network_with_six_voltage_levels`, `clean_up_imported_network` |
  | One variable reused for different things | `res` holding flows, then masked flows, then monitored flows | `flows`, `masked_flows`, `monitored_flows` |
  | Generic containers | `df`, `data`, `tmp`, `result` | `branch_results`, `outage_group_table` |
  | Single letters outside formulas | `b = network.get_buses()` | `buses = network.get_buses()` |

- Single letters are fine for indices in short loops and for symbols from a formula the docstring cites.
- Keep the established conventions: `n_` prefix for counts and dimensions, `_mask` for boolean arrays, `_ids` for string identifiers,
  `from_node` / `to_node` for branch ends, `UPPER_SNAKE_CASE` for module constants.
- Booleans read as a statement: `is_monitored`, `has_converged`, `enable_bb_outages`.

### Docstrings and comments

- Every public function has a NumPy-style docstring with `Parameters`, `Returns` and `Raises` where applicable
  (`ruff.toml`: `convention = "numpy"`). One-line docstrings are fine for small helpers without parameters.
- Document what the signature cannot say: intent, units (MW, p.u., degrees), array shapes, invariants, and why edge cases are handled
  the way they are. Do not restate the type hints.
- Dataclass and Pydantic attributes are documented with a string directly below the field.
- Comments explain *why*. If a comment explains *what* a block does, extract the block into a well-named function instead.
- When you change behaviour, update the docstring, the relevant `docs/` page, the [glossary](./glossary.md), and the
  [architecture model](https://github.com/eliagroup/ToOp/tree/main/docs/architecture) if cross-package flow changed.

## Release Process

Releases are managed through GitHub Actions and only generate Git tags. No commits are created during the release process.

### Release Types

1. **Stable Releases** (from `main` branch):
   - Follow semantic versioning (e.g., `v1.2.3`)
   - Used for production-ready code

2. **Development Releases** (from feature branches):
   - Include a development identifier (e.g., `v1.2.3.dev12345`)
   - Used for testing and preview purposes

### Release Workflow

The release process is automated through `.github/workflows/release.yaml`:

1. **Manual Trigger**: Releases are triggered manually via GitHub Actions
2. **Version Calculation**: Commitizen analyzes commit history to determine the next version
3. **Tag Creation**: A Git tag is created with the new version
4. **Tag Push**: The tag is pushed to the repository

**Note**: Releases only create Git tags. No packages are published to external registries.

### Creating a Release

1. Ensure your branch has the changes you want to release
2. Go to the GitHub Actions tab in the repository
3. Select the "release" workflow
4. Click "Run workflow" and select the appropriate branch
5. The workflow will automatically determine the next version and create a tag

## Getting Help

If you have questions about contributing:

- Check existing issues and pull requests
- Review the codebase documentation in the `docs/` directory
- Reach out to the maintainers

Thank you for contributing to ToOp! 🚀

## License

When you contribute to ToOp, you acknowledge and agree to the terms set out for any current and future contributions you provide.

All contributions will be licensed according to the license specified in the repository.
