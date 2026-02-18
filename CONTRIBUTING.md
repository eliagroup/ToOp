# Contributing to ToOp

Thank you for your interest in contributing to ToOp! This guide will help you understand our development workflow and contribution process.

## Branch Strategy

We use [trunk-based development](https://trunkbaseddevelopment.com/):

- **`main`**: The stable trunk branch containing all development and releases

### Working with Branches

1. **Feature Development**: Create feature branches from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/your-feature-name
   ```

2. **Bug Fixes**: Create fix branches from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b fix/fix-description
   ```

3. **Pull Requests**:
   - All feature and fix branches should target `main`
   - Ensure your changes are tested and pass all CI checks before requesting review

## Local Development Setup

We provide a [Development Container](https://containers.dev/) configuration that sets up a complete development environment with all dependencies and tools pre-configured.

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed and running
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/eliagroup/ToOp.git
   cd ToOp
   ```

2. **Open in Dev Container**:
   - Open the repository in VS Code
   - When prompted, click "Reopen in Container" or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
   - Wait for the container to build and configure (first time may take several minutes)

### What's Included

The devcontainer automatically provides:

- **Python environment** with all dependencies installed via `uv`
- **Pre-commit hooks** automatically installed and configured
- **VS Code extensions** for Python, Jupyter, Azure tools, and code quality
- **Docker-in-Docker** for containerized workflows
- **Git configuration** with safe directory setup
- **Testing suite** based on `pytest` with VS Code testing integration.
   - Run all tests: `uv run pytest` (this may take some time)
   - Run some test: `uv run pytest packages/<package_name>_pkg/<package_name>_tests`

### Benefits

- **Consistent Environment**: Everyone uses the same development setup
- **Quick Start**: No manual dependency installation required
- **Isolation**: No impact on your local system
- **Pre-configured Tools**: All linting, formatting, and validation tools ready to use

### Commit Message Validation

Commit messages are validated both locally (via pre-commit hooks) and in CI. The validation uses [Commitizen](https://commitizen-tools.github.io/commitizen/) with the conventional commits standard.

If your commit message doesn't follow the convention:
- Locally: The pre-commit hook will block the commit
- In CI: Pull requests will fail validation

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

## Pull Request Process

1. **Create a Branch**: Create a new branch from the `main` branch
2. **Make Changes**: Implement your feature or fix
3. **Write Tests**: Ensure your changes are covered by tests
4. **Push**: Push your branch to the repository
5. **Open PR**: Create a pull request targeting the `main` branch

### PR Requirements

- Single purpose: Each PR should contain one type of change - either a feature, a bugfix, or a refactor. Avoid mixing different types of changes in a single PR
- Title of the pull request must pass conventional commit validation
- Code must pass all pre-commit hooks
- Tests must pass.
   - Code coverage must be over 90% and aimed for 100%.
   - Test locally by running `uv run pytest`
- Documentation should be updated if needed

#### PR Commit Message Standards

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

#### Developer Certificate of Origin

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


## Getting Help

If you have questions about contributing:

- Check existing issues and pull requests
- Review the codebase documentation in the `docs/` directory
- Reach out to the maintainers

Thank you for contributing to ToOp! 🚀

## License

When you contribute to ToOp, you acknowledge and agree to the terms set out for any current and future contributions you provide.

All contributions will be licensed according to the license specified in the repository.
