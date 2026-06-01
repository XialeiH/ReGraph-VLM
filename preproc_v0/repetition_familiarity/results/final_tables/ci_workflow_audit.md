# CI Workflow Audit

Status counts: {'ready': 10}

| Item | Status | Evidence |
| --- | --- | --- |
| workflow name | ready | publication-preflight workflow name is present |
| main push trigger | ready | workflow runs on main pushes |
| pull request trigger | ready | workflow runs on pull requests |
| ubuntu runner | ready | publication preflight uses ubuntu-latest |
| checkout and Python setup | ready | workflow checks out source and configures Python |
| Python version | ready | workflow pins Python 3.10 for audit execution |
| TeX dependencies | ready | workflow installs latexmk and required TeX package bundles |
| audit dependencies | ready | workflow installs lightweight audit dependency set |
| compile-required preflight | ready | workflow runs compile-required publication preflight |
| clean-worktree enforcement | ready | workflow fails if generated artifacts drift or new files appear |
