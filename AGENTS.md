# AGENTS.md

Mnemosyne is a Hermes Agent memory plugin. It is consumed as a **directory** at
`$HERMES_HOME/plugins/mnemosyne/`, not as an installed Python distribution.

## Layout

| Path | What it is |
| --- | --- |
| `__init__.py` | `MnemosyneMemoryProvider` — composes Honcho + Hindsight; the Hermes entry point |
| `cli.py` | `hermes mnemosyne …` subcommands |
| `config.py` `conflict.py` `dedup.py` `fact_store.py` `forget.py` `importer.py` `recovery.py` | plugin internals, all package-relative imports |
| `plugin.yaml` | Hermes plugin manifest (hooks, tool surface, pip deps) |
| `install.sh` | installs `honcho-ai` + `hindsight-client` into the Hermes venv |
| `tests/` | pytest suite; `conftest.py` stubs Hermes and both backends |
| `flake.nix` | 4 inputs, outputs delegated to `numtide/blueprint` with `prefix = "nix"` |
| `nix/devshell.nix` | numtide devshell (python3, pytest, ruff, uv, just, formatter) |
| `nix/formatter.nix` | treefmt-nix config + the hermetic format check |
| `nix/packages/mnemosyne/` | `default.nix` shim + nixpkgs-style `package.nix` |

New flake outputs are never added to `flake.nix` — they are files under `nix/`,
which blueprint discovers by path.

## Commands

| Command | What it does |
| --- | --- |
| `just test` | `pytest tests` |
| `just lint` | `ruff check .` — advisory, not part of `nix flake check` |
| `just fmt` | `nix fmt` — nix, python, yaml, toml, markdown, shell |
| `just check` | `nix flake check` — build + tests + formatting, the single CI gate |
| `just build` | `nix build .#mnemosyne` |

`direnv allow` (or `nix develop`) puts all of the above on `PATH`.

## Invariants

Flake evaluation only sees **git-tracked** files — `git add` new files before
running any `nix` command, or they are invisible to the build and to the
formatter check.

The two runtime dependencies, `honcho-ai` and `hindsight-client`, are **not in
nixpkgs**. Nothing in this tree imports them directly: they are reached through
Hermes (`plugins.memory.hindsight`) at runtime, `install.sh` puts them in the
Hermes venv, and the test suite stubs them. That is why the Nix build needs only
pytest, and why `package.nix` installs a plugin tree instead of a wheel — the
flat `py-modules` layout in `pyproject.toml` would install `cli`, `config`, …
as top-level modules that cannot resolve each other's relative imports.

The formatter check runs inside `nix flake check` via `passthru.tests` (not
`meta.tests`, which blueprint silently ignores). After touching
`nix/formatter.nix`, confirm the check still exists:

```bash
nix eval .#checks.x86_64-linux --apply builtins.attrNames   # must list pkgs-formatter-check
```

CI pins nixpkgs to `flake.lock`'s revision rather than a channel, so a local
`nix flake check` and the CI one evaluate the same nixpkgs.

`nix fmt` formats Python with `ruff format` only. `ruff check --fix` is
deliberately not part of the formatter: under ruff 0.16's default rule set this
tree has 288 findings, 174 of which `--fix` would rewrite as a side effect of
formatting, and 114 of which have no fix at all — so wiring it in would make
`nix flake check` permanently red. Linting stays advisory (`just lint`); if the
project wants it enforced, add a `[tool.ruff]` section to `pyproject.toml`
selecting the rules it actually intends, then enable `programs.ruff-check`.
