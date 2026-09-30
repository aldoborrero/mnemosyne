# AGENTS.md

Mnemosyne is a Hermes Agent memory plugin. It is consumed as a **directory** at
`$HERMES_HOME/plugins/mnemosyne/`, not as an installed Python distribution.

## Layout

| Path | What it is |
| --- | --- |
| `__init__.py` `cli.py` | thin Hermes directory/CLI discovery adapters; keep these at the root |
| `src/mnemosyne/__init__.py` | public provider exports for the Python package |
| `src/mnemosyne/provider.py` | `MnemosyneMemoryProvider` (Honcho + Hindsight) and registration of the provider selected by policy |
| `src/mnemosyne/prefetch.py` | parallel prefetch, anchor mtime cache, Honcho profile TTL and explicit invalidation; borrows the provider's executor |
| `src/mnemosyne/extraction_filter.py` | removes assistant paragraphs that repeat prefetched context before ingestion; leaves user messages to the provider |
| `src/mnemosyne/tool_schemas.py` | composite tool schemas, default exposure order, dispatch table and tool configuration constants; execution stays in the provider |
| `src/mnemosyne/recall_processing.py` | recall result formatting, deduplication, forget filtering, conflict labels and output truncation |
| `src/mnemosyne/commands.py` | `hermes mnemosyne …` subcommands |
| `src/mnemosyne/` | all implementation modules, using package-relative imports |
| `src/mnemosyne/approved.py` `src/mnemosyne/policy.py` | the approved-writes provider, and which provider runs and what it may do |
| `src/mnemosyne/memory_files.py` `src/mnemosyne/reconcile.py` | Hermes' built-in memory files as the source of truth; making a backend hold exactly the approved entries |
| `src/mnemosyne/hindsight_store.py` `src/mnemosyne/openviking_store.py` | backends for approved writes: one Hindsight document or one OpenViking file per entry |
| `src/mnemosyne/text_utils.py` | shared text normalisation for the similarity heuristics |
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
| `just lint` | `ruff check .` — Python lint without fixes, also enforced by `nix flake check` |
| `just fmt` | `nix fmt` — safe Python lint fixes + formatting (nix, python, yaml, toml, markdown, shell) |
| `just check` | `nix flake check` — build + tests + lint + formatting, the single CI gate |
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
pytest, and why `package.nix` installs a plugin tree instead of a wheel. The
tree must include both root adapters and `src/mnemosyne/`. `pyproject.toml`
discovers the package under `src/`; tests use that package directly and cover
Hermes' file-based discovery separately. The adapters preserve the package name
chosen by Hermes, including names that isolate separate plugin sources.

The formatter check runs inside `nix flake check` via `passthru.tests` (not
`meta.tests`, which blueprint silently ignores). After touching
`nix/formatter.nix`, confirm the check still exists:

```bash
nix eval .#checks.x86_64-linux --apply builtins.attrNames   # must list pkgs-formatter-check
```

CI pins nixpkgs to `flake.lock`'s revision rather than a channel, so a local
`nix flake check` and the CI one evaluate the same nixpkgs.

`nix fmt` runs `ruff check --fix` (safe fixes, priority 1) before `ruff format`
(priority 2). `pyproject.toml` explicitly selects `E4`, `E7`, `E9`, `F`, and `I`
for basic correctness and import ordering, targeting Python 3.11. The formatter
check enforces both lint and formatting through `nix flake check`.

The Hermes entry adapters and `tests/conftest.py` have local `E402` exceptions:
their delayed imports require package initialization or runtime stubs first.
Keep these imports after the setup they depend on.
