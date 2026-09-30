![Mnemosyne — memory for Hermes. An amber M connected to a constellation of memories.](docs/assets/mnemosyne.svg)

# Mnemosyne

Long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Combine a user model and factual recall in one provider, or mirror only the memory entries approved through Hermes.

[Get started](#get-started) · [Configuration](#configuration) · [Approved writes](docs/approved-writes.md) · [Development](#development) · [Apache-2.0](LICENSE)

## Choose your memory mode

| | Conversation memory (`turns`, default) | Approved memory (`approved_writes`) |
| --- | --- | --- |
| Source | Conversation turns and explicit memory writes | Hermes' `MEMORY.md` and `USER.md` files |
| Backends | Honcho for the user model; Hindsight for facts | OpenViking (default), Hindsight, or both |
| Writes | Delegates conversation ingestion to the inner providers | Reconciles backend entries with the memory files |
| Recall | Profile, factual search, reflection, and prefetched context | Search approved entries; optional prefetch and reflection |
| Removal | Preview and confirm a soft-delete through Mnemosyne | Remove entries through Hermes' approved `memory` tool |

The conversation provider adds date and repetition metadata, an anchor card for pinned facts, recall deduplication, conflict annotations, crash recovery, and session imports. Prefetch combines the anchor card, Honcho profile, and Hindsight results; repeated prefetched text is filtered out of assistant content before ingestion.

Approved memory runs a separate provider. It does not ingest conversation turns. Enable Hermes' write approval alongside this mode and configure the desired backend before activation. See the [approved-writes guide](docs/approved-writes.md) for reconciliation, profile isolation, and backend requirements.

## Get started

### Requirements

- Python **3.11+**, on Linux or macOS.
- A working Hermes Agent installation and its Python virtual environment.
- Configured memory backends for your chosen mode. Conversation mode uses Hermes' Honcho and Hindsight providers; approved mode needs Hermes' OpenViking client and connection settings, or a Hindsight API endpoint.

The plugin is loaded as a **directory**, including its root adapters and `src/` package. Keep the full checkout together.

### Install

```bash
git clone https://github.com/aldoborrero/mnemosyne.git \
  "${HERMES_HOME:-$HOME/.hermes}/plugins/mnemosyne"
cd "${HERMES_HOME:-$HOME/.hermes}/plugins/mnemosyne"
./install.sh
```

`install.sh` installs `honcho-ai` and `hindsight-client>=0.4.22` into Hermes' virtual environment. It does not provision backend servers. The default environment is `~/.hermes/hermes-agent/venv`; override it when needed:

```bash
HERMES_VENV=/path/to/hermes/venv ./install.sh
```

### Configure and activate

Choose your mode in [configuration](#configuration) first. For conversation mode, check the [embedding and reranker defaults](#conversation-backend-routing): they expect local services that the installer does not start.

```bash
hermes config set memory.provider mnemosyne
hermes gateway restart
```

Inspect the local setup and gateway logs:

```bash
hermes mnemosyne status
tail -n 100 "${HERMES_HOME:-$HOME/.hermes}/logs/agent.log"
```

`status` reports local files and the availability of the Honcho/Hindsight providers. In approved mode, use the gateway logs to verify the selected backend initialized. Then try a memory query in Hermes, such as “What do you remember about my current project?”

## Usage

### Conversation tools

Hermes exposes these tools to the agent in conversation mode:

| Tool | Purpose |
| --- | --- |
| `memory_profile` | Read or update the Honcho profile card |
| `memory_reasoning` | Ask Honcho about preferences, habits, or communication style |
| `memory_conclude` | Record a stable user-related conclusion in Honcho |
| `memory_recall` | Search Hindsight for remembered facts |
| `memory_reflect` | Ask Hindsight to synthesize an answer from memories |
| `memory_forget` | Preview matching memories, then confirm forgetting |

Forgetting uses local signatures to filter future recall, with optional Hindsight tombstones. It is a soft-delete mechanism, not guaranteed erasure from every backend.

### CLI

```bash
# Inspect the plugin and pinned context.
hermes mnemosyne status
hermes mnemosyne anchor list

# Pin a short fact for conversation-mode prefetch.
hermes mnemosyne anchor add --text "The current project is Mnemosyne."

# Import past sessions into Hindsight (conversation mode).
hermes mnemosyne import --days 90 --min-turns 5

# Interactively select and confirm memories to forget (conversation mode).
hermes mnemosyne forget "an outdated project"
```

In approved mode, `hermes mnemosyne reconcile` synchronizes the backends with the current memory files. `import` and `forget` refuse to run in that mode; use Hermes' approved memory workflow instead. Backend cleanup commands are described in the [guide](docs/approved-writes.md).

## Configuration

Create `config.json` under the profile's `$HERMES_HOME/plugins/mnemosyne/` directory when you need overrides. If the file is absent, built-in defaults apply; the loader does not create it.

**Precedence:** mapped environment variables → `config.json` → built-in defaults. The full settings and environment mapping live in [config.py](src/mnemosyne/config.py).

For example, this configuration selects approved writes with OpenViking and leaves automatic recall and reflection off:

```json
{
  "ingest": { "mode": "approved_writes" },
  "approved": {
    "backends": ["openviking"],
    "prefetch": false,
    "reflect": false
  }
}
```

Also enable `memory.write_approval: true` in Hermes and configure OpenViking's connection settings. Configure the mode before starting the gateway; an environment variable set only in an interactive shell may not reach an already-running service.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MNEMOSYNE_INGEST_MODE` | `turns` | Select `turns` or `approved_writes` |
| `MNEMOSYNE_APPROVED_BACKENDS` | `openviking` | Comma-separated approved backends: `openviking`, `hindsight` |
| `MNEMOSYNE_APPROVED_PREFETCH` | `false` | Automatically recall approved entries each turn |
| `MNEMOSYNE_APPROVED_REFLECT` | `false` | Expose Hindsight reflection in approved mode |
| `MNEMOSYNE_HINDSIGHT_URL` | Empty | Hindsight endpoint for approved writes |
| `MNEMOSYNE_PREFETCH_MAX_TOKENS` | `4500` | Conversation prefetch budget |
| `MNEMOSYNE_TIMEOUT_RECALL` | `180` | Recall tool timeout, in seconds |
| `MNEMOSYNE_RECOVERY_ENABLED` | `true` | Enable conversation-mode startup recovery |

### Conversation backend routing

The conversation provider supplies these Hindsight defaults when the corresponding environment variables are unset. Override them in the gateway environment or the `hindsight_env` object in `config.json`; nonempty environment values take precedence.

| Setting | Default |
| --- | --- |
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | `openai` |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | `http://localhost:8000/v1` |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | `jina-embeddings-v5-text-small-retrieval-mlx` |
| `HINDSIGHT_API_RERANKER_PROVIDER` | `cohere` |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | `http://localhost:4000/v1/rerank` |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | `rerank` |

Both `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` and `HINDSIGHT_API_RERANKER_COHERE_API_KEY` default to the local placeholder `sk-local-litellm`. Set credentials appropriate to your services.

### Local state

Runtime files live under the profile's plugin directory:

| File | Purpose |
| --- | --- |
| `config.json` | Optional local configuration |
| `anchor_card.md` | Pinned facts for conversation prefetch |
| `fact_store.db` | Fact metadata, repetition counts, and forgotten signatures |
| `recovery_cursor.json` | Startup replay progress |
| `import_cursor.json` | Bulk import progress |

The database and cursor files are gitignored. `config.json` and `anchor_card.md` are not currently ignored; keep local configuration and personal facts out of commits. Approved mode uses the profile's `memories/MEMORY.md` and `memories/USER.md` as its source of truth.

## Update or roll back

For a checkout installed from this fork:

```bash
cd "${HERMES_HOME:-$HOME/.hermes}/plugins/mnemosyne"
git pull --ff-only
# Run ./install.sh if Python dependencies changed.
hermes gateway restart
```

Keep local runtime state when updating. To switch away, select your previous provider and restart; for example, if it was Honcho:

```bash
hermes config set memory.provider honcho
hermes gateway restart
```

## Development

```bash
nix develop
just test
just lint
just fmt
just check
```

| Command | What it verifies or changes |
| --- | --- |
| `just test` | Runs the pytest suite |
| `just lint` | Runs Ruff without changing files |
| `just fmt` | Applies safe Ruff fixes and formats supported files through treefmt-nix |
| `just check` | Runs the Nix CI gate: plugin build, tests, lint, and formatting |
| `just build` | Builds the plugin directory at `result/share/hermes/plugins/mnemosyne` |

Nix reads Git-tracked files: **stage new files before running Nix commands**. Ruff selects `E4`, `E7`, `E9`, `F`, and `I`, targeting Python 3.11.

```text
__init__.py, cli.py       Hermes directory and CLI discovery adapters
src/mnemosyne/
  provider.py            Conversation provider and lifecycle
  tool_schemas.py        Tool schemas and dispatch metadata
  prefetch.py            Parallel prefetch and caches
  recall_processing.py   Recall formatting, filtering, and deduplication
  extraction_filter.py   Filters prefetched assistant text before ingestion
  approved.py, policy.py Approved-writes provider and write policy
  commands.py            CLI implementation
  ...                    Backend stores and memory helpers
tests/                   Unit and provider behavior tests
docs/                    Approved-writes guide and project artwork
nix/                     Development shell, formatter, and plugin package
```

Tests stub Hermes and backend services. They cover both provider modes and provider-first/CLI-first loading, but do not replace a smoke test against a real Hermes installation. Keep changes focused and include behavior tests when changing memory handling. See [AGENTS.md](AGENTS.md) for repository conventions.

## Credits and license

This fork builds on [johnnykor82/mnemosyne](https://github.com/johnnykor82/mnemosyne). Conversation memory integrates Honcho and Hindsight through Hermes; approved memory supports OpenViking and Hindsight.

Named after Mnemosyne, the Greek Titaness of memory. Licensed under [Apache-2.0](LICENSE).
