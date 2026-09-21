# Mnemosyne — Long-Term Memory Plugin for Hermes Agent

> 🇷🇺 Читать на русском: [README.ru.md](README.ru.md)

**Mnemosyne** (Μνημοσύνη — the Greek Titaness of memory, mother of the nine Muses) is a composite long-term memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It combines two existing memory providers — **[Honcho](https://github.com/plastic-labs/honcho)** (user model: style, preferences, conversational personality) and **[Hindsight](https://hindsight.app)** (fact store: entity graph, multi-strategy semantic recall) — behind a single `MemoryProvider` interface, and adds date tagging, repetition counting, conflict detection, crash recovery, and explicit forgetting on top.

Pairs naturally with its mythological daughter [Hermes-Mneme](https://github.com/johnnykor82/hermes-mneme) — a separate plugin that handles *short-term, in-session* context engineering.

## Why a composite memory provider

Hermes Agent permits only one external memory provider at a time. That's a problem, because the two best providers in the ecosystem are complementary, not interchangeable:

- **Honcho** is excellent at modelling *who the user is* — dialectic reasoning, peer cards, communication style, stable preferences. It's weaker at precise factual recall: no decay, no deduplication, LLM-generated representations.
- **Hindsight** is the inverse — strong factual store, entity graph, hybrid semantic + keyword recall, but no model of the user as a person.

Mnemosyne wraps both behind a single `MemoryProvider` and routes every call to whichever inner provider is right for the job:

- **Writes** → both providers (Honcho keeps its user-model fed; Hindsight builds the fact graph).
- **Auto-inject (prefetch)** → Hindsight facts + Honcho peer card; Honcho's noisy session summary / user representation is suppressed (we recommend configuring Honcho to `recallMode: tools`).
- **Tools** → six curated `memory_*` tools with tight role-based descriptions that route to the right inner provider under the hood.

The composite layer also adds:

1. **Date tags** on every retained fact (`fact_store.db`) — so "what did we discuss last week?" actually works.
2. **Repetition counter** tracking how often a fact comes up — important facts surface higher in recall.
3. **Explicit forgetting** — `memory_forget` tool + CLI, soft-delete via signature with semantic deduplication so the same fact doesn't sneak back in.

## Status

**Scaffold (Stage A).** Currently pure fan-out delegation to the two inner providers. The composite layer (date tags, repetition counter, conflict resolver, forgetting, anchor card, prefetch fusion) is on the roadmap below.

## Requirements

- **Python 3.11+**
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** installed and working (the plugin assumes a Hermes venv at `~/.hermes/hermes-agent/venv`, configurable via `HERMES_VENV`).
- **[Honcho](https://honcho.dev)** — Python client (`honcho-ai`). Already in place if you've been using Honcho as your memory provider.
- **[Hindsight](https://hindsight.app)** — install via `hermes memory setup` and pick `hindsight`.
- **macOS or Linux.** The plugin is platform-agnostic Python; the installer covers both.

## Installation

```bash
git clone https://github.com/johnnykor82/mnemosyne.git \
  ~/.hermes/plugins/mnemosyne
cd ~/.hermes/plugins/mnemosyne
./install.sh
```

The installer detects your Hermes venv (default `~/.hermes/hermes-agent/venv`, override with `HERMES_VENV=...`), installs the two Python dependencies (`honcho-ai`, `hindsight-client>=0.4.22`), and verifies they import cleanly.

The plugin lives in `$HERMES_HOME/plugins/mnemosyne/` (default `~/.hermes/plugins/mnemosyne/`) and survives `git pull` of `hermes-agent` itself — it's a user-installed addon, not a core component.

## Activation

```bash
hermes config set memory.provider mnemosyne
hermes gateway restart
```

Verify it loaded:

```bash
tail -f ~/.hermes/logs/agent.log | grep -i mnemosyne
```

To roll back to a previous provider:

```bash
hermes config set memory.provider honcho   # or whatever you were using
hermes gateway restart
```

## Configuration

Mnemosyne reads its configuration from `mnemosyne/config.json` (created at plugin directory on first run). Most defaults work out of the box — the plugin is designed so that a fresh install "Just Works" against a local LiteLLM proxy at `http://localhost:8000`.

Key things you can override (via `config.json` `hindsight_env` block, or via shell env vars — shell wins over config):

| Variable | Default | Purpose |
|---|---|---|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | `openai` | Embedding provider name passed to Hindsight |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | `http://localhost:8000/v1` | Where the embeddings endpoint lives (local LiteLLM by default) |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | `sk-local-litellm` | Placeholder for the local proxy. Replace if pointing to a real API. |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | `jina-embeddings-v5-text-small-retrieval-mlx` | Default embedding model (Jina MLX local) |
| `HINDSIGHT_API_RERANKER_PROVIDER` | `cohere` | Reranker provider name |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | `http://localhost:4000/v1/rerank` | Reranker endpoint |
| `HINDSIGHT_API_RERANKER_COHERE_API_KEY` | `sk-local-litellm` | Placeholder for the local proxy |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | `rerank` | Default reranker model |

Plugin-internal storage:

- `fact_store.db` — SQLite store for date tags, repetition counters, forgotten signatures. Created on first run.
- `recovery_cursor.json` — offset into `~/.hermes/sessions/` for crash-recovery imports.

Both are gitignored — they are local runtime state, not part of the plugin.

## Approved-writes mode

For deployments where a human must approve what the agent remembers, set `ingest.mode` to `approved_writes` (or `MNEMOSYNE_INGEST_MODE=approved_writes`) together with `memory.write_approval: true` in Hermes. Mnemosyne then registers a different provider whose backends hold exactly the entries of Hermes' built-in memory files — `MEMORY.md` and `USER.md` under the profile's `memories/` — and nothing else: no conversation turns, session ends, compressions or delegations, and no memory written by a backend's own LLM.

**Why the files, not `on_memory_write`.** Hermes stages a memory write for approval whenever no inline approval prompt is available (always on the gateway), and `/memory approve` later applies it to the files without calling memory providers. The files are therefore the only view of approved memory that every approval path reaches. Each entry gets an id from a hash of its text, and a *reconcile* makes a backend hold exactly the set in the files: new ids are stored, ids no longer in the file are deleted. It runs after `initialize`, after each `on_memory_write`, and at turn start when the files changed. A file that exists — even empty — drives deletions; a file that is missing or unreadable leaves that target's copies untouched (a file that disappears is logged with the purge command), and a backend whose listing fails is skipped for that round. Cron, subagent and flush sessions read but never reconcile.

**Isolation is per profile.** Hermes' built-in memory belongs to the profile and is injected into every session of it, so rooms whose members differ need one Hermes profile each, each with its own backend credentials in its `.env`. Mnemosyne namespaces everything by profile (`sha256(profile:home)[:16]`), but path prefixes and bank names are not a server-enforced boundary: give each profile its own OpenViking user/API key and Hindsight bank credentials.

| Setting | Env var | Default | Effect |
|---|---|---|---|
| `ingest.mode` | `MNEMOSYNE_INGEST_MODE` | `turns` | `approved_writes` selects this mode |
| `approved.backends` | `MNEMOSYNE_APPROVED_BACKENDS` | `["openviking"]` | any of `openviking`, `hindsight`; anything else (Honcho included) fails closed |
| `approved.prefetch` | `MNEMOSYNE_APPROVED_PREFETCH` | `false` | inject matching entries every turn (sends the user's message to the backends as a query) |
| `approved.reflect` | `MNEMOSYNE_APPROVED_REFLECT` | `false` | expose `memory_reflect` (Hindsight, an LLM call the server traces by default) |
| `hindsight_direct.api_url` | `MNEMOSYNE_HINDSIGHT_URL` / `HINDSIGHT_API_URL` | — | required for the Hindsight backend; the key comes from `HINDSIGHT_API_KEY`; Hindsight Cloud is refused unless `hindsight_direct.allow_cloud` |

The provider starts without memory and enables itself only at the end of an `initialize` that got the profile's `hermes_home` from Hermes and connected at least one backend; any failure leaves the session without memory rather than with a shared or default store. Tools: `memory_recall` (search the approved entries), `memory_read` (one OpenViking entry by URI), `memory_reflect` (opt-in). `memory_forget`, `memory_conclude`, `memory_profile` and the Honcho tools do not exist in this mode: removal is a `remove` through Hermes' approved `memory` tool.

Commands: `hermes mnemosyne reconcile` runs a reconcile now; `hermes mnemosyne purge --yes` deletes everything the profile's namespace holds in every backend. `import` and `forget` refuse to run in this mode.

### OpenViking

Mnemosyne uses the Hermes OpenViking plugin's HTTP client and connection settings (`OPENVIKING_*`) but not its provider, which uploads every turn and commits sessions — each commit makes the server extract memories with its own LLM. Entries are plain files at `viking://user/<space>/memories/mnemosyne/<ns>/<target>/mem_<id>.md`, written with `mode=create`, `processing_mode=vectors_only` and `wait=true` (searchable as soon as the write returns). Writes under `memories/` run no LLM on the server; the server does linkify bare `viking://` URIs inside the text. Search is `find` below the root with `read_content`, so recall returns the stored files' text. Do not run a `semantic_and_vectors` reindex over this namespace (it would send the files to the VLM), and consider `memory.extraction_enabled=false` on the server if nothing else of this user needs session extraction.

### Hindsight

Mnemosyne talks to the Hindsight API with `hindsight_client` instead of Hermes' Hindsight provider, which retains through LLM extraction and recalls LLM-consolidated observations only. Each profile gets bank `mnemosyne-<ns>`, configured at connect time with `retain_extraction_mode=chunks` (stored verbatim, no LLM), observations and auto-consolidation off and `store_document_text`; the backend refuses to run unless the server reports that configuration back. Each entry is document `mn-<target>-<id>`, so removal deletes the document and its memory units. Recall asks for `world` facts only. On a shared server set `HINDSIGHT_API_LLM_TRACE_ENABLED=false`: the server otherwise keeps every LLM prompt (reflect queries included) for a day.

## Tools exposed to the LLM

| Tool | Purpose |
|---|---|
| `memory_profile` | Read or update the user's profile card (name, role, communication style, stable preferences). Routes to Honcho. |
| `memory_reasoning` | Questions *about the user as a person* — style, habits, behavioural patterns, what works best with them. Routes to Honcho. |
| `memory_conclude` | Record a stable user-related conclusion (preference, habit, style). Routes to Honcho. |
| `memory_recall` | **The main long-term memory tool.** "Do you remember when…", "we discussed this", multi-strategy semantic + entity-graph search over all past conversations. Routes to Hindsight. |
| `memory_reflect` | LLM synthesis across past-conversation facts — summaries spanning multiple sources ("what did we conclude about X?"). Routes to Hindsight. |
| `memory_forget` | Explicit forgetting via signature soft-delete. Implemented inside the composite layer. |

These are the default mode's tools; approved-writes mode has its own set (see above).

## Hooks

Mnemosyne hooks into Hermes at six lifecycle points (declared in `plugin.yaml`):

- `on_memory_write` — intercept memory store operations.
- `on_session_end` — finalize state at session end.
- `on_session_switch` — handle multi-session context.
- `on_pre_compress` — pre-compression deduplication.
- `on_delegation` — agent delegation events.
- `on_turn_start` — per-turn initialization.

## Updating

When new commits land on `main`:

```bash
cd ~/.hermes/plugins/mnemosyne
git pull
./install.sh              # reinstalls deps if requirements changed
hermes gateway restart
```

Your runtime data (`fact_store.db`, `recovery_cursor.json`) is gitignored and survives updates.

## Contributing

Contributions and bug reports are very welcome. Standard GitHub flow:

1. **Issues** — open an issue describing the problem or feature idea.
2. **Pull requests** — fork, branch, commit, push, open a PR against `main`.

Before submitting a PR:

- Verify your change works on both **macOS** and **Linux** if it touches `install.sh` or filesystem paths.
- Run `ruff check` to catch obvious style issues.
- Keep commits focused — one concern per commit.

## License

[Apache-2.0](LICENSE)

## Roadmap

| Stage | Status | What |
|---|---|---|
| Scaffold (Stage A) | ✅ current | Fan-out delegation, union of inner tool schemas |
| 1, 2 | planned | Date tags + repetition counter (SQLite `fact_store`) |
| 3 | planned | Pre-write dedup — short recall before each `retain` |
| 5 | planned | Anchor card — manually-curated pinned facts always in prefetch |
| 6 | planned | Curated tools — six renamed `memory_*` with tight role descriptions |
| 6′ | planned | Conflict resolver — two-voice display when Honcho and Hindsight disagree |
| 7 | planned | Prefetch fusion — anchor → Honcho peer card → Hindsight recall |
| 7′ | planned | Recovery — re-send transcripts from `~/.hermes/sessions/` after crashes |
| 8 | planned | Bulk import — last 90 days of session transcripts on demand |
| 10 | planned | Built-in memory bridge — `on_memory_write` → Hindsight with `mention_count=10` |
| 11 | planned | Forgetting — `memory_forget` tool + CLI, soft-delete via `forgotten:` tag |
