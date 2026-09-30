# Approved-writes mode

[Back to the README](../README.md)

For deployments where a human must approve what the agent remembers, set `ingest.mode` to `approved_writes` (or `MNEMOSYNE_INGEST_MODE=approved_writes`) together with `memory.write_approval: true` in Hermes. Mnemosyne then registers a different provider whose backends hold exactly the entries of Hermes' built-in memory files — `MEMORY.md` and `USER.md` under the profile's `memories/` — and nothing else: no conversation turns, session ends, compressions or delegations, and no memory written by a backend's own LLM.

## Reconciliation

Hermes can apply approved memory edits to files without calling `on_memory_write`. Mnemosyne therefore reads the files themselves instead of relying only on that hook.

Each entry gets an id from a hash of its text. Reconciliation stores new ids and deletes ids no longer present in the files. It is scheduled after initialization, after memory writes, and at turn start when the files have changed.

- An existing file, even an empty one, drives deletions.
- A missing or unreadable file leaves that target's backend copies untouched.
- A backend whose listing fails is skipped for that round.
- Cron, subagent, and flush sessions read but never reconcile.

## Profile isolation

**Isolation is per profile.** Hermes' built-in memory belongs to the profile and is injected into every session of it, so rooms whose members differ need one Hermes profile each, each with its own backend credentials in its `.env`. Mnemosyne namespaces everything by profile (`sha256(profile:home)[:16]`), but path prefixes and bank names are not a server-enforced boundary: give each profile its own OpenViking user/API key and Hindsight bank credentials.

## Settings

| Setting | Env var | Default | Effect |
|---|---|---|---|
| `ingest.mode` | `MNEMOSYNE_INGEST_MODE` | `turns` | `approved_writes` selects this mode |
| `approved.backends` | `MNEMOSYNE_APPROVED_BACKENDS` | `["openviking"]` | any of `openviking`, `hindsight`; anything else (Honcho included) fails closed |
| `approved.prefetch` | `MNEMOSYNE_APPROVED_PREFETCH` | `false` | inject matching entries every turn (sends the user's message to the backends as a query) |
| `approved.reflect` | `MNEMOSYNE_APPROVED_REFLECT` | `false` | expose `memory_reflect` (Hindsight, an LLM call the server traces by default) |
| `hindsight_direct.api_url` | `MNEMOSYNE_HINDSIGHT_URL` / `HINDSIGHT_API_URL` | — | required for the Hindsight backend; the key comes from `HINDSIGHT_API_KEY`; Hindsight Cloud is refused unless `hindsight_direct.allow_cloud` |

The provider starts disabled. Initialization must resolve the profile's `hermes_home`, validate the policy, and connect at least one backend before tools can run. If neither backend connects, memory stays disabled. One available backend can keep the provider running when the other is unavailable.

## Tools and commands

Tools: `memory_recall` (search approved entries), `memory_read` (one OpenViking entry by URI), and `memory_reflect` (opt-in, Hindsight). `memory_forget`, `memory_conclude`, `memory_profile`, and the Honcho tools do not exist in this mode. Remove entries through Hermes' approved `memory` tool.

Commands: `hermes mnemosyne reconcile` runs a reconcile now; `hermes mnemosyne purge --yes` deletes everything the profile's namespace holds in every backend. `import` and `forget` refuse to run in this mode.

## OpenViking

Mnemosyne uses the Hermes OpenViking plugin's HTTP client and connection settings (`OPENVIKING_*`) but not its provider, which uploads every turn and commits sessions — each commit makes the server extract memories with its own LLM. Entries are plain files at `viking://user/<space>/memories/mnemosyne/<ns>/<target>/mem_<id>.md`, written with `mode=create`, `processing_mode=vectors_only` and `wait=true` (searchable as soon as the write returns). Writes under `memories/` run no LLM on the server; the server does linkify bare `viking://` URIs inside the text. Search is `find` below the root with `read_content`, so recall returns the stored files' text. Do not run a `semantic_and_vectors` reindex over this namespace (it would send the files to the VLM), and consider `memory.extraction_enabled=false` on the server if nothing else of this user needs session extraction.

## Hindsight

Mnemosyne talks to the Hindsight API with `hindsight_client` instead of Hermes' Hindsight provider, which retains through LLM extraction and recalls LLM-consolidated observations only. Each profile gets bank `mnemosyne-<ns>`, configured at connect time with `retain_extraction_mode=chunks` (stored verbatim, no LLM), observations and auto-consolidation off and `store_document_text`; the backend refuses to run unless the server reports that configuration back. Each entry is document `mn-<target>-<id>`, so removal deletes the document and its memory units. Recall asks for `world` facts only. On a shared server set `HINDSIGHT_API_LLM_TRACE_ENABLED=false`: the server otherwise keeps every LLM prompt (reflect queries included) for a day.
