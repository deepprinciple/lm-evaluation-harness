# Recorded SDE API models

`sde-chat-completions` is a no-tools DeepSeek Chat Completions adapter. It uses
harness tasks, chat formatting, filters and metrics, and returns execution metadata
alongside generated text. Legacy model registrations keep their existing behavior.
The `sde-messages` and `sde-responses` adapters add Opus Messages and Astra
Responses SSE profiles using the same request lifecycle. `sde-claude-code`
adds a recorded Claude Code CLI agent with shared MCP tools. `sde-codex` adds
a Codex CLI agent with API-key and explicit ChatGPT-subscription profiles.

## Configuration

The adapter requires `model`, `record_dir` and an explicit `max_tokens` fallback.
The token budget is not inferred from the model name. It uses the standard harness
precedence: task generation kwargs, including global `--gen_kwargs` overrides,
then the model fallback. Set global `max_tokens` to prevent task YAML budgets such
as 4096 from overriding the intended experiment budget.

The default profile sends `reasoning_effort=max`, `thinking.type=enabled` and
`stream=false`. It omits temperature, seed and stop, including those inherited from
YAML. Other generation kwargs are rejected, rather than silently forwarded.
`reasoning_effort` and `thinking` are model arguments.

The defaults are four concurrent questions, a 3600-second question deadline and
at most three total attempts. The deadline starts when a worker is available and
includes request time and backoff. Once started, it also includes downtime after
process interruption. The timeout and attempt count are configurable; these
runtime defaults do not imply equal compute across models.

This is a usage example, not a record of an executed experiment:

```bash
lm_eval --model sde-chat-completions \
  --model_args model=deepseek-flash,max_tokens=393216,record_dir=results/flash_max/raw,reasoning_effort=max,num_concurrent=4,timeout=3600,max_attempts=3 \
  --gen_kwargs max_tokens=393216 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/flash_max
```

Authentication uses `DEEPSEEK_API_KEY` by default. `api_key_env` can name another
environment variable, and `base_url` can select another Chat Completions endpoint.
The outgoing thinking fields are a DeepSeek profile, not a guarantee that another
provider accepts these parameters.

## Records and retry policy

Each draw has an atomic JSON record containing the exact payload, configuration,
adapter checksum, task/document/request identity, repeat index and every attempt.
Records retain full HTTP bodies, parsed responses, reasoning text if returned by
the provider, usage and partial bodies from interrupted reads. Headers and API keys
are not stored; an echoed key is redacted from the body.

Completed records can be reused only for matching identities. Repeats have separate
records. An interrupted in-flight attempt consumes one attempt when resuming.
Use a different `record_dir` for a new independent experiment. Concurrent processes
cannot use the same directory; file locking requires macOS or Linux.

408, 429, 5xx, transport errors, malformed responses and empty/incomplete generation
can trigger bounded technical retries. A normal answer, a structured refusal or
length truncation does not trigger retry. Scientific scores are never consulted.
401/403 is recorded and stops the batch. After fixing credentials, use a new
record directory; the saved authentication failure is not silently reused as an
answer. `--use_cache` is rejected to protect independent sampling.

Sample logs contain status, repeat index, selected-attempt usage, all available
attempt usage and the raw record path under `response_metadata`. Only `ok` text is
sent to the original task filters; other statuses return empty scoring text while
retaining the actual provider output in raw records. Free-text refusals without a
provider refusal signal remain ordinary text; this adapter does not use an LLM
judge to classify them.

This change does not add a status-aware benchmark summary or per-repeat accuracy
aggregation. A task's existing filters still determine which repeat is scored.


## Opus Messages SSE

`sde-messages` ports the no-tools payload and conservative relay handling from
our existing Opus evaluation scripts. It is not the Claude Code agent harness.
The shared SDE runtime still supplies deadlines, bounded technical retries,
locking, independent repeat identities and atomic JSON records.

The required arguments remain `model`, `max_tokens` and `record_dir`. Defaults:
`base_url=https://api.anthropic.com/v1/messages`,
`api_key_env=ANTHROPIC_API_KEY`, `effort=max`,
`anthropic_version=2023-06-01`, four workers, 3600 seconds per question and three
attempts. Override the endpoint and credential environment name for a relay.

The Messages profile sends `output_config.effort`, `max_tokens` and `stream=true`.
It does not forward DeepSeek thinking fields, temperature, seed, stop or tools.
Leading system messages become top-level text blocks; user/assistant few-shot
messages retain their contents. A system message appearing after conversation
has begun is rejected. This ports the earlier payload without adding an explicit
thinking toggle. Requested effort is recorded, but an endpoint accepting a
parameter does not prove its reasoning behavior; only returned thinking is saved.

Example only; neither this migration nor its tests call a live provider:

```bash
lm_eval --model sde-messages \
  --model_args model=claude-opus-5-5,max_tokens=128000,record_dir=results/opus_max/raw,base_url=https://mira.grecharge.ai/v1/messages,api_key_env=ANTHROPIC_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3 \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/opus_max
```

128000 is an explicit example budget, not an inferred or verified provider maximum.
The harness override precedence above applies here too.

Each attempt writes a redacted `.attemptNN.sse` sidecar beside its JSON record.
Complete lines are flushed while the request is running; the final partial line
is flushed on normal exit, transport failure or timeout. Abrupt process termination
may lose the last incomplete line; already flushed lines remain on disk. As in the
shared runtime, an interrupted attempt consumes an attempt on resume, rather than
silently resubmitting outside the budget. SSE files are evidence, not an additional
answer cache.

The parser reconstructs text, thinking, signatures, redacted thinking and usage.
A normal answer requires a closed, well-formed message, `message_stop`,
`stop_reason=end_turn` and the requested returned model ID. Missing end markers
and stream overload errors are technical failures. Malformed control fields,
unclosed frames and unexpected tool calls are protocol errors. Model mismatch is
terminal, as are explicit refusal and token truncation; neither scientific scores
nor free-text refusal keywords drive retry selection.

A distinct same-model assistant message can supersede a previous segment only
when that segment contains unfinished thinking, no text/tool blocks, no stop
reason and no completed-message marker. All segments and their usage remain in
the raw record. A restart after any answer text or completed generation is
ambiguous and is not accepted. The parser never combines answers across segments.

`response_metadata.stream_metadata` records the SSE paths, closure checks, event
counts, parser errors and per-segment usage. The normal `usage` field describes
the selected final message; it is not a claim about total relay billing. Full
thinking and text live in the raw JSON/SSE files, not in the text sent to filters.
Status-aware summaries and per-repeat accuracy aggregation remain pending.


## Astra Responses SSE

`sde-responses` ports the no-tools Astra payload from the existing Responses
experiment. It is not Astra@Codex and does not use subscription authentication.
The model is a configured string, not an alias resolved by the harness.

The required arguments are `model`, `max_tokens` and `record_dir`. Defaults:
`base_url=https://api.openai.com/v1/responses`, `api_key_env=OPENAI_API_KEY`,
`effort=max`, four workers, 3600 seconds per question and three total attempts.
Use the relay endpoint and your credential environment name explicitly. The
harness `max_tokens`/`max_gen_toks` budget maps to `max_output_tokens` in the HTTP
payload. It is not forwarded under both names. All role/content messages,
including system prompts and few-shot examples, are retained in `input`.

The payload sends `reasoning={effort: ...}` and `stream=true`, omitting temperature,
seed, stop and tools. This preserves the old profile; the migration does not
request extra reasoning summaries or encrypted reasoning, change the model,
or launch a new run. No model-specific token ceiling is inferred.

This is an example with an explicit budget, not a live compatibility test or
claim that 128000 is the provider's maximum:

```bash
lm_eval --model sde-responses \
  --model_args model=gpt-6-astra,max_tokens=128000,record_dir=results/astra_max/raw,base_url=https://mira.grecharge.ai/v1/responses,api_key_env=NEW_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3 \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/astra_max
```

The event schemas were checked against the
[OpenAI Responses streaming reference](https://developers.openai.com/api/reference/resources/responses/streaming-events).
The API distinguishes `response.completed`, `response.incomplete` and
`response.failed`, and provides separate text, refusal and reasoning events.
This documentation does not establish a third-party relay's current compatibility.

The adapter accepts normal answers only from a valid `response.completed` event
whose response status is `completed`, model ID matches, and reasoning effort
matches the requested effort. The optional relay `[DONE]` sentinel cannot replace
that event. It rejects duplicate terminal events, response restarts, changed
response identity/model, replayed sequence numbers when supplied, malformed
SSE frames, contradictory statuses and unexpected tool output. It permits a
terminal-only stream without requiring omitted optional intermediate events.

A completed response's output is authoritative. Text deltas are preserved as
stream evidence and are not an answer fallback when terminal output is empty.
This intentionally tightens the old script's delta fallback. Effort mismatch,
malformed protocol, empty output, 524, transport failure and incomplete generation
use the existing bounded technical retry policy. As in the other profiles,
refusals and token truncation are terminal, and scientific scores are never
consulted to select an attempt. `response.incomplete` with reason
`max_output_tokens` is recorded as truncated; content filtering or a refusal
content block is recorded as refusal. Explicit failed/error events are recorded
as technical failures with the provider's error preserved.

The adapter records all returned reasoning items, encrypted fields, summary/text
deltas, output items, usage and the raw SSE. It cannot recover reasoning text
that the provider does not disclose. A reasoning token count is not a transcript;
a summary is not the full reasoning; encrypted content is not readable text.
The echoed effort is a protocol check, not proof of the amount of computation.

`response_metadata` adds `returned_effort` and per-attempt `stream_metadata`
with SSE paths, response IDs, terminal events, optional sentinels, event counts
and parser errors. Complete/partial usage remains under the existing `usage` and
`attempt_usage` fields when returned. Partial stream fragments never enter the
original task filters as completed answers.

Opus and Astra share `sde_stream.py` for SSE framing and durable capture. The
Opus payload and bounded request lifecycle remain unchanged; both streaming
profiles record the shared helper's checksum so resume identities include that
implementation. The JSON and SSE files have the same persistence limitations
already described above. Historical results remain untouched; use a fresh record
directory for a new experiment under a revised implementation.


## Claude Code CLI agent

`sde-claude-code` launches a fresh Claude Code process for each question attempt.
This is an API-key profile for Flash or Opus, not a subscription profile. It keeps
the original task/filter/metric pipeline and returns only a successful CLI result
to scoring. It preserves every attempt's trace, stderr, effective Messages payload,
raw HTTP/SSE, returned thinking, tool requests and tool results. The API capture
includes Claude Code's effective system prompt and advertised tool schemas.

Required arguments: `model` (CLI ID), `api_model` (actual Messages ID),
`max_tokens`, `record_dir`, `claude_bin`, exact `cli_version`, `tool_python`, and
`base_url` (provider root, not the full `/v1/messages` URL). Default authentication
is `ANTHROPIC_API_KEY`; select `DEEPSEEK_API_KEY` for Flash. The MCP server requires
`SERPAPI_API_KEY` and `JINA_API_KEY`. Keys enter the child environment and proxy
headers only; request logs, traces and tool results redact exact secret strings.
For an npm CLI, specify `node_bin` so the sanitized child PATH uses its intended
Node installation. The CLI version and binary hash, optional Node version/hash,
Python version, compute package versions and adapter/helper hashes are recorded.

The constructor makes no model requests. It checks `claude --version`, records
NumPy/SciPy/SymPy/RDKit/requests versions, and runs an offline filesystem/network
isolation probe using the selected compute Python. The run requires Linux x86_64
with Landlock ABI >= 3 and seccomp. It fails before evaluation if that probe fails;
it does not silently replace the Linux computation environment with Mac shell.
Local Mac tests cover mocked CLI processes, local HTTP forwarding and budget
logic; they do not establish that a remote container supports isolation.

The CLI uses `--tools ""`, only the `sde_tools` MCP server and three allowed tools:
`web_search`, `read_url`, `run_shell`. Native web/shell/file tools, additional agents
and user/project settings are excluded by the command and isolated HOME. Init
traces and observed tool calls are checked; unexpected tools are a terminal
`tool_profile_mismatch`. Claude Code's internal `EndConversation` may appear as
conversation control; it is not an additional search or compute tool. Managed
machine policy cannot be removed by this adapter; actual prompts and tool schemas
are captured, and real CLI compatibility remains a preflight requirement.
CLI flags were checked against the
[Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference).

All runs use the same MCP schema and implementation:

- Google/SerpAPI: three backend attempts per question, including failures, with
  a file-locked atomic counter shared across CLI retries and process restarts.
  Results contain at most eight title/URL/snippet entries, without an LLM summary.
- Jina: public HTTP(S) input URLs, benchmark/private input sources excluded, at
  most 32000 returned characters. Redirect following by Jina remains provider
  behavior; the adapter validates the original URL rather than claiming control
  over the reader's full internal retrieval path.
- Linux compute: the existing Landlock/seccomp launcher, no shell network or
  benchmark files, one thread, 60 CPU seconds, 2 GiB address space, 20 MiB file
  limit, 70-second wall limit, stdout/stderr bounded to 24000/8000 characters.
  The tail read is bounded in bytes before decoding and slicing; incomplete UTF-8
  at its starting boundary is replaced.

The x86_64 seccomp launcher also blocks the x32 syscall ABI. Its readable runtime
roots include system binaries/libraries and the selected Python runtime/venv;
keep datasets, answer keys and prior results outside those roots. Each question
attempt has its own compute directory. A normal deadline/cancellation terminates
the CLI process group and gives the MCP server a short cleanup window to terminate
its compute group. Abrupt OS kills can bypass cleanup and final partial-line
flushes; flushed evidence survives, but the adapter does not claim transactional
process cleanup after a power loss.

The adapter defaults to `effort=max`, four workers, 3600 seconds per question and
three total attempts using the same runtime as the API adapters. Queue wait is
excluded, while started attempts, tools and backoff share one deadline. The
inherited backoff is 2/4 seconds for the two retries; the older draft protocol's
5/15 seconds has not been installed by this migration. There is no CLI turn or
main-request count limit and no per-question total token cap. `max_tokens` remains
a per-response budget and is inherited from harness kwargs using the same
precedence as the API adapters.

A transparent loopback proxy forwards Messages bodies unchanged. It checks the
actual outgoing model, `max_tokens` and `output_config.effort`, and rejects native
or additional tool definitions. A smaller or larger CLI budget is recorded as
terminal `budget_mismatch`, not silently rewritten or claimed to be matched.
Model mismatch is terminal. This strict profile also applies to any CLI helper
or compaction generation; incompatibility needs review rather than an unnoticed
exception. Requests and raw returned thinking are retained even if the CLI fails.
Closed Messages streams may end in a tool call; final scoring still requires the
CLI's unique successful result and a zero process exit code. Refusal, token/context
limits, turn limits and configuration mismatches do not trigger technical retry.
Scientific correctness or answer-format extraction never selects a retry.

Any observable upstream transport/protocol failure ends that CLI attempt and
blocks further proxy forwarding, preventing the CLI's nested transport retries
from bypassing the outer attempt budget. Observable 401/403 stops the batch under
the common runtime. Hidden retries inside the provider cannot be counted. An
interrupted process consumes its attempt on resume; already spent search attempts
stay spent. Resume restarts a new isolated CLI attempt; it does not restore a
partial Claude conversation. New independent experiments require a fresh
`record_dir`.

A single user question is passed unchanged on stdin. Leading harness system
messages are appended as instructions. Few-shot user/assistant histories are
preserved as labelled role/content JSON within the CLI input; this is not a claim
that CLI history has the same API-level roles as the no-tools Messages profile.

Example only, with locally installed paths and an explicitly pinned CLI version:

```bash
lm_eval --model sde-claude-code \
  --model_args 'model=deepseek-flash[1m],api_model=deepseek-flash,max_tokens=128000,record_dir=results/flash_cc_max/raw,claude_bin=/opt/sde/claude,cli_version=2.1.296,tool_python=/opt/sde/tools-venv/bin/python,node_bin=/opt/sde/node/bin/node,base_url=https://api.deepseek.com/anthropic,api_key_env=DEEPSEEK_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3' \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/flash_cc_max
```

For Opus, explicitly select the CLI and API model `claude-opus-5-5`, relay root
`https://mira.grecharge.ai` and its credential environment. This example is not
an executed experiment, deployment instruction or verified third-party endpoint
compatibility. CLI output limits, tool connection and actual effort need a
separately authorized live smoke test. Status summaries, repeat accuracy
aggregation remain pending.


## Codex CLI agent

`sde-codex` uses Codex for the agent loop and retains the same harness question,
filter and metric pipeline. It shares the three MCP tools, Linux compute isolation,
search quota, question deadline, technical retry lifecycle and appended scientific
instructions with `sde-claude-code`. It does not port the historical scripts'
six-turn limit or fabricated tool-discovery generation. The CLI's own base prompt
and conversation representation remain framework-specific; identical tools and
budgets do not imply identical harness instructions or inference cost.

Required arguments are `model`, `max_tokens`, `record_dir`, `codex_bin`, exact
`cli_version`, `tool_python`, `model_catalog` and `base_url`. The selected catalog
must contain exactly one entry for that model supporting the requested effort.
Its existing model instructions are retained, not synthesized or substituted.
The full file checksum and selected entry are recorded along with the CLI binary,
version, feature inventory and shared runtime/helper checksums. Runtime validation
rejects a CLI missing the required isolation/configuration switches. An npm CLI
can also specify `node_bin` for the isolated PATH. This draft was checked with
`codex-cli 0.162.0`; a different version requires its own compatibility check.

There are two explicit authentication profiles:

- `auth_mode=api_key` (default): reads only the named environment variable
  (`DEEPSEEK_API_KEY` by default). An `auth_file` is rejected. The CLI sees a
  loopback session token; the proxy adds the actual API credential upstream.
- `auth_mode=chatgpt`: requires an explicit `auth_file` containing a ChatGPT
  login and the exact endpoint `https://chatgpt.com/backend-api/codex/responses`.
  It refuses a relay endpoint rather than forwarding subscription credentials to
  one. The login's account identity is hashed and checked on each attempt; token
  values are not configuration/resume identities. A 0600 authentication copy is
  placed in temporary CODEX_HOME outside the retained result tree, then deleted.
  The CLI may refresh that copy; the original file is never overwritten. This is
  not an API key minted from a subscription and does not establish live access to
  any particular model. Concurrent token refresh behavior needs a live check.

Both profiles require `SERPAPI_API_KEY` and `JINA_API_KEY`. Each attempt starts a
fresh CLI process and isolated HOME without personal settings, hooks, skills,
plugins, project AGENTS documents or previous conversations. Native shell, search,
file operations and additional agents are excluded from the model-facing tool
schemas. Only `web_search`, `read_url` and `run_shell` through the shared MCP server
are supplied upstream. Unexpected function/native tool events are blocked before
being forwarded to the CLI, recorded as `tool_profile_mismatch`, and not retried.

The loopback Responses proxy is deliberately **not byte-transparent**:

- It preserves the CLI's original request, then replaces tool schemas with the
  three shared MCP functions and explicitly sets `max_output_tokens` to the
  harness budget. Flat provider function names are mapped to Codex MCP namespaces;
  arguments and call IDs are retained. Actual instructions and conversation input
  are captured so this transformation can be reviewed.
- For subscription CLI WebSocket traffic, it records each original JSON request
  and bridges generation to an upstream HTTP SSE request. When a frame references
  the last response ID, it materializes the preceding input/output plus new input;
  an unknown ID is rejected. This preserves the visible conversation rather than
  relying on undocumented upstream WebSocket state.
- `generate=false` WebSocket connection prewarm receives a local, empty control
  acknowledgement with no upstream inference. Its original frame and local reply
  are saved separately as `control_NNN.json` and `cli_controls`, not `api_calls` or
  generated answers. No synthetic scientific answer or tool-discovery generation
  is inserted into the model history.

CLI configuration and event format were checked against the
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and [noninteractive CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
No documented Codex setting for a per-response output ceiling was found. The
recorded proxy injection is therefore explicit; it must not be described as a
native CLI setting. A response echoing a different budget is rejected. If the
response omits that field, `output_budget_echo_matches` is null: sending a budget
is not proof the server honors it. In particular, acceptance of the injected field
by the real subscription backend or a relay remains unverified; rejection is a
recorded failure, never a reason to silently remove the field or switch models.

Defaults remain `effort=max`, four questions concurrently, 3600 seconds per started
question and three total attempts with 2/4-second backoff. There is no CLI turn
limit or whole-question token ceiling. All generation calls, including any CLI
helper/compaction request, must match the model/effort/tool profile. API provider
transport retries are set to zero. The reserved built-in subscription provider
cannot be redefined by current Codex; the proxy blocks any further upstream calls
once an observable failure occurs, so built-in retries cannot create additional
model generations within that failed attempt. Prewarm controls are not model calls.
Hidden provider retries remain unobservable. Quota exhaustion and 401/403 are
recorded and stop the batch. Refusal, token/context limits, model/profile mismatch
and wrong scientific answers never select retries.

Every attempt retains CLI trace JSONL, stderr, answer file, MCP tool log,
original/effective model requests, raw upstream SSE and parsed per-call output and
usage. The CLI must exit successfully with a unique completed turn, and its final
answer must exactly match the last API answer. Partial traces, completed tool
turns or an answer file alone do not constitute a completed question. Returned
reasoning text, summaries and encrypted fields are stored as supplied; unavailable
reasoning is not reconstructed. CLI turn usage and per-call API usage are kept
separately and are not a guarantee about billing. Keys and subscription tokens are
redacted from retained records. Normal interruption cleanup and persistence have
the same limitations described for Claude Code above.

A single user question is passed unchanged on stdin. Leading system messages are
appended as developer instructions; few-shot conversations are preserved as
labelled role/content JSON inside the CLI prompt, not as native API-role history.
The constructor makes no model requests and requires the same Linux isolation
preflight as Claude Code. Local Mac validation substitutes that preflight and a
fake MCP server; it does not prove remote compute isolation or live provider access.

Example only; use the full provider Responses endpoint and an externally pinned
catalog rather than assuming Codex can discover these third-party models:

```bash
lm_eval --model sde-codex \
  --model_args 'model=deepseek-flash,max_tokens=128000,record_dir=results/flash_codex_max/raw,codex_bin=/opt/sde/codex,cli_version=0.162.0,tool_python=/opt/sde/tools-venv/bin/python,model_catalog=/opt/sde/catalogs/flash.json,base_url=https://api.deepseek.com/responses,api_key_env=DEEPSEEK_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3' \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/flash_codex_max
```

For a subscription profile, explicitly supply `auth_mode=chatgpt`, `auth_file`,
its required endpoint and a catalog/model available to that account. No migration
command launches an experiment. Tests cover offline failure cases and real local
Codex CLI against mocked model/MCP services with fake credentials. Those CLI probes
may attempt authentication control-plane refresh with the fake tokens; they do
not call a live model or use personal credentials. A separately authorized live
smoke test is still required before full evaluation. Status-aware summaries and
per-repeat accuracy aggregation remain pending.
