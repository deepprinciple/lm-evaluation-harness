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
includes request time and backoff. Resuming an interrupted question opens a fresh
deadline window, so process downtime is not charged as a model timeout; every
window is saved under `deadline_windows`, and `elapsed_seconds` remains wall-clock
time since the first start. The timeout and attempt count are configurable; these
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
can trigger bounded technical retries. Failures before any generation is returned
(connection not established, or 429/503/529 without a quota marker such as
`insufficient_quota`) are retried without consuming the attempt budget, using
`Retry-After` (capped at 60 seconds) or capped exponential backoff, and only within
the question deadline. They are kept as attempts marked `budget_exempt`;
`response_metadata.budget_attempts` counts the attempts that used the budget.
500/502/504/524, quota errors and mid-stream failures still consume an attempt. A normal answer, a structured refusal or
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

## Independent repeats and status-aware accuracy

`--repeats N` overrides the task YAML's repeat count for `generate_until` tasks;
omitting it preserves the YAML value. The effective count is saved in task configs.
Use fresh request contexts: recorded SDE models reject `--cache_requests`, whose
cached Instances can retain an old repeat count or generation budget. Ordinary
model registrations retain their original filtering and result schema.

For example, to prepare a six-draw DeepSeek evaluation with the same 128000-token
budget on each draw (this command has not been executed):

```bash
lm_eval --model sde-chat-completions \
  --model_args model=deepseek-flash,max_tokens=128000,record_dir=results/flash_max_repeat6/raw,reasoning_effort=max \
  --gen_kwargs max_tokens=128000 --repeats 6 \
  --tasks science_chem_quantum_software --apply_chat_template \
  --log_samples --output_path results/flash_max_repeat6
```

All recorded SDE adapters add `sde.schema_version=1` and a `sde.tasks` object to
the aggregated results JSON. Each task contains:

- `request_status_counts`: final statuses across all requests and draws, counted
  once per request rather than once per filter. These are not retry-attempt counts.
- `repeats`: one entry per draw, with document counts, document-level
  `status_counts`, and `completed_doc_ids`. A document is completed only if all
  its requests have status `ok`; mixed failure kinds use `mixed_incomplete`.
- `metrics_all_documents`: original task filters/scoring/aggregation applied
  independently to this draw, with incomplete documents receiving zero credit.
- `metrics_completed_documents`: scores restricted to documents with `ok`
  responses; values are `null` if no documents completed.
- `across_repeats`: the unweighted mean and sample standard deviation of each
  draw's scores, plus `scored_repeats`. Standard deviation is `null` for fewer
  than two scores. Completed-only cohorts can differ between draws; this is a
  descriptive summary, not an estimate on a fixed common subset.

Sample JSONL entries additionally contain `repeat_results`, including each draw's
original document ID, request statuses, filtered answers and scalar scores. Use
these to intersect completed document IDs across models before making a common-
subset comparison. Correct and incorrect ordinary answers both remain completed;
missing answer tags do not become technical failures. No best-of-N selection,
majority vote, LLM judge or automatic domain reweighting is performed.

The existing terminal table and top-level `results` are preserved. They still
follow the original filters (usually `take_first`) and must not be interpreted as
six-draw averages. Use `sde.tasks` for independent-draw comparisons. With
`--predict_only`, only status/count reporting is added; repeat metrics stay empty.

This additional reporting supports single-rank SDE generation with scalar
accuracy/partial-credit metrics. It is not a general pooled BLEU/perplexity or
multi-rank evaluator. No API call is made by reporting; it rescores saved text
already returned in this evaluation. Actual provider/CLI compatibility and full
evaluation results still require separately authorized live runs.


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
Status-aware summaries and per-repeat accuracy use the shared reporting described
above; thinking text is not included in scored answers.


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
This intentionally tightens the old script's delta fallback. Malformed protocol,
empty output, 524, transport failure and incomplete generation use the existing
bounded technical retry policy. An effort echo that differs from the request is
recorded as terminal `effort_mismatch`, like a model mismatch: the endpoint's
profile is deterministic, so a retry would only repeat the cost. As in the other
profiles, refusals and token truncation are terminal, and scientific scores are never
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
This API-key profile supports Flash or Opus and retains the original question,
filter and metric pipeline. The current agent policy is **all available tools**:
`--tools default` and `--permission-mode bypassPermissions`, without `--restricted`,
a tool whitelist, a search-count cap or a CLI turn cap. Native WebSearch/WebFetch,
Bash, file operations and agent tools are permitted when the pinned CLI/provider
and execution environment actually supply them. A setting does not prove a tool
was used or that its backend works; the actual declarations and calls are saved.

Required arguments: `model` (CLI ID), `api_model` (Messages ID), `max_tokens`,
`record_dir`, `claude_bin`, exact `cli_version`, `tool_python` and provider-root
`base_url`. Select `DEEPSEEK_API_KEY` for Flash; `ANTHROPIC_API_KEY` is the default.
`SERPAPI_API_KEY` and `JINA_API_KEY` are no longer required or passed to the CLI.
For npm installs, use `node_bin` to select the Node runtime. CLI/Node/Python and
scientific package versions and source hashes are recorded. The constructor checks
versions without making a model request; the old Landlock/seccomp compute probe
is no longer a prerequisite for this unrestricted profile.

The CLI retains a clean per-attempt HOME and workspace, without personal config,
hooks, previous conversations or benchmark AGENTS files. Its PATH includes the
selected scientific Python environment. **These directories are not a filesystem
sandbox.** Native Bash and file tools bypass the CLI permission sandbox. Run the
agent in a dedicated container or VM that has no answer keys, scoring sources,
other question results or personal files mounted. Host/container restrictions,
uninstalled browsers and missing plugin credentials remain environmental limits;
this adapter neither removes them nor claims unavailable tools are supported.

The loopback Messages proxy keeps the actual request and raw SSE, including native
server search and client tool calls. The first primary request must match the pinned
model, max_tokens and effort. Later native requests retain their actual parameters;
budget differences are recorded without blocking subagents or compaction. They
are not automatically labelled as helper requests. Native search helpers are distinguished by their
server search/fetch tool declarations and recorded with their actual budgets and
usage rather than incorrectly applying the answer-generation budget to them.
Model mismatches, broken streams, nonzero CLI exits, explicit errors and unfinished
turns remain failures. A completed search/tool turn is not a completed question.
A final answer is scored only after the CLI's unique successful result. Refusal and
token/context stop reasons are taken from the main thread; a subagent refusal is
counted in `subagent_refusals` and does not override a successful final answer.
Scientific correctness never selects a retry.

Defaults: max effort, four concurrent questions, 3600 seconds per started question,
three total attempts with 2/4-second technical retry backoff. No search-count or
turn-count limit and no whole-question token cap are imposed. `max_tokens` is a
per-response budget; task/CLI generation kwargs retain their existing precedence.
Started attempts, helpers, tools and backoff share the question deadline. The local
proxy transparently retries the same pre-response failures as the API adapters
(connection errors and non-quota 429/503/529) before forwarding anything to the
CLI, recording each under the call's `upstream_retries`; the CLI receives one
response. Other upstream failures still end the attempt. A CLI connection reset
after the proxy has forwarded `message_stop` (or a complete JSON body) is kept as
`transport_tail_error` and the call is classified from the received response; an
earlier reset remains a failure. Observable
401/403 stops the batch; refusals and token/context limits are not technical retries.

Every attempt retains trace, stderr, tool calls, primary/helper request parameters,
raw responses and usage. Exact secret strings are redacted. Because every agent
request resends the whole conversation, `response_metadata.agent_metadata` in
sample logs keeps only per-call summaries (index, kind, status, HTTP status,
model, usage, profile checks, retry counts and raw paths); full payloads and
responses stay in the raw record and its `api_NNN.json`/`.sse` files. Normal cancellation
terminates the CLI process group; detached native processes and abrupt OS kills
can escape that cleanup, so a disposable per-run execution environment is required
for stronger containment. Resume uses a new CLI attempt and retains prior evidence.
The revised tool configuration and source hashes produce new record identities;
use a fresh directory for experiments rather than merging restricted and
unrestricted runs.

```bash
lm_eval --model sde-claude-code \
  --model_args 'model=deepseek-flash[1m],api_model=deepseek-flash,max_tokens=128000,record_dir=results/flash_cc_native/raw,claude_bin=/opt/sde/claude,cli_version=2.1.296,tool_python=/opt/sde/tools-venv/bin/python,node_bin=/opt/sde/node/bin/node,base_url=https://api.deepseek.com/anthropic,api_key_env=DEEPSEEK_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3' \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/flash_cc_native
```

For Opus select CLI/API model `claude-opus-5-5`, its provider root and key environment.
This command is an example, not an executed evaluation. Leading harness system
messages are appended; few-shot histories are passed as labelled role/content
JSON in the prompt, not native API-role history.

## Codex CLI agent

`sde-codex` retains the same question/filter/metric pipeline and now permits the
pinned Codex CLI's available native search, shell, file, image and agent tools.
The command sets `agents.enabled=true` and
`--dangerously-bypass-approvals-and-sandbox`. The previous capability-disable flags
and fixed three-tool MCP profile are removed. Personal memory, hooks, host skill
loading and user configs remain excluded to avoid carrying information between
experiments. Tool availability remains constrained by the selected model, provider,
CLI version and installed environment. Neither complete answers nor optional
browser/plugin support are inferred from configuration alone.

Required arguments: `model`, `max_tokens`, `record_dir`, `codex_bin`, exact
`cli_version`, `tool_python`, `model_catalog` and full Responses `base_url`.
The selected catalog must contain one matching model with the requested effort.
Its instructions and `tool_mode` are preserved, including `code_mode_only`;
the adapter no longer disables native code mode. Catalog/CLI/runtime hashes and
actual feature inventory are saved. The child environment is explicitly constructed;
native shells inherit that environment and can use its scientific Python PATH.
Temporary login-shell profiles restore that PATH after system shell startup;
they do not change the user's personal shell files.
They do not inherit the parent process's API-key environment.

Search routes are selected by `search_backend`. It is inferred only for the
verified API models (`deepseek-flash` -> `deepseek_anthropic_bridge`,
`gpt-6-astra` -> `native_responses_hosted`); any other API model must set
`deepseek_anthropic_bridge`, `native_responses_hosted` or `native_responses`
explicitly, so a renamed slug cannot silently lose search. The bridge stays
pinned to `deepseek-flash`, and subscription profiles always use
`native_subscription_web`.

Search routes:

- Astra profiles request Codex `web_search="live"`. The installed CLI/catalog
  determines whether search is hosted inside Responses or exposed as the
  standalone native web tool. Codex 0.162.1 subscription code mode uses
  `/alpha/search`; the proxy forwards that exact route only to the fixed official
  subscription backend, retaining commands, settings, its tool budget and raw
  results separately. It never substitutes a subscription search for an API-key
  profile. The tested relay returned 404 for `/v1/alpha/search`. For Astra API-key
  profiles the proxy instead adds the provider-native `web_search` tool to the
  existing Responses tool list, preserving all CLI tools and the original request.
  The 2026-10-11 smoke verified real hosted search and page opening on that same
  relay, without borrowing subscription credentials or third-party search.
- Flash@Codex uses the previously verified DeepSeek Anthropic native search API
  via a small client-side MCP bridge. The primary model stays on Responses.
  Codex's unsupported hosted Responses search route is disabled for this model;
  the bridge supplies `web_search` using DeepSeek's own service, without a
  per-question search cap. This is provider-native search, not a claim that Flash
  supports OpenAI's hosted Responses tool directly.
- The Flash helper uses `deepseek-flash`, thinking disabled, 4096 output tokens,
  `web_search_20250305` and `max_uses=1` for each helper request. The latter limits
  one helper invocation, not the number of searches allowed for a question.
  Helpers have separate raw request/response, search evidence and token records.
  Text alone is not accepted as proof of a real search. No Jina reader or SerpAPI
  fallback is installed. Reading pages remains a separate native-tool capability
  to validate.

The bridge sees only a loopback URL/token. The real DeepSeek key remains in the
parent proxy, is added only to the fixed upstream search request and is never
placed in MCP commands or the CLI environment.

Authentication profiles remain explicit:

- `auth_mode=api_key`: uses only `api_key_env` (default `DEEPSEEK_API_KEY`) and
  refuses an `auth_file`. The CLI receives a loopback token, not the real API key.
- `auth_mode=chatgpt`: requires an explicit ChatGPT `auth_file` and the exact
  `https://chatgpt.com/backend-api/codex/responses` endpoint; subscription tokens
  are never sent to a relay. A temporary 0600 auth copy outside retained results
  is deleted afterwards. The account identity is hashed and pinned. This does
  not establish current model access, network reachability or available quota.

The Responses proxy preserves CLI tools, tool_choice, namespaces, additional_tools
and tool history. It records the original payload and effective request; the only
budget normalization is explicit `max_output_tokens` injection, not native Codex
configuration. A conflicting budget echo is rejected; an absent echo does not
prove enforcement. The 2026-10-11 subscription smoke accepted and echoed 128000;
this verifies the request/echo, not independent enforcement or a provider maximum.
Unsupported requests fail visibly rather than silently removing
parameters, tools or changing models.

WebSocket requests are recorded and bridged to HTTP SSE. A known previous-response
ID is materialized from that response's own input/output plus new input, including
interleaved subagents; unknown IDs fail.
`generate=false` connection prewarm receives an empty local acknowledgement saved
as a control, not a billed inference or fabricated answer. Native tool SSE and
CLI events are retained in agent mode; no-tools API parsers continue rejecting
actual tool use. Missing terminals, identity changes, replayed sequence numbers
and stream errors are not accepted as completed answers.
Subscription terminals with an empty output snapshot are accepted only when all
indexed stream items were explicitly closed and validated. The reconstructed
snapshot is marked separately; raw SSE is unchanged. A client reset after a
successfully forwarded, validated terminal is retained as a tail transport event;
an early reset remains a failure. Native Messages citation deltas are retained
in agent mode without weakening the no-tools parser.

The default max effort/concurrency/deadline/technical retry budgets match Claude
Code. No human-imposed search or turn cap remains. Every attempt stores primary
API calls, separate Flash search_calls, traces, stdout/stderr and final answers.
Sample logs carry the same per-call summaries as Claude Code, not full payloads.
The CLI must complete a real turn and exit successfully; a retained answer file
alone does not establish completion. The final API answer must match the CLI's
answer and root thread identity where supplied. If no request identity is supplied,
the weaker answer-only attribution is recorded explicitly. Subscription quota and authentication
failures stop the batch. Refusals, truncation and wrong answers do not cause retries.

As with Claude Code, clean HOME/work directories do **not** restrict host file or
network access. Use a dedicated container/VM without labels, scoring sources or
other outputs mounted. Native tool capabilities, instructions, provider limits and
search implementations can differ between harnesses; these differences must be
reported rather than claimed as identical.

```bash
lm_eval --model sde-codex \
  --model_args 'model=deepseek-flash,max_tokens=128000,record_dir=results/flash_codex_native/raw,codex_bin=/opt/sde/codex,cli_version=0.162.0,tool_python=/opt/sde/tools-venv/bin/python,model_catalog=/opt/sde/catalogs/flash.json,base_url=https://api.deepseek.com/responses,api_key_env=DEEPSEEK_API_KEY,effort=max,num_concurrent=4,timeout=3600,max_attempts=3' \
  --gen_kwargs max_tokens=128000 \
  --tasks science_chem_quantum_software --limit 1 \
  --apply_chat_template --log_samples --output_path results/flash_codex_native
```

For subscription usage supply `auth_mode=chatgpt`, an explicit `auth_file`, the
required endpoint and a catalog/model available to that account. These examples
are not a full-run authorization. Tests cover self-contained fake CLI/provider
processes; live native-tool and subscription compatibility must be verified
separately. Historical restricted-run evidence remains unchanged.

### Live engineering smoke, 2026-10-11

On Mac, using Claude Code 2.1.296, Codex 0.162.1, Node 22.16.0, max effort,
128000 requested primary output tokens and a 600-second question deadline:

| Path | Primary answer | Actual search/read/compute/file checks |
| --- | --- | --- |
| Flash@Claude Code | Passed | All four passed |
| Opus@Claude Code | Passed | All four passed |
| Flash@Codex API | Passed | All four passed, DeepSeek native search bridge |
| Astra@Codex subscription | Passed | All four passed, official native `/alpha/search` |
| Astra@Codex relay API | Passed | All four passed after hosted Responses search and login-shell Python fixes |

These are synthetic engineering requests, not scientific scores or full-run
authorization. Subscription auth remained unchanged. Earlier failed attempts,
provider requests/results and CLI traces were preserved; live evidence is under
`results_question/lm_eval_native_smoke_20261011` in the surrounding SDE workspace.
The final relay rerun (`retry4`) completed in 83.4 seconds with native search/open
events, NumPy output `5.0` and file read-back `5.0`. Commentary remains in raw
evidence; only the final-answer phase is matched to the CLI answer. The scoped
adapter regression suite passed 477 tests, and changed Python files passed Ruff
and mypy. These checks do not prove that every possible native tool was exercised.

## Benchmark-source audit

The fixed MCP tool profile used to block benchmark sources (Hugging Face, the
SDE repository and paper) inside its own search/read tools. The native tool
profiles above do not apply that blocklist: native search, page reading, shell
and file tools are restricted only by the system prompt and the execution
environment. Audit every agent record directory before reporting its results:

```bash
python -m lm_eval.sde_audit results/flash_cc_native/raw > results/flash_cc_native/audit.json
```

The audit scans model-facing attempt evidence (provider streams, API calls, CLI
traces, answers, search and MCP tool records) for Hugging Face URLs or caches,
the SDE dataset organizations, the TMC ground-truth record, the SDE repository
and paper, and the record directory's own path. Main record JSON, CLI stderr,
the pinned catalog and harness path fields are skipped. Each flagged question is
listed with task, document, repeat, status, file and surrounding context for
manual review. A hit is not proof of leakage; no hit does not prove its absence,
because encrypted search results and pages without a recorded URL cannot be
checked. No API call is made. The audit does not change records or scores.
Provider-side domain filters for the DeepSeek search bridge are not configured
because DeepSeek support for them has not been verified.
