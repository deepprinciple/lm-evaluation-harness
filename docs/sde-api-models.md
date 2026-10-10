# Recorded SDE API models

`sde-chat-completions` is a no-tools DeepSeek Chat Completions adapter. It uses
harness tasks, chat formatting, filters and metrics, and returns execution metadata
alongside generated text. Legacy model registrations keep their existing behavior.
The `sde-messages` adapter adds an Opus Messages SSE profile using the same
request lifecycle. Responses and CLI agent adapters are not implemented yet.

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
