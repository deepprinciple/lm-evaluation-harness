# SDE agentic benchmark-source exclusion

Policy ID: `benchmark-source-exclusion-v1`. This is a change to the experiment
protocol, not a guarantee that arbitrary tool output or training data is free
of benchmark information.

## Forwarded content

The guard recognizes the SDE question datasets and recognizable mirrors,
SCILLM-benchmarks, GPQA answer/evaluation pages, benchmark repositories/local
result paths, and explicit recorded-answer claims. It does not consult a
question's reference answer or score to decide what to withhold.

When a returned tool block matches, the entire block is replaced by a fixed
notice asking for independent scientific sources or computation. Removing only
links is insufficient because a search summary may already contain the answer.
Normal scientific sources and numerical output remain unchanged. Search,
terminal, file tools, model ID, effort, output budget and iteration policy are
not disabled or reduced by this change.

| Route | Enforcement point |
|---|---|
| Claude Code | `tool_result` contents filtered before the next model request, including subagent requests through the proxy |
| Codex terminal/custom tools | `function_call_output` and `custom_tool_call_output` filtered after WebSocket history materialization and before forwarding |
| Codex subscription native search | Readable search response inspected; a matching response is withheld in full, including its encrypted output, with HTTP 422 `benchmark_source_blocked` |
| DeepSeek native-search bridge | Results, summary and citations inspected; a match returns an empty, failed tool result rather than partial contaminated evidence |

Native search remains the search backend. A local source block is a tool-level
rejection, not an authentication failure or a reason to switch models.

The initial question, system instructions and ordinary assistant text are not
scanned as tool results. The guard never treats the evaluator's stored reference
answer as evidence that the model was exposed to it.

## Audit trail and resume

Forwarded API payloads contain the filtered tool results. `source_filters` stores
the replaced original content, rule names, location and SHA256 in the separate
API audit record; `source_filter_count` is included in the compact call summary.
Native search and bridge provider responses are retained in the raw audit files.
An audit must distinguish withheld raw evidence from content forwarded to the
primary model; a blocked source in a raw provider response alone does not mean
it reached the primary model.

Both agent adapters record the policy ID and guard-file SHA256 in their request
identity. Runs made before this policy must not be reused with `--resume` under
the new identity. Use a new output directory and a fresh question session;
previously exposed conversation history cannot be made clean retroactively.
Existing old outputs are retained for review.

## Limits

This guard does not provide filesystem/network isolation. It can miss unknown
mirrors, answer-only text without recognizable provenance, images, opaque or
custom encodings, and information the provider used internally. In particular,
web search executed inside a single hosted Responses API generation cannot be
filtered before the provider's model sees it; do not describe that route as
protected by the standalone subscription-search guard.

Continue reviewing new traces for reference exposure. Do not accept a higher
score as evidence that an answer is clean. Rejected or timed-out reruns must not
fall back to an older contaminated answer in the official comparison.
