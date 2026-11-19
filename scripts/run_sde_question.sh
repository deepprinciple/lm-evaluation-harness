#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_sde_question.sh --group GROUP --output DIR [--repeat N] [--mode smoke|full] [--resume] [--dry-run]
GROUP: flash opus5.5 astra flash@claude_code opus5.5@claude_code
       flash@codex astra@codex astra@codex_subscription
Defaults: --repeat 1 --mode full (all four domains).
--output is required; there is no default output directory.
Relative output paths are resolved from the repository root.
Environment: SDE_CONCURRENCY, SDE_TIMEOUT, SDE_ATTEMPTS,
             SDE_TASKS, SDE_OPUS_ROOT, SDE_ASTRA_URL, SDE_ASTRA_KEY_ENV,
             CLAUDE_BIN, CODEX_BIN, NODE_BIN, TOOL_PYTHON, SDE_CATALOG_DIR,
             SDE_CODEX_AUTH_FILE.
--dry-run prints the command without calling APIs or requiring credentials.
--resume keeps normal answers and reopens technical failures with a new round budget.
EOF
}
fail() { printf '%s\n' "$*" >&2; exit 1; }
group=''
output=''
mode=full
repeats=1
dry_run=false
resume=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group|--output|--repeat|--mode)
      [[ $# -ge 2 && $2 != --* ]] || fail "Missing value for $1."
      case "$1" in
        --group) group=$2 ;;
        --output) output=$2 ;;
        --repeat) repeats=$2 ;;
        --mode) mode=$2 ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=true; shift ;;
    --resume) resume=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1; use --help." ;;
  esac
done
[[ -n $group ]] || fail 'Specify --group; use --help to list groups.'
[[ -n $output ]] || fail 'Specify --output DIR; there is no default output directory.'

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo"
case "$group" in
  flash|opus5.5|astra) default_concurrency=32 ;;
  flash@claude_code|opus5.5@claude_code|flash@codex|astra@codex|astra@codex_subscription) default_concurrency=16 ;;
  *) fail "Unknown group: $group" ;;
esac
case "$mode" in
  smoke)
    default_concurrency=1; default_timeout=600; default_attempts=1
    tasks=${SDE_TASKS:-science_chem_quantum_software}
    ;;
  full)
    default_timeout=3600; default_attempts=3
    tasks=${SDE_TASKS:-science_biology,science_chemistry,science_materials,science_physics}
    ;;
  *) fail "Unknown mode: $mode" ;;
esac
concurrency=${SDE_CONCURRENCY:-$default_concurrency}
timeout=${SDE_TIMEOUT:-$default_timeout}
attempts=${SDE_ATTEMPTS:-$default_attempts}
for number in "$concurrency" "$timeout" "$attempts" "$repeats"; do
  [[ $number =~ ^[1-9][0-9]*$ ]] || fail 'Concurrency, timeout, attempts and repeat must be positive integers.'
done
opus_root=${SDE_OPUS_ROOT:-https://mira.grecharge.ai}
astra_url=${SDE_ASTRA_URL:-https://mira.grecharge.ai/v1/responses}
astra_key=${SDE_ASTRA_KEY_ENV:-OPENAI_API_KEY}
key_env=''
catalog=''
auth_file=''
cli=''
case "$group" in
  flash)
    adapter=sde-chat-completions; key_env=DEEPSEEK_API_KEY
    args='model=deepseek-flash,reasoning_effort=max,thinking=enabled'
    ;;
  opus5.5)
    adapter=sde-messages; key_env=ANTHROPIC_API_KEY
    args="model=claude-opus-5-5,effort=max,base_url=${opus_root%/}/v1/messages,api_key_env=$key_env,thinking_display=summarized"
    ;;
  astra)
    adapter=sde-responses; key_env=$astra_key
    args="model=gpt-6-astra,effort=max,reasoning_summary=auto,base_url=$astra_url,api_key_env=$key_env"
    ;;
  *)
    node=${NODE_BIN:-$(command -v node || true)}
    tool_python=${TOOL_PYTHON:-$(command -v python || true)}
    [[ -n $node && -n $tool_python ]] || fail 'Activate the Python environment and configure NODE_BIN.'
    case "$group" in
      flash@claude_code|opus5.5@claude_code)
        adapter=sde-claude-code
        cli=${CLAUDE_BIN:-$repo/.tooling/claude/node_modules/.bin/claude}
        runtime="claude_bin=$cli,cli_version=2.1.296,tool_python=$tool_python,node_bin=$node,effort=max"
        if [[ $group == flash@claude_code ]]; then
          key_env=DEEPSEEK_API_KEY
          args="model=deepseek-flash[1m],api_model=deepseek-flash,$runtime,base_url=https://api.deepseek.com/anthropic,api_key_env=$key_env"
        else
          key_env=ANTHROPIC_API_KEY
          args="model=claude-opus-5-5,api_model=claude-opus-5-5,$runtime,base_url=${opus_root%/},api_key_env=$key_env"
        fi
        ;;
      *)
        adapter=sde-codex
        cli=${CODEX_BIN:-$repo/.tooling/codex/node_modules/.bin/codex}
        runtime="codex_bin=$cli,cli_version=0.162.1,tool_python=$tool_python,node_bin=$node,effort=max,reasoning_summary=auto"
        catalog_dir=${SDE_CATALOG_DIR:-$repo/config/sde-catalogs}
        if [[ $group == flash@codex ]]; then
          key_env=DEEPSEEK_API_KEY; catalog=$catalog_dir/flash.json
          args="model=deepseek-flash,$runtime,model_catalog=$catalog,base_url=https://api.deepseek.com/responses,api_key_env=$key_env,search_backend=deepseek_anthropic_bridge"
        else
          catalog=$catalog_dir/astra.json
          args="model=gpt-6-astra,$runtime,model_catalog=$catalog"
          if [[ $group == astra@codex ]]; then
            key_env=$astra_key
            args="$args,base_url=$astra_url,api_key_env=$key_env,search_backend=native_responses_hosted"
          else
            auth_file=${SDE_CODEX_AUTH_FILE:-$HOME/.codex/auth.json}
            args="$args,auth_mode=chatgpt,auth_file=$auth_file,base_url=https://chatgpt.com/backend-api/codex/responses"
          fi
        fi
        ;;
    esac
    ;;
esac
args="$args,max_tokens=128000,num_concurrent=$concurrency,timeout=$timeout,max_attempts=$attempts,record_dir=$output/raw,resume=$resume"
cmd=(python -m lm_eval --model "$adapter" --model_args "$args"
     --gen_kwargs max_tokens=128000 --tasks "$tasks" --repeats "$repeats"
     --num_fewshot 0 --seed 1 --apply_chat_template --log_samples
     --output_path "$output")
if [[ $mode == smoke ]]; then cmd+=(--limit 1); fi
if [[ $dry_run == true ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
[[ -z $key_env || $key_env =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail 'Invalid API-key environment variable name.'
if [[ -n $key_env ]]; then
  [[ -n ${!key_env:-} ]] || fail "Export $key_env before running (key value is never printed)."
fi
[[ -z $cli || -x $cli ]] || fail "CLI not installed: $cli; see README_new.md."
[[ -z $catalog || -f $catalog ]] || fail "Missing catalog: $catalog; obtain the matching experiment catalog first."
[[ -z $auth_file || -f $auth_file ]] || fail 'Codex login file missing; run codex login first.'
python -c 'import lm_eval, numpy, scipy, sympy, rdkit, math_verify'
if [[ -n $cli ]]; then
  "$tool_python" -c 'import numpy, scipy, sympy, rdkit, requests'
fi
printf 'Group=%s mode=%s repeat=%s resume=%s concurrency=%s timeout=%ss attempts=%s output=%s\n' \
  "$group" "$mode" "$repeats" "$resume" "$concurrency" "$timeout" "$attempts" "$output"
exec "${cmd[@]}"
