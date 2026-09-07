# 2026-09-08 - Consolidating the pi provider surface: one engine-neutral name

Status: PLAN. Not started. Written because the obvious version of this change
(rename the provider, delete the duplicate) silently breaks harness manifests
in nine repos, and because the two providers differ in a way that is not
cosmetic.

## The problem, in three parts

**1. The name is wrong.** pi's `llama-server` provider now serves NInfer as
well as llama.cpp. `llama-server/qwen38-ninfer` reads as a contradiction, and
the container/hostname/volume named `llama_server` genuinely IS llama.cpp - so
the same string means two different things depending on where it appears.

**2. There are two providers pointing at the same endpoint.**

| | `llama-server` | `external` |
|---|---|---|
| baseUrl | `http://localhost:11434/v1` | `http://127.0.0.1:11434/v1` |
| models | 8, registered LIVE from the proxy | 1, static |
| addressed by | preset stem (`qwen38`) | served alias (`qwen3.8-27b-nvfp4`) |
| `supportsReasoningEffort` | false | true |
| `maxTokens` | **16,384** (hardcoded in the extension) | **65,536** |

Same engine, same endpoint, two names, two conventions.

**3. The maxTokens difference is a live trap, measured 2026-09-08.** The same
one-line prompt to the same model:

- `pi --model llama-server/qwen38-ninfer` -> engine logs `max output 16,384`
- `pi --model external/qwen3.8-27b-nvfp4:medium` -> engine logs `max output 65,536`

16,384 is exactly the cap that truncated a 35,747-token xhigh response
mid-thought on 2026-09-07 and produced two loop iterations that exited 0
having written nothing. `llama-server-dynamic.ts` pins it deliberately, with a
comment: the proxy exposes no per-preset max-output field, and omitting
`maxTokens` from `registerProvider` crashes `formatTokenCount`. So the pin was
correct when written; it is now the reason the two paths behave differently.

**Consolidating onto `llama-server` as-is would reintroduce the truncation
trap for NInfer.** Fix the cap first, then consolidate.

## Blast radius (measured, not estimated)

`rg -l --hidden` for both provider strings over `~/dotfiles ~/infra
~/lexicanum`: **25 files across 10 repos.**

| repo | files | what |
|---|---|---|
| `lockstep` | 7 | 6 harness manifests + loop-harness.md |
| `memledger` | 5 | 4 harness manifests + a plan doc |
| `ai/llm-compose` | 4 | spike plan, `.pi/harness.json`, `scripts/pi-loop-agent.sh`, a Go test comment |
| `dotfiles` | 3 | loop skill `docs/models.md`, `docs/lessons.md`, `.pi/harness.json` |
| `crier`, `hearth`, `knotea`, `mnemosyne`, `monitoring-compose`, `research` | 1 each | harness manifests and plan docs |

Note `rg` skips hidden directories unless passed `--hidden`, so a search
without it misses every `.pi/harness*.json` - which is most of the surface.
That is how this looked like a two-file change at first.

## Order of work

Each step leaves the tree working. Do not reorder: 1 must precede 3, or the
consolidation ships the truncation cap.

### 1. Publish a per-preset output cap (removes the 16,384 pin)

- Preset schema: optional `runtime.max_output_tokens` (llama.cpp has no such
  concept - `n_predict=-1` is unlimited - so absent stays absent for those).
- `qwen38-ninfer.toml`: set it to 65536, matching what `external` already sends.
- Proxy `/v1/models`: publish it as `meta.max_output`.
- `llama-server-dynamic.ts`: `maxTokens: meta.max_output ?? 16384`. Keep the
  fallback - the comment's crash reason still holds for presets without one.
- Verify at the engine, not in config: `pi --model <provider>/qwen38-ninfer`
  must log `max output 65,536`. Correlate by prompt size or message count; the
  engine log interleaves with any live session.

### 2. Decide the name

Candidates, with the cost of each:

- `llmc` - matches the CLI and the stack, engine-neutral, short. Collides with
  nothing in the model-string namespace.
- `local` - clearest to a reader, but vague once a second local endpoint exists.
- `llmc-proxy` - explicit, longer in every manifest.

Recommendation: **`llmc`**. It is the name of the thing that actually serves
the request, and it survives adding a third engine.

Whatever is chosen, the extension file wants renaming too
(`llama-server-dynamic.ts` -> `llmc-dynamic.ts`), and the `LLAMA_SERVICE` /
`llama_server` container and volume names must NOT change - those describe
llama.cpp itself and are correct.

### 3. Rename, with the aliases in place first

pi resolves a bare model id across providers, so a flag-day rename breaks 25
files at once. Sequence:

1. Register the new provider name alongside the old one (the extension can
   register both; the static list can keep `llama-server` as a deprecated
   duplicate). Both resolve, nothing breaks.
2. Update the 25 references, repo by repo, verifying each repo's harness still
   resolves its rung 0. `loop run --dry` is the cheap check.
3. Remove the old name from the extension and the static list.
4. Grep again with the hidden-files flag before declaring it done.

### 4. Retire `external`

Once `qwen38-ninfer` is addressable by preset stem on the new provider,
`external` is a duplicate pointing at the same endpoint. Delete it from
`models.json`. Its `supportsReasoningEffort: true` is not needed: the proxy
injects the preset's effort when absent and coerces values the template
rejects, which is why `llama-server/qwen38-ninfer` works today despite the
provider declaring `supportsReasoningEffort: false`.

## Adjacent tech debt (do not fold into this change)

Recorded here so it is not rediscovered, not to widen the scope:

- `llmc/proxy.py` carries NInfer support that will drift from the Go proxy.
  It is the rollback lane only (cutover 2026-08-19), so drift is tolerable
  until a rollback is actually needed - at which point it is not. Either keep
  them in lockstep deliberately or write down that the rollback lane is
  engine-blind.
- `README.md` volume table still lists only the llama volumes; AGENTS.md has
  the `llmc-ninfer-models` row and the Engines section.
- `~/infra/secretctl/AGENTS.md` does not mention that sops input format is
  content-sniffed and binary envelopes are supported (2026-09-07 fix).
- 10 unpushed commits in `llm-compose`, 1 in `secretctl` as of 2026-09-08.
- The static `llama-server` model list in `models.json` has a null `maxTokens`
  for every entry and is only a fallback for when the proxy is unreachable.
  Once step 1 lands, that fallback disagrees with the live registration.

## Verify before done

- A hidden-inclusive grep for both old provider strings returns only
  historical references in plan docs (which should keep the old name, since
  they record what was true then).
- One harness per affected repo resolves rung 0: `loop run --dry` exits
  without a provider error.
- `pi --model llmc/qwen38-ninfer` logs `max output 65,536` at the engine and
  `thinking medium`.
- `pi --model llmc/qwen38` still serves through llama.cpp.
- `make test` and `go test ./...` green in llm-compose.
