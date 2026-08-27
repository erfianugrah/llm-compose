# proxy-go live smoke - run against the soak port:
#   hurl --variables-file - tests/hurl/proxy-go-smoke.hurl  (base=http://127.0.0.1:11435)
# Covers: health, models, resident chat, lock FIFO (409/202/grant),
# serve-in-place capability routing (R3), Anthropic shim (R5), cleanup.

GET {{base}}/health
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"

GET {{base}}/v1/models
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.data" count >= 10

# Resident model serves with no swap.
POST {{base}}/v1/chat/completions
Content-Type: application/json
{"model":"qwen38","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.role" == "assistant"

# Lock qwen38 for owner smoke-a.
POST {{base}}/mode
Content-Type: application/json
{"lock":"qwen38","owner":"smoke-a"}
HTTP 200
[Asserts]
jsonpath "$.locked" == "qwen38"
jsonpath "$.lock_owners" count == 1
jsonpath "$.lock_owners[0]" == "smoke-a"
jsonpath "$.lock_expires_at" isInteger
jsonpath "$.lock_ttl_seconds" isInteger

# Lock renewal: the owner heartbeats the TTL; non-holders get 409.
POST {{base}}/mode
Content-Type: application/json
{"renew":true,"owner":"smoke-a"}
HTTP 200
[Asserts]
jsonpath "$.renewed" == true
jsonpath "$.locked" == "qwen38"
jsonpath "$.lock_expires_at" isInteger

POST {{base}}/mode
Content-Type: application/json
{"renew":true,"owner":"smoke-nobody"}
HTTP 409

# GET /mode exposes the lock expiry.
GET {{base}}/mode
HTTP 200
[Asserts]
jsonpath "$.locked" == "qwen38"
jsonpath "$.lock_expires_at" isInteger

# Contended lock without wait: 409, no hijack.
POST {{base}}/mode
Content-Type: application/json
{"lock":"gemma4","owner":"smoke-b"}
HTTP 409

# With wait: 202, queued at position 1.
POST {{base}}/mode
Content-Type: application/json
{"lock":"gemma4","owner":"smoke-b","wait":true}
HTTP 202
[Asserts]
jsonpath "$.queued" == true
jsonpath "$.position" == 1

# R3 serve-in-place: "loop" (code-only) requested with vision capability
# while vision-capable qwen38 is resident + locked -> no swap, no 422.
POST {{base}}/v1/chat/completions
Content-Type: application/json
X-LLM-Capability: vision
{"model":"loop","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 200
[Asserts]
jsonpath "$.model" contains "Qwen3.8"

# R5 Anthropic shim (Claude Code surface).
POST {{base}}/v1/messages
Content-Type: application/json
{"model":"qwen38","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}
HTTP 200
[Asserts]
jsonpath "$.type" == "message"
jsonpath "$.role" == "assistant"

# Cleanup: drop smoke-b's queue entry, then force-clear the lock.
POST {{base}}/mode
Content-Type: application/json
{"lock":false,"owner":"smoke-b"}
HTTP 200

POST {{base}}/mode
Content-Type: application/json
{"lock":false}
HTTP 200
[Asserts]
jsonpath "$.locked" == null

GET {{base}}/mode
HTTP 200
[Asserts]
jsonpath "$.locked" == null
jsonpath "$.lock_expires_at" == null
jsonpath "$.lock_queue" count == 0

# Ephemeral preset registry (Task 4b): register a throwaway, list shows it,
# delete removes it. Never touches the live models/ dir.
POST {{base}}/v1/presets
Content-Type: application/json
{"name":"smoke-ephemeral","vram_gb":1.0,"model":{"repo":"org/x","file":"smoke-ephemeral.gguf"},"runtime":{"context_size":4096,"parallel_slots":1}}
HTTP 201
[Asserts]
jsonpath "$.registered" == "smoke-ephemeral"
jsonpath "$.model_id" == "smoke-ephemeral"

# Collision with an on-disk preset is rejected.
POST {{base}}/v1/presets
Content-Type: application/json
{"name":"qwen38-dup","vram_gb":1.0,"model":{"repo":"org/x","file":"Qwen3.8-27B-Q4_K_M.gguf"},"runtime":{"context_size":4096}}
HTTP 409

DELETE {{base}}/v1/presets/smoke-ephemeral
HTTP 204

# Task routing (R2/R5): unknown alias 404; resident alias served in place.
POST {{base}}/v1/chat/completions
Content-Type: application/json
{"model":"auto:nope","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 404

# "auto" resolves to the default chain. With qwen38 resident (first in the
# default chain) it serves in place - no swap.
POST {{base}}/v1/chat/completions
Content-Type: application/json
{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 200
[Asserts]
jsonpath "$.model" contains "Qwen3.8"

# Aliases appear in the model listing: data count >= 22 (18 presets + 4 aliases).
GET {{base}}/v1/models
HTTP 200
[Asserts]
jsonpath "$.data" count >= 22
