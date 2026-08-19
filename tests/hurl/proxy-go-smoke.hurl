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
jsonpath "$.lock_queue" count == 0
