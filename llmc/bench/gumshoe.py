"""llmc bench gumshoe - research-agent protocol suite against candidates on the 5090.

Points the 18-case suite (canonical: gumshoe repo scripts/gumshoe-fixtures-draft.json)
at a candidate served through the llmc proxy, WITHOUT the live gateway. Tool
results are STUBS (short canned strings with chainable entities - e.g. the
osint_domain stub contains an IP so the g15 chain case is executable).

What this measures that the live-gateway run cannot:
  - raw JSON-protocol validity per model step (the gateway parse-and-nudge
    hides malformed output; here every step is parsed and counted)
  - first-tool selection, chain execution, steps-to-final, forced-final rate

Loop mechanics replicated 1:1 from gumshoe/gateway/agent.py (vendored with
provenance - track that file): build_system_prompt, _json_candidates,
parse_action, the nudge/unknown-tool/forced-final paths, temp 0.2,
max_tokens 1200, thinking off via chat_template_kwargs.

NOTE: the system prompt lines use python-interpreted unicode escapes so the
prompt the model sees stays byte-identical to production (which has em-dashes).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.bench.perf import wait_ready
from llmc.presets import load_all

PROXY = "http://127.0.0.1:11434"
MAX_STEPS = 6
MAX_RESULT_CHARS = 4000
CASES_PATH = Path.home() / "infra" / "gumshoe" / "scripts" / "gumshoe-fixtures-draft.json"

LogFn = Callable[[str], None]

# ── Vendored from gumshoe/gateway/agent.py (2026-08-16, track upstream) ──

TOOL_SPECS: dict[str, tuple[str, str]] = {
    "web_search":     ("general web search for facts / current info", '{"query": "..."}'),
    "fetch":          ("fetch & clean the text of one web page", '{"url": "https://..."}'),
    "osint_username": ("find accounts for a username across platforms", '{"username": "..."}'),
    "osint_email":    ("which platforms an email is on + breach exposure", '{"email": "..."}'),
    "osint_domain":   ("DNS, subdomains, certs, whois for a domain", '{"domain": "..."}'),
    "osint_ip":       ("geolocation, open ports, reverse DNS for an IP", '{"ip": "..."}'),
    "osint_url":      ("urlscan.io report for a URL / domain", '{"url": "..."}'),
    "osint_phone":    ("carrier / region metadata for a phone number", '{"phone": "+14155552671"}'),
    "osint_cve":      ("NVD lookup for a CVE id", '{"cve_id": "CVE-2021-44228"}'),
    "osint_threat":   ("VirusTotal reputation for a hash / URL / IP / domain", '{"target": "..."}'),
    "osint_geo":      ("find places NEAR a location - supermarkets, clinics, stations, cafes - from OpenStreetMap; also geocodes a place name to coordinates", '{"query": "Berlin, Germany", "kind": "supermarket"}'),
    "archive_lookup": ("what a web page USED to say and when it changed", '{"url": "example.com/pricing"}'),
}


def build_system_prompt(tool_names: list[str]) -> str:
    lines = [
        "You are a research assistant. At EVERY step you MUST output exactly ONE JSON object \u2014 NO prose, NO explanation, NOTHING else.",
        "",
        'Call a tool:  {"tool": "<name>", "args": {...}}',
        'Answer:       {"final": "<full markdown answer>"}',
        "",
        "Rules:",
        "  \u2022 For ANY question about products, recommendations, comparisons, current events, prices, people, or companies \u2014 call web_search FIRST, then answer.",
        "  \u2022 After a tool result, call another tool or give your {\"final\": ...} answer.",
        "  \u2022 Cite sources in the final answer when tools were used.",
        "  \u2022 NEVER output prose \u2014 always exactly ONE JSON object per step, no exceptions.",
        "",
        "Example first step for a product question:",
        '  {"tool": "web_search", "args": {"query": "best ergonomic coffee mug spill proof 2024"}}',
        "",
        "Available tools:",
    ]
    for n in tool_names:
        desc, args = TOOL_SPECS[n]
        lines.append(f"  {n} \u2014 {desc}; args {args}")
    return "\n".join(lines)


def _json_candidates(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    yield t
    depth = 0
    start = None
    for i, c in enumerate(t):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield t[start:i + 1]
                    start = None


def parse_action(text: str) -> Optional[dict]:
    for cand in _json_candidates(text or ""):
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
            return obj
    return None


# ── Stub tool results (canned, chainable entities) ─────────────────────

STUB_RESULTS: dict[str, str] = {
    "web_search": "[1] Tailscale vs WireGuard comparison (tailscale.com) - easiest multi-site mesh\n[2] nginx release list (nginx.org)\n[3] example result excerpt",
    "fetch": "Example Domain. This domain is for use in illustrative examples in documents.",
    "osint_username": "torvalds: GitHub (yes), Twitter (yes), Instagram (no), Mastodon (yes) - 3 found",
    "osint_email": "someone@example.com: registered on 2 platforms (gravatar, github); 0 breaches",
    "osint_domain": "google.com A 142.250.185.78; MX smtp.google.com; subdomains: www, mail, maps, drive; NS ns1.google.com",
    "osint_ip": "142.250.185.78: US, Mountain View, AS15169 Google; open ports 80, 443; rDNS lax1s11-in-f14.1e100.net",
    "osint_url": "example.com/login: 3 recent scans; verdict clean; last scan 2026-08-10",
    "osint_phone": "+14155552671: US, California, carrier Twilio, type voip",
    "osint_cve": "CVE-2021-44228: Log4Shell, CVSS 10.0 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H), CWE-502",
    "osint_threat": "275a021b...fd0f: 68/72 engines flag EICAR-Test-File (test signature, harmless)",
    "osint_geo": "Bukit Panjang: Hillion Mall (0.3km), Bukit Panjang Plaza (0.5km), Prime Supermarket (0.8km)",
    "archive_lookup": "supabase.com/pricing changed 2026-06-10 (+4.1KB) and 2026-03-02 (-1.2KB); 2 transitions in window",
}

# ── LLM fn + loop ──────────────────────────────────────────────────────

def make_llm_fn(model_id: str, proxy: str = PROXY) -> Callable[..., str]:
    def llm(messages: list[dict], max_tokens: int = 1200, temperature: float = 0.2) -> str:
        body = json.dumps({
            "model": model_id, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"{proxy}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"].get("content") or "").strip()
    return llm


def run_case(llm_fn: Callable[..., str], case: dict, max_steps: int = MAX_STEPS) -> dict:
    """Replicates gumshoe run_agent with stub tools. Returns the trace."""
    names = list(TOOL_SPECS)
    messages = [{"role": "system", "content": build_system_prompt(names)}]
    messages.extend(case.get("history", []))
    messages.append({"role": "user", "content": case["prompt"]})
    steps: list[dict] = []
    llm_steps = 0
    json_invalid = 0
    llm_errors = 0

    for _ in range(max_steps):
        try:
            raw = llm_fn(messages, max_tokens=1200, temperature=0.2)
            llm_steps += 1
        except Exception:
            llm_errors += 1
            if llm_errors >= 2:
                return {"steps": steps, "answer": "", "llm_steps": llm_steps,
                        "json_invalid": json_invalid, "forced": False, "error": True}
            continue
        llm_errors = 0
        action = parse_action(raw)
        if action is None:
            json_invalid += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             'Output ONLY a JSON object - no prose, no markdown, no explanation.'
                             ' Example: {"tool": "web_search", "args": {"query": "..."}}'
                             ' or {"final": "your answer here"}.'})
            continue
        name = action.get("tool")
        if not name and "final" in action:
            return {"steps": steps, "answer": str(action["final"]), "llm_steps": llm_steps,
                    "json_invalid": json_invalid, "forced": False}
        if name not in names and "final" not in action:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"Unknown tool {name!r}. Choose one of: {', '.join(names)}."})
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        messages.append({"role": "assistant", "content": raw})
        result = STUB_RESULTS.get(name, "(no result)")
        steps.append({"tool": name, "args": args})
        messages.append({"role": "user", "content":
                         f"Result of {name}:\n{str(result)[:MAX_RESULT_CHARS]}\n\n"
                         'Call another tool or give your {"final": ...} answer.'})

    # Budget exhausted: forced final
    try:
        forced = llm_fn(messages + [{"role": "user", "content":
                                     'Give your best final answer now as {"final": "..."}.'}],
                        max_tokens=800, temperature=0.3)
        llm_steps += 1
    except Exception:
        return {"steps": steps, "answer": "", "llm_steps": llm_steps,
                "json_invalid": json_invalid, "forced": True, "error": True}
    fa = parse_action(forced)
    answer = fa["final"] if (fa and "final" in fa) else (forced or "").strip()
    return {"steps": steps, "answer": str(answer), "llm_steps": llm_steps,
            "json_invalid": json_invalid, "forced": True}


# ── Checking (same oracle logic as the gateway-side runner) ────────────

def _expand_alts(expect_tools: list) -> list[set[str]]:
    return [set(t["any"]) if isinstance(t, dict) else {t} for t in expect_tools]


def _ordered_subsequence(actual: list[str], expected: list[set[str]]) -> bool:
    pos = 0
    for want in expected:
        found = False
        while pos < len(actual):
            if actual[pos] in want:
                found = True
                pos += 1
                break
            pos += 1
        if not found:
            return False
    return True


def check_trace(case: dict, trace: dict) -> bool:
    exp = case.get("expect", {})
    tools = [s["tool"] for s in trace["steps"]]
    want = _expand_alts(exp.get("tools", []))
    if want and not _ordered_subsequence(tools, want):
        return False
    for val in (exp.get("args") or {}).values():
        if not any(val in json.dumps(s["args"]) for s in trace["steps"]):
            return False
    if len(tools) > MAX_STEPS:
        return False
    return bool(trace["answer"])


# ── Runner ─────────────────────────────────────────────────────────────

def run_gumshoe(preset_names: list[str], repeats: int = 3,
                cases_path: Optional[Path] = None,
                rid: Optional[str] = None, log: LogFn = print) -> int:
    from llmc.cli import ProxyClient

    cases_path = Path(cases_path) if isinstance(cases_path, str) else (cases_path or CASES_PATH)
    cases = json.loads(cases_path.read_text())["cases"]
    presets = {p.name: p for p in load_all(store.REPO_ROOT / "models").values()}
    unknown = [p for p in preset_names if p not in presets]
    if unknown:
        log(f"unknown preset(s): {', '.join(unknown)}")
        return 2

    client = ProxyClient()
    rid = rid or store.run_id()
    rc = 0
    for name in preset_names:
        preset = presets[name]
        log(f"\n=== {name} ({len(cases)} cases x {repeats}) ===")
        client.set_lock(name, owner="bench")
        status, payload = client.set_mode("llm", model=name)
        if status != 200:
            log(f"  switch failed ({status}): {payload.get('error', payload)}")
            client.set_lock(False, owner="bench")
            rc = 1
            continue
        if not wait_ready(preset.model_id):
            log("  FAIL: model did not become ready")
            client.set_lock(False, owner="bench")
            rc = 1
            continue

        llm_fn = make_llm_fn(preset.model_id)
        per_case: dict[str, float] = {}
        total_steps = total_llm_steps = total_invalid = total_forced = 0
        for case in cases:
            hits = 0
            for _ in range(repeats):
                trace = run_case(llm_fn, case)
                hits += 1 if check_trace(case, trace) else 0
                total_steps += len(trace["steps"])
                total_llm_steps += trace["llm_steps"]
                total_invalid += trace["json_invalid"]
                total_forced += 1 if trace["forced"] else 0
                time.sleep(0.5)
            per_case[case["id"]] = round(hits / repeats, 3)
            log(f"  {case['id']}: {hits}/{repeats}")
        n_runs = len(cases) * repeats
        metrics = {
            "case_hit_rate": round(sum(per_case.values()) / len(per_case), 3),
            "json_valid_rate": round(1 - total_invalid / total_llm_steps, 4) if total_llm_steps else 0,
            "mean_steps": round(total_steps / n_runs, 2),
            "forced_final_rate": round(total_forced / n_runs, 3),
            "per_case": per_case,
        }
        rec = store.make_record(
            "gumshoe", name, store.REPO_ROOT / "models" / f"{name}.toml", metrics, rid,
            extra={"model_file": preset.model.file, "cases_path": str(cases_path),
                   "repeats": repeats},
        )
        store.append(rec)
        log(f"  -> hit={metrics['case_hit_rate']} json={metrics['json_valid_rate']} "
            f"steps={metrics['mean_steps']} forced={metrics['forced_final_rate']}")
        client.set_lock(False, owner="bench")
    return rc
