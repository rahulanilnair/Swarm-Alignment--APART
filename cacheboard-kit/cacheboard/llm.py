"""Minimal OpenAI-compatible chat client: standard library only.

Works with vLLM/SGLang servers and OpenRouter. The API key comes from an
environment variable that the caller names (for example via `kc env`), so no
key is ever passed on the command line or written to disk. Optional on-disk
cache: identical requests (same endpoint, model and body) return the stored
response, which makes reruns free and their outputs identical.
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class ChatError(RuntimeError):
    """A request that failed after all retries, or a malformed response."""


class ChatClient:
    def __init__(self, base_url, model, key_env=None, timeout=900, max_retries=6,
                 extra_body=None, cache_path=None, headers=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.key:
            raise ChatError(f"environment variable {key_env} is empty")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self.headers = dict(headers or {})
        self.cache_path = cache_path
        self._cache = {}
        self._lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]

    def describe(self):
        return {"base_url": self.base_url, "model": self.model, "extra_body": self.extra_body}

    def _body(self, messages, temperature, max_tokens, **params):
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        body.update({k: v for k, v in params.items() if v is not None})
        body.update(self.extra_body)
        return body

    def chat(self, messages, temperature=0.0, max_tokens=1024, use_cache=True, **params):
        """Send one chat completion. Returns a dict with content, reasoning, usage,
        finish_reason, model, latency_s and cached. Raises ChatError on failure."""
        body = self._body(messages, temperature, max_tokens, **params)
        key = hashlib.sha256(json.dumps([self.base_url, body], sort_keys=True).encode()).hexdigest()
        if use_cache and key in self._cache:
            return dict(self._cache[key], cached=True)
        started = time.time()
        raw = self._post(body)
        try:
            choice = raw["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"malformed response: {str(raw)[:300]}") from exc
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        response = {
            "content": message.get("content") or "",
            "reasoning": reasoning if isinstance(reasoning, str) else json.dumps(reasoning),
            "usage": raw.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
            "model": raw.get("model", self.model),
            "latency_s": round(time.time() - started, 3),
        }
        if use_cache and self.cache_path:
            with self._lock:
                self._cache[key] = response
                with open(self.cache_path, "a", encoding="utf8") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")
        return dict(response, cached=False)

    def _post(self, body):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                             headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    payload = json.load(resp)
                if isinstance(payload, dict) and payload.get("error"):
                    raise ChatError(f"provider error: {str(payload['error'])[:300]}")
                return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf8", "replace")
                last = ChatError(f"HTTP {exc.code}: {detail}")
                if exc.code not in RETRY_STATUS:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = ChatError(f"{type(exc).__name__}: {exc}")
            except ChatError as exc:
                last = exc
            if attempt < self.max_retries:
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        raise last


def extract_json(text):
    """Return the last top-level JSON object found in text, or None.

    Models often wrap JSON in prose or code fences; reasoning models sometimes
    echo a draft before the final object, so the last complete object wins.
    """
    if not text:
        return None
    found, depth, start, in_str, escape = None, 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    found = obj
    return found


class ClaudeCLIClient(ChatClient):
    """Claude through the local Claude Code CLI (`claude -p`), for a subscription login.

    Clean call: our system prompt replaces Claude Code's, no tools, no MCP servers, no settings
    sources (so no hooks or CLAUDE.md), no slash commands, no session persistence, an empty temp
    working directory, and the prompt on stdin. Temperature cannot be set through the CLI.
    The reply's usage is kept so callers can check the context really was clean."""

    def __init__(self, model, effort="low", timeout=900, cache_path=None, executable="claude", config_dir=None):
        super().__init__("claude-cli", model, key_env=None, timeout=timeout, cache_path=cache_path,
                         extra_body={"effort": effort})
        self.effort = effort
        self.executable = executable
        # A separate Claude Code profile keeps this login apart from the user's main CLI login.
        self.config_dir = config_dir or os.environ.get("CACHEBOARD_CLAUDE_CONFIG_DIR")

    def _post(self, body):
        import subprocess
        import tempfile
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        turns = [m for m in body["messages"] if m["role"] != "system"]
        prompt = turns[-1]["content"] if len(turns) == 1 else "\n\n".join(
            f"[{m['role']}]\n{m['content']}" for m in turns)
        cmd = [self.executable, "-p", "--model", self.model, "--system-prompt", system,
               "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
               "--output-format", "json", "--effort", self.effort]
        last = None
        for attempt in range(self.max_retries + 1):
            with tempfile.TemporaryDirectory() as cwd:
                try:
                    env = dict(os.environ)
                    if self.config_dir:
                        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(self.config_dir)
                    run = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                         cwd=cwd, timeout=self.timeout, env=env)
                    out = json.loads(run.stdout)
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                    last = ChatError(f"claude cli: {type(exc).__name__}: {str(exc)[:200]}")
                else:
                    if not out.get("is_error"):
                        usage = out.get("usage") or {}
                        return {"choices": [{"message": {"content": out.get("result") or ""},
                                             "finish_reason": out.get("stop_reason")}],
                                "usage": usage, "model": next(iter(out.get("modelUsage") or {self.model: 0}))}
                    last = ChatError(f"claude cli error: {str(out.get('result'))[:300]}")
                    if "disabled" in str(out.get("result")) or "login" in str(out.get("result")).lower():
                        raise last
            if attempt < self.max_retries:
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        raise last
