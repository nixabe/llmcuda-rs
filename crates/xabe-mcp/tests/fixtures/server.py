"""Deterministic MCP stdio fixture; writes protocol messages only to stdout."""
import json
import os
import sys
import threading
import time

lock = threading.Lock()
log_path = os.path.join(sys.argv[1], f"events-{os.getpid()}.jsonl")


def log(message):
    with open(log_path, "a", encoding="utf8") as target:
        target.write(json.dumps(message) + "\n")


def reply(request, result):
    with lock:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}) + "\n")
        sys.stdout.flush()


def slow(request):
    time.sleep(10)
    reply(request, {"content": [{"type": "text", "text": "late"}]})


for line in sys.stdin:
    request = json.loads(line)
    log(request)
    method = request["method"]
    if method == "initialize":
        reply(request, {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}})
    elif method == "tools/list":
        cursor = request.get("params", {}).get("cursor")
        names = ["slow", "huge", "error"] if cursor else ["echo"]
        result = {"tools": [{"name": name, "description": name, "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}} for name in names]}
        if not cursor:
            result["nextCursor"] = "page2"
        reply(request, result)
    elif method == "tools/call":
        name = request["params"]["name"]
        if name == "slow":
            threading.Thread(target=slow, args=(request,), daemon=True).start()
        elif name == "huge":
            reply(request, {"content": [{"type": "text", "text": "x" * 10000}]})
        elif name == "error":
            reply(request, {"isError": True, "content": [{"type": "text", "text": "operation failed"}]})
        else:
            reply(request, {"content": [{"type": "text", "text": "ok"}], "structuredContent": {"arguments": request["params"].get("arguments"), "pid": os.getpid()}})
    elif method == "ping":
        reply(request, {})
