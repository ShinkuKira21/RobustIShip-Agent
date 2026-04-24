# Created by Edward Patch | RobustIShip - Agent

#!/usr/bin/env python3
"""
rebustIship - Local CLI agent with:
- Qwen on GPU server (heavy generation)
- Gemma on CPU (planning, validation, error recovery)
- Interactive mode with smart error recovery
- Persistent fix memory (remembers your corrections)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Persistent Fix Memory
# ----------------------------

class FixMemory:
    """Remembers fixes across the session"""
    def __init__(self):
        self.command_fixes = {}  # pattern -> replacement
        self.load_from_file()
    
    def load_from_file(self):
        """Load saved fixes from ~/.rebustIship_fixes.json"""
        try:
            fix_file = Path.home() / ".rebustIship_fixes.json"
            if fix_file.exists():
                with open(fix_file, "r") as f:
                    data = json.load(f)
                    self.command_fixes = data.get("command_fixes", {})
                    print(f"   📚 Loaded {len(self.command_fixes)} saved fixes")
        except Exception:
            pass
    
    def save_to_file(self):
        """Save fixes to ~/.rebustIship_fixes.json"""
        try:
            fix_file = Path.home() / ".rebustIship_fixes.json"
            with open(fix_file, "w") as f:
                json.dump({"command_fixes": self.command_fixes}, f, indent=2)
        except Exception:
            pass
    
    def add_fix(self, original: str, fixed: str):
        """Add or update a fix pattern"""
        # Extract the command pattern (e.g., "python" from "python -m py_compile ...")
        pattern = original.split()[0] if original else original
        if pattern and pattern != fixed.split()[0] if fixed else False:
            self.command_fixes[pattern] = fixed.split()[0]
            print(f"   💾 Remembered: use '{fixed.split()[0]}' instead of '{pattern}'")
            self.save_to_file()
    
    def apply_fixes(self, cmd: str) -> str:
        """Apply known fixes to a command"""
        result = cmd
        for pattern, replacement in self.command_fixes.items():
            if result.startswith(pattern + " ") or result == pattern:
                result = replacement + result[len(pattern):]
        return result

# Global fix memory instance
fix_memory = FixMemory()

# ----------------------------
# Helper functions
# ----------------------------

def _redact(text: str, secret: str | None) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "[REDACTED]")

def _preview(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "…[truncated]"

def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    for start in starts:
        for end in range(len(s) - 1, start, -1):
            if s[end] != "}":
                continue
            candidate = s[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
    return None

ALLOWED_TOOLS = {"write_file", "run_command"}

def normalize_action(action: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(action, dict):
        return None, "Action must be a JSON object"
    if "final" in action:
        if not isinstance(action.get("final"), str):
            return None, '"final" must be a string'
        return {"final": action["final"]}, None
    tool = action.get("tool")
    args = action.get("args") or {}
    if not isinstance(tool, str):
        return None, '"tool" must be a string'
    if not isinstance(args, dict):
        return None, '"args" must be an object'
    tool_aliases = {
        "create_file": "write_file", "write": "write_file", "save_file": "write_file",
        "exec": "run_command", "command": "run_command", "shell": "run_command",
    }
    tool = tool_aliases.get(tool, tool)
    if tool in {"create_folder", "create_dir", "create_directory", "mkdir"}:
        path = args.get("path") or args.get("dir") or args.get("directory")
        if isinstance(path, str) and path:
            return {"tool": "run_command", "args": {"cmd": f"mkdir -p {shlex.quote(path)}"}}, None
    if tool == "write_file":
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            return None, '"write_file" requires args.path string'
        if not isinstance(content, str):
            return None, '"write_file" requires args.content string'
        return {"tool": "write_file", "args": {"path": path, "content": content}}, None
    if tool == "run_command":
        cmd = args.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            return None, '"run_command" requires args.cmd string'
        # Apply persistent fixes
        cmd = fix_memory.apply_fixes(cmd)
        return {"tool": "run_command", "args": {"cmd": cmd}}, None
    return None, f'Unknown tool: {tool} (allowed: {sorted(ALLOWED_TOOLS)})'

def is_within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def write_file(root: Path, rel_path: str, content: str) -> None:
    path = (root / rel_path).resolve()
    if not is_within_root(root, path):
        raise PermissionError(f"Refusing to write outside root: {rel_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def run_command(root: Path, cmd: str, timeout_s: int = 1800, capture: bool = True) -> dict[str, Any]:
    dangerous = (" rm ", " mkfs", " dd ", " shutdown", " reboot", ":(){", ">|", " chown ", " chmod 777", " wipe")
    if any(x in f" {cmd.strip()} ".lower() for x in dangerous):
        return {"ok": False, "code": 126, "stdout": "", "stderr": "Refused dangerous command"}
    if capture:
        proc = subprocess.run(cmd, cwd=str(root), shell=True, text=True, capture_output=True, timeout=timeout_s, env=os.environ.copy())
        return {"ok": proc.returncode == 0, "code": proc.returncode, "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-20000:]}
    else:
        proc = subprocess.Popen(cmd, cwd=str(root), shell=True, text=True, env=os.environ.copy())
        try:
            proc.wait(timeout=timeout_s)
            return {"ok": proc.returncode == 0, "code": proc.returncode, "stdout": "", "stderr": ""}
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"ok": False, "code": -1, "stdout": "", "stderr": "Timeout"}

# ----------------------------
# HTTP Client for GPU Server
# ----------------------------

def http_post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    *,
    debug: bool = False,
    debug_show_messages: bool = False,
    debug_max_chars: int = 4000,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    if debug:
        msg_count = len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else None
        meta = {
            "url": url,
            "model": payload.get("model"),
            "messages": msg_count,
            "max_tokens": payload.get("max_tokens"),
            "stream": payload.get("stream"),
            "body_bytes": len(body),
        }
        print(f"[debug] HTTP POST meta={meta}", file=sys.stderr)
        if debug_show_messages and isinstance(payload.get("messages"), list):
            for i, m in enumerate(payload["messages"]):
                role = m.get("role")
                content = _preview(str(m.get("content", "")), 240)
                print(f"[debug] msg[{i}] role={role} content={content!r}", file=sys.stderr)
    try:
        with urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if debug:
                dt_ms = int((time.perf_counter() - t0) * 1000)
                status = getattr(resp, "status", None) or resp.getcode()
                print(f"[debug] HTTP {status} in {dt_ms}ms, bytes={len(raw)}", file=sys.stderr)
                print(f"[debug] HTTP response preview={_preview(_redact(raw, api_key), debug_max_chars)!r}", file=sys.stderr)
            return json.loads(raw)
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        if debug:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            print(f"[debug] HTTPError {e.code} in {dt_ms}ms", file=sys.stderr)
            print(f"[debug] HTTP error body preview={_preview(_redact(raw, api_key), debug_max_chars)!r}", file=sys.stderr)
        raise RuntimeError(f"HTTP {e.code} from server: {raw}") from e
    except URLError as e:
        if debug:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            print(f"[debug] URLError in {dt_ms}ms: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to reach server at {url}: {e}") from e

def chat_server(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    *,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 1500,
    debug: bool = False,
    debug_show_messages: bool = False,
    debug_max_chars: int = 4000,
) -> str:
    """Call GPU server for heavy generation"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    data = http_post_json(
        url,
        payload,
        api_key=api_key,
        debug=debug,
        debug_show_messages=debug_show_messages,
        debug_max_chars=debug_max_chars,
    )
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception:
        raise RuntimeError(f"Unexpected server response: {data}")

# ----------------------------
# Local CPU Model (Gemma)
# ----------------------------

class LocalCPUModel:
    def __init__(self, model_id: str, max_memory_gib: float = 8.0):
        self.model_id = model_id
        self.max_memory_gib = max_memory_gib
        self.model = None
        self.tokenizer = None
        self.last_response = None

    def load(self):
        print(f"🧠 Loading CPU model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        ram_gb = self.model.num_parameters() * 4 / 1e9
        print(f"   ✅ Loaded ({self.model.num_parameters() / 1e9:.1f}B params, ~{ram_gb:.1f} GiB RAM)")

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 256, temperature: float = 0.0) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_response = response
        return response.strip()

# ----------------------------
# Prompts for CPU Model
# ----------------------------

CPU_VALIDATION_PROMPT = """
You validate if the assistant's response is correct and safe.
Output ONLY JSON:
{"valid": true/false, "reason": "why", "fixed_action": {...} (optional)}
"""

# ----------------------------
# Error Recovery Functions
# ----------------------------

def handle_command_failure(tool: str, args: dict[str, Any], error: str, cpu_model, root: Path, messages: list) -> tuple[str, dict | None]:
    """
    Handle failed command with user choices.
    Returns: (action, fixed_action)
    """
    print(f"\n❌ Command failed: {tool}")
    print(f"   Error: {error[:500]}")
    print("\nOptions:")
    print("  [R]etry with CPU brain auto-fix")
    print("  [C]ustom prompt (tell CPU brain what to fix)")
    print("  [S]yntax check only (Python files)")
    print("  [M]anual test (you run, then report)")
    print("  [Cont]inue anyway (ignore error)")
    print("  [A]bort (stop and save conversation)")
    print("  [F]orget (clear a saved fix)")
    
    while True:
        choice = input("\nYour choice [R/c/s/m/cont/a/f]: ").strip().lower()
        
        if choice in ["", "r", "retry"]:
            print("   🤖 CPU brain auto-fixing the command...")
            fix_prompt = f"""
The command failed with error:
{error}

Original command: {tool} {json.dumps(args)}

Suggest a fixed version. Output ONLY JSON:
{{"tool": "write_file" or "run_command", "args": {{...}}}}
If cannot fix, output: {{"abort": true, "reason": "why"}}
"""
            fix_messages = [
                {"role": "system", "content": "You fix failed commands. Output ONLY valid JSON."},
                {"role": "user", "content": fix_prompt}
            ]
            fixed_response = cpu_model.generate(fix_messages, max_tokens=512, temperature=0.0)
            extracted = _extract_json_object(fixed_response)
            if extracted:
                try:
                    fixed_action, _ = normalize_action(json.loads(extracted))
                    if fixed_action and "abort" not in fixed_action:
                        print("   ✅ CPU brain fixed the command")
                        return ("retry", fixed_action)
                except Exception:
                    pass
            print("   ⚠️  CPU brain couldn't auto-fix. Try custom prompt or manual?")
            continue
            
        elif choice in ["c", "custom"]:
            print("\n📝 Tell the CPU brain what to fix (natural language):")
            print(f"   Original command: {args.get('cmd', '')}")
            print(f"   Error: {error[:200]}")
            custom_prompt = input("   Your instruction: ").strip()
            if custom_prompt:
                print(f"   🤖 CPU brain fixing with: {custom_prompt}")
                fix_prompt = f"""
The command failed with error:
{error}

Original command: {tool} {json.dumps(args)}

User instruction: {custom_prompt}

Based on the user's instruction, suggest a fixed version. Output ONLY JSON:
{{"tool": "run_command", "args": {{"cmd": "the fixed command"}}}}
"""
                fix_messages = [
                    {"role": "system", "content": "You fix commands based on user instructions. Output ONLY valid JSON."},
                    {"role": "user", "content": fix_prompt}
                ]
                fixed_response = cpu_model.generate(fix_messages, max_tokens=256, temperature=0.0)
                extracted = _extract_json_object(fixed_response)
                if extracted:
                    try:
                        fixed_action, _ = normalize_action(json.loads(extracted))
                        if fixed_action:
                            new_cmd = fixed_action.get('args', {}).get('cmd', '')
                            print(f"   ✅ CPU brain fixed: {new_cmd[:100]}...")
                            # Store the fix in memory
                            original_cmd = args.get('cmd', '')
                            if original_cmd and new_cmd:
                                fix_memory.add_fix(original_cmd, new_cmd)
                            return ("retry", fixed_action)
                    except Exception:
                        pass
                print("   ⚠️  CPU brain couldn't fix. Try manual?")
                continue
            else:
                print("   No instruction provided. Returning to menu.")
                continue
                
        elif choice in ["s", "syntax"]:
            if tool == "write_file" and args.get("path"):
                path = args["path"]
                # Try common Python interpreters
                for python_cmd in ["/opt/rocm-venv/bin/python", "/opt/rocm-venv/bin/python3", "python3", "python"]:
                    result = run_command(root, f"{python_cmd} -m py_compile {path}", timeout_s=30)
                    if result["ok"]:
                        print(f"   ✅ Syntax check passed with {python_cmd} for {path}")
                        break
                    elif "not found" in result["stderr"]:
                        continue
                    else:
                        print(f"   ❌ Syntax error with {python_cmd}:\n{result['stderr'][:500]}")
                else:
                    print(f"   ⚠️  Could not find a working Python interpreter.")
                    print("   Try [C]ustom prompt like: 'use /opt/rocm-venv/bin/python'")
                print("\nOptions: [R]etry, [C]ustom, [M]anual, [Cont]inue, [A]bort")
                continue
            else:
                print("   Syntax check only works for Python files")
                continue
                
        elif choice in ["m", "manual"]:
            print("\n📝 Manual test mode")
            print("   Please run the command manually.")
            print("   When done, reply with:")
            print("     'success' - command worked, finish")
            print("     'error: <description>' - what went wrong")
            print("     'continue' - proceed anyway")
            print("     'abort' - stop")
            
            while True:
                manual_response = input("\nResult: ").strip().lower()
                if manual_response == "success":
                    return ("success", None)
                elif manual_response.startswith("error:"):
                    error_desc = manual_response[6:].strip()
                    print(f"   🤖 CPU brain fixing based on: {error_desc}")
                    fix_messages = [
                        {"role": "system", "content": "Fix the command based on error. Output ONLY JSON."},
                        {"role": "user", "content": f"Command failed. Error: {error_desc}\nOriginal: {json.dumps(args)}\nSuggest fix."}
                    ]
                    fixed_response = cpu_model.generate(fix_messages, max_tokens=512, temperature=0.0)
                    extracted = _extract_json_object(fixed_response)
                    if extracted:
                        try:
                            fixed_action, _ = normalize_action(json.loads(extracted))
                            if fixed_action:
                                return ("retry", fixed_action)
                        except Exception:
                            pass
                    return ("abort", None)
                elif manual_response == "continue":
                    return ("continue", None)
                elif manual_response == "abort":
                    return ("abort", None)
                else:
                    print("   Unknown response. Try: success, error:..., continue, abort")
                    
        elif choice in ["cont", "continue"]:
            print("   ⏭️ Continuing despite error")
            return ("continue", None)
            
        elif choice in ["f", "forget"]:
            print("\n📝 Current saved fixes:")
            if fix_memory.command_fixes:
                for pattern, replacement in fix_memory.command_fixes.items():
                    print(f"   {pattern} -> {replacement}")
                pattern_to_forget = input("Enter pattern to forget (or 'all'): ").strip()
                if pattern_to_forget == "all":
                    fix_memory.command_fixes = {}
                    print("   ✅ All fixes cleared")
                elif pattern_to_forget in fix_memory.command_fixes:
                    del fix_memory.command_fixes[pattern_to_forget]
                    print(f"   ✅ Forgot fix for '{pattern_to_forget}'")
                else:
                    print(f"   Pattern '{pattern_to_forget}' not found")
                fix_memory.save_to_file()
            else:
                print("   No saved fixes")
            continue
            
        elif choice in ["a", "abort"]:
            print("   🛑 Aborted by user")
            return ("abort", None)
        else:
            print("   Invalid choice. Try: R, C, S, M, Cont, A, or F")

def confirm_execution(tool: str, args: dict[str, Any], is_retry: bool = False) -> tuple[bool, str]:
    """Ask user for confirmation before executing tools"""
    if is_retry:
        print(f"\n🔄 Retry: {tool}")
    else:
        print(f"\n⚠️  About to execute: {tool}")
    
    if tool == "write_file":
        print(f"   Path: {args.get('path')}")
        preview = args.get('content', '')[:200]
        if preview:
            print(f"   Content preview:\n{preview}...")
    elif tool == "run_command":
        print(f"   Command: {args.get('cmd')}")
    
    response = input("   Execute? [Y/n/syntax/abort]: ").strip().lower()
    
    if response in ["", "y", "yes"]:
        return (True, "execute")
    elif response in ["syntax", "s"]:
        return (False, "syntax")
    elif response in ["abort", "a"]:
        return (False, "abort")
    else:
        return (False, "skip")

# ----------------------------
# Main Agent
# ----------------------------

SYSTEM_PROMPT = """
You are a coding assistant running in a local CLI agent.
You MUST respond with exactly ONE JSON object per step with ONLY one of these keys:
- {"tool":"write_file","args":{"path":"relative/path","content":"file content"}}
- {"tool":"run_command","args":{"cmd":"command string"}}
- {"final":"string"}  (finish)
Rules:
- Always wrap your output in JSON.
- Prefer minimal safe changes.
- Never propose destructive commands.
- Do NOT include reasoning, analysis, or any other text outside the JSON.
- Note: write_file automatically creates parent directories.
- When the task is complete, output {"final":"your answer here"}
"""

def interactive_loop(cpu_model, args, root, fix_memory):
    """Interactive mode with conversation history and loop prevention"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Workspace root: {root}\nPermissions: write={'yes' if args.apply else 'no'}, run={'yes' if args.run else 'no'}\nYou are in interactive mode. Help the user with their tasks."}
    ]
    
    print("\n" + "=" * 60)
    print("💬 Interactive Mode Started")
    print("Commands: /exit, /clear, /status, /help, /save, /done, /fixes")
    print("=" * 60 + "\n")
    
    last_file = None
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd == "/exit":
                    print("👋 Goodbye!")
                    break
                elif cmd == "/clear":
                    messages = messages[:2]
                    print("🧹 Conversation history cleared.")
                    continue
                elif cmd == "/status":
                    print(f"📊 Messages in context: {len(messages)}")
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info()
                        print(f"   GPU VRAM: {(total-free)/1e9:.1f}GB / {total/1e9:.1f}GB")
                    if fix_memory.command_fixes:
                        print(f"   📚 Saved fixes: {len(fix_memory.command_fixes)}")
                    continue
                elif cmd == "/fixes":
                    if fix_memory.command_fixes:
                        print("📚 Saved fixes:")
                        for pattern, replacement in fix_memory.command_fixes.items():
                            print(f"   {pattern} -> {replacement}")
                    else:
                        print("No saved fixes")
                    continue
                elif cmd == "/save":
                    filename = f"conversation_{int(time.time())}.json"
                    with open(filename, "w") as f:
                        json.dump(messages, f, indent=2)
                    print(f"   ✅ Conversation saved to {filename}")
                    continue
                elif cmd == "/done":
                    print("✅ Manually marked as complete. Stopping.")
                    break
                elif cmd == "/help":
                    print("Commands:")
                    print("  /exit   - Exit interactive mode")
                    print("  /clear  - Clear conversation history")
                    print("  /status - Show memory usage and saved fixes")
                    print("  /fixes  - Show saved command fixes")
                    print("  /save   - Save conversation to file")
                    print("  /done   - Force mark task as complete")
                    print("  /help   - Show this help")
                    continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Process conversation
            step = 1
            max_steps = args.max_steps
            action = None
            task_complete = False
            
            while step <= max_steps and not task_complete:
                if args.verbose:
                    print(f"\n[Step {step}] Processing...")
                
                # Get response from GPU
                assistant = chat_server(
                    args.base_url,
                    args.model,
                    messages,
                    api_key=args.api_key,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                    debug=args.debug,
                )
                
                # Validate with CPU
                validation_messages = [
                    {"role": "system", "content": CPU_VALIDATION_PROMPT},
                    {"role": "user", "content": f"Validate this response: {assistant}"}
                ]
                validation = cpu_model.generate(validation_messages, max_tokens=256, temperature=0.0)
                
                # Parse action
                action = None
                action_error = None
                extracted = _extract_json_object(assistant)
                if extracted:
                    try:
                        parsed = json.loads(extracted)
                        action, action_error = normalize_action(parsed)
                    except Exception:
                        pass
                
                # Fix if needed
                if action is None or "valid\": false" in validation:
                    if args.verbose:
                        print(f"[Step {step}] CPU brain fixing response...")
                    fix_messages = [
                        {"role": "system", "content": "Fix this into valid JSON with proper tool format. Output ONLY the JSON."},
                        {"role": "user", "content": f"Original: {assistant}\nError: {action_error or 'Invalid JSON'}"}
                    ]
                    fixed = cpu_model.generate(fix_messages, max_tokens=512, temperature=0.0)
                    extracted = _extract_json_object(fixed)
                    if extracted:
                        try:
                            parsed = json.loads(extracted)
                            action, action_error = normalize_action(parsed)
                            if action:
                                assistant = fixed
                        except Exception:
                            pass
                
                if action is None:
                    consecutive_failures += 1
                    if args.verbose:
                        print(f"[Step {step}] Invalid response (failure {consecutive_failures}/{max_consecutive_failures})...")
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many failures. Aborting.")
                        return
                    messages.append({"role": "assistant", "content": assistant})
                    messages.append({"role": "user", "content": "Invalid JSON. Respond with exactly one JSON object."})
                    step += 1
                    continue
                
                consecutive_failures = 0
                
                if "final" in action:
                    print(f"\n🤖 Assistant: {action['final']}\n")
                    messages.append({"role": "assistant", "content": action['final']})
                    break
                
                # Execute tool
                tool = action.get("tool")
                tool_args = action.get("args") or {}
                
                if tool == "write_file":
                    last_file = tool_args.get("path")
                
                # Confirm execution
                if args.yes:
                    confirm = True
                    confirm_action = "execute"
                else:
                    confirm, confirm_action = confirm_execution(tool, tool_args)
                
                if confirm_action == "abort":
                    print("   🛑 Aborted by user")
                    if input("Save conversation? [y/N]: ").strip().lower() == "y":
                        filename = f"conversation_{int(time.time())}.json"
                        with open(filename, "w") as f:
                            json.dump(messages, f, indent=2)
                        print(f"   ✅ Conversation saved to {filename}")
                    return
                elif confirm_action == "syntax" and last_file:
                    # Try common Python interpreters with fixes applied
                    for python_cmd in ["python3", "python"]:
                        python_cmd = fix_memory.apply_fixes(python_cmd)
                        result = run_command(root, f"{python_cmd} -m py_compile {last_file}", timeout_s=30)
                        if result["ok"]:
                            print(f"   ✅ Syntax OK with {python_cmd}")
                            break
                        elif "not found" in result["stderr"]:
                            continue
                        else:
                            print(f"   ❌ Syntax error:\n{result['stderr'][:500]}")
                    else:
                        print("   ⚠️  Could not find working Python interpreter")
                    if input("\nExecute anyway? [y/N]: ").strip().lower() != "y":
                        result = {"ok": True, "skipped": True}
                        messages.append({"role": "assistant", "content": assistant})
                        messages.append({"role": "user", "content": json.dumps({"tool_result": result})})
                        step += 1
                        continue
                    confirm = True
                elif not confirm:
                    print("   ⏭️ Skipped")
                    result = {"ok": True, "skipped": True}
                    messages.append({"role": "assistant", "content": assistant})
                    messages.append({"role": "user", "content": json.dumps({"tool_result": result})})
                    step += 1
                    continue
                
                # Execute the command
                try:
                    if tool == "write_file":
                        write_file(root, tool_args["path"], tool_args["content"])
                        result = {"ok": True, "path": tool_args["path"]}
                        print(f"   ✅ File written: {tool_args['path']}")
                        if not args.yes:
                            if input("\n✅ File written. Is the task complete? [Y/n]: ").strip().lower() in ["", "y", "yes"]:
                                task_complete = True
                                break
                    elif tool == "run_command":
                        result = run_command(root, tool_args["cmd"])
                        if result["ok"]:
                            print(f"   ✅ Command executed")
                            if result["stdout"]:
                                print(f"   Output: {result['stdout'][:500]}")
                            if not args.yes:
                                if input("\n✅ Command succeeded. Is the task complete? [Y/n]: ").strip().lower() in ["", "y", "yes"]:
                                    task_complete = True
                                    break
                        else:
                            print(f"   ❌ Command failed: {result['stderr'][:200]}")
                            action_choice, fixed_action = handle_command_failure(
                                tool, tool_args, result['stderr'], cpu_model, root, messages
                            )
                            
                            if action_choice == "retry" and fixed_action:
                                action = fixed_action
                                tool = action.get("tool")
                                tool_args = action.get("args") or {}
                                if args.yes:
                                    confirm = True
                                else:
                                    confirm, _ = confirm_execution(tool, tool_args, is_retry=True)
                                if confirm:
                                    continue
                                else:
                                    result = {"error": "Retry cancelled"}
                            elif action_choice == "success":
                                result = {"ok": True, "manual_success": True}
                                print("   ✅ User confirmed success")
                                task_complete = True
                                break
                            elif action_choice == "continue":
                                result = {"ok": True, "manual_continue": True}
                                print("   ⏭️ Continuing despite error")
                            elif action_choice == "abort":
                                if input("Save conversation? [y/N]: ").strip().lower() == "y":
                                    filename = f"conversation_{int(time.time())}.json"
                                    with open(filename, "w") as f:
                                        json.dump(messages, f, indent=2)
                                    print(f"   ✅ Conversation saved to {filename}")
                                return
                    else:
                        result = {"error": f"Unknown tool: {tool}"}
                        print(f"   ❌ Unknown tool: {tool}")
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                    print(f"   ❌ Error: {result['error']}")
                    action_choice, fixed_action = handle_command_failure(
                        tool, tool_args, str(e), cpu_model, root, messages
                    )
                    if action_choice == "retry" and fixed_action:
                        action = fixed_action
                        tool = action.get("tool")
                        tool_args = action.get("args") or {}
                        continue                    
                    elif action_choice == "abort":
                        return
                
                messages.append({"role": "assistant", "content": assistant})
                messages.append({"role": "user", "content": json.dumps({"tool_result": result})})
                step += 1
            
            if step > max_steps:
                print("⚠️  Reached max steps without final answer.")
            elif task_complete:
                print("\n✅ Task complete! Ready for next request.\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

def main() -> int:
    parser = argparse.ArgumentParser(description="rebustIship - Local agent: GPU server + CPU brain")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="GPU server URL (Qwen)")
    parser.add_argument("--model", default="qwen/Qwen2.5-Coder-7B-Instruct", help="Main model on server")
    parser.add_argument("--cpu-model", default="google/gemma-4-E4B-it", help="CPU model for planning/validation")
    parser.add_argument("--cpu-model-max-memory-gib", type=float, default=8.0, help="RAM limit for CPU model")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-key", default=os.getenv("MODEL_API_KEY"))
    parser.add_argument("--prompt", help="One-shot prompt (if not provided, enters interactive mode)")
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true", help="Actually execute write_file and run_command")
    parser.add_argument("--run", action="store_true", help="Actually execute run_command (requires --apply)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmations (auto-yes)")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode even if --prompt provided")
    parser.add_argument("--clear-fixes", action="store_true", help="Clear all saved fixes on startup")
    args = parser.parse_args()

    if args.yes:
        args.apply = True
        args.run = True

    # Clear fixes if requested
    if args.clear_fixes:
        fix_memory.command_fixes = {}
        fix_memory.save_to_file()
        print("🧹 Cleared all saved fixes")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or None
    
    if hf_token is None:
        print("⚠️  HF_TOKEN not set; attempting anonymous download.", file=sys.stderr)

    print("=" * 60)
    print("🤖 rebustIship - Local Agent")
    print("=" * 60)
    
    # Load CPU model (Gemma)
    print("\n📦 Loading CPU brain (Gemma)...")
    cpu_model = LocalCPUModel(args.cpu_model, args.cpu_model_max_memory_gib)
    cpu_model.load()
    
    print("\n✅ Agent ready! Connecting to GPU server...")
    print(f"   GPU Server: {args.base_url}")
    print(f"   CPU Model: {args.cpu_model} (local)")
    if args.yes:
        print("   ⚠️  Auto-execution mode: --yes enabled, skipping confirmations")
    if fix_memory.command_fixes:
        print(f"   📚 Loaded {len(fix_memory.command_fixes)} saved fixes")
        for pattern, replacement in list(fix_memory.command_fixes.items())[:3]:
            print(f"      {pattern} -> {replacement}")
        if len(fix_memory.command_fixes) > 3:
            print(f"      ... and {len(fix_memory.command_fixes) - 3} more")
    print("=" * 60)

    root = Path(args.root).resolve()
    
    # Decide mode
    if args.interactive or (args.prompt is None):
        interactive_loop(cpu_model, args, root, fix_memory)
        return 0
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
