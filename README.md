# RobustIShip Agent

RobustIShip is a self-healing, dual-model coding agent that runs entirely on your hardware. No API keys. No cloud. No limits.

**Qwen generates. Gemma guides. The system heals itself.**

---

## Why RobustIShip?

- **Self-healing**: Failed commands auto-diagnose and retry with targeted fixes
- **Quality-gated**: Every file scored before writing. Below 8/10? Auto-retries.
- **TDD Mode**: Tests written first, implementation follows, tests must pass
- **System Memory**: Learns your environment across projects. Fixes compound over time.
- **Cross-platform**: Linux, macOS, Windows. Auto-detects your OS, shell, Python path.
- **Terminal-first**: `/plan`, `/go`, `/fix` — same workflow, more power.
- **Open models**: Qwen 14B + Gemma 7.9B. Runs on your GPU + CPU.

---

## Quick Start

```bash
# 1. Start your Qwen server (vLLM, Ollama, etc.)
python run_qwen.py --model Qwen/Qwen2.5-Coder-14B-Instruct --serve --port 8000

# 2. Run the agent
python cli.py \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --agent-model google/gemma-4-E4B-it \
  --apply --run --interactive

# 3. Plan and execute
/plan Build a CLI task manager with JSON storage
/go
```

---

## Features

### Self-Healing Micro Sprints
Every action goes through: execute → fail? → system cache → Gemma reflection → Qwen retry → succeed.

### TDD Mode (`--tdd`)
Tests first, implementation follows, tests must pass before advancing. TDD Assembly curates Qwen's best attempts into production-quality tests.

### Quality Gates
Every file scored 1-10. Below 8? Gemma rewrites the prompt and Qwen tries again. Loop detection triggers TDD Assembly.

### System Memory (`~/.robustIship/`)
Learns `python` means `python3` on your machine. Remembers that `tree` isn't installed and `find` works instead. Compounds across all projects. No model calls needed after first fix.

### Interactive Mode
Review every file before it's written. See quality scores. Skip, accept, or defer to Gemma.

---

## Commands

| Command | Description |
|---------|-------------|
| `/plan <goal>` | Create structured plan |
| `/go` | Execute plan |
| `/run <goal>` | Plan + execute in one command |
| `/save` | Save state to disk |
| `/load` | Load previous state |
| `/status` | Show progress |
| `/fix` | Analyze failures |
| `/fixes` | Show saved command fixes |
| `/clear` | Clear state |
| `/exit` | Exit |

---

## Flags

### Server & Model
| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://127.0.0.1:8000/v1` | GPU server URL (Qwen) |
| `--model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Main model on GPU server |
| `--cpu-model` | `google/gemma-4-E4B-it` | Agent model (Gemma) — will be `--agent-model` in v0.27 |
| `--api-key` | `$MODEL_API_KEY` | API key for the server |
| `--max-tokens` | `4096` | Max tokens per generation |
| `--temperature` | `0.2` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus sampling |

### Agent Model (Gemma)
| Flag | Default | Description |
|------|---------|-------------|
| `--am-device` | `cpu` | Device: `cpu`, `cuda`, or `auto` |
| `--am-quant` | `fp16` | Quantization: `none`, `4bit`, `8bit`, `fp16` |
| `--am-max-memory-gib` | `None` | RAM limit for agent model |
| `--am-max-vram-gib` | `None` | VRAM limit for agent model (GPU) |

### Execution
| Flag | Default | Description |
|------|---------|-------------|
| `--apply` | `False` | Actually write files to disk |
| `--run` | `False` | Actually execute shell commands |
| `--yes` | `False` | Skip all confirmations (full auto) |
| `--max-steps` | `25` | Maximum steps per plan |
| `--root` | `.` | Workspace root directory |
| `--prompt` | `None` | One-shot prompt (non-interactive) |
| `--interactive` | `False` | Force interactive mode |

### Debug & Logging
| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | `low` | Output level: `low`, `medium`, `high` |
| `--debug` | `False` | Show raw HTTP responses and quality debug |
| `--log-output` | `False` | Write model calls to `.robustIship/logs/` |
| `--review-mode` | `failures-only` | Gemma code review: `off`, `failures-only`, `all` |

### Feature Flags
| Flag | Description |
|------|-------------|
| `--tdd` | Test-driven development mode |
| `--minimal` | Fast path only, no dual-gen or takeover |
| `--experimental` | Enable experimental multi-gen features |
| `--dual-gen` | Force-enable dual generation (Qwen + Gemma parallel) |
| `--multi-gen` | Force-enable multi generation (3 Qwen + Gemma assembly) |
| `--no-dual-gen` | Disable dual generation |
| `--no-pre-flight` | Disable pre-flight test validation |
| `--no-retry-step` | Disable RETRY_STEP command |
| `--no-session-filter` | Disable session filtering in history |

### Quality Thresholds
| Flag | Default | Description |
|------|---------|-------------|
| `--dual-gen-threshold` | `6` | Quality score ≤ this triggers dual-gen |
| `--takeover-threshold` | `2` | Quality score ≤ this triggers Gemma takeover |

### Other
| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `--clear-fixes` | Clear all saved command fixes on startup |

---

## Configuration Examples

```bash
# Basic usage
python cli.py \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --agent-model google/gemma-4-E4B-it \
  --root ./ProjectLocation/

# TDD mode with verbose output
python cli.py \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --agent-model google/gemma-4-E4B-it \
  --tdd --verbose high --apply --run --root ./ProjectLocation/

# Minimal mode (fast, no dual-gen)
python cli.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --agent-model google/gemma-4-E4B-it \
  --minimal --apply --run --yes --root ./ProjectLocation/
```

---

## How Self-Healing Works

```
Qwen generates code → Quality gate scores it 6/10
  → Gemma reflects: "Missing error handling in storage.py"
  → Qwen retries with targeted context
  → Quality gate: 9/10 → Written to workspace
```

---

## How System Memory Works

```
Step: run_command "tree"
  → Fails: "tree: command not found"
  → Gemma: "Use find . -maxdepth 3 | head -80 instead"
  → Works ✓
  → Saved to ~/.robustIship/system_fixes.json
  → Next time: instant, no model call needed
```

---

## Requirements

- **GPU**: Qwen 7B-14B (or any OpenAI-compatible endpoint) - Not Tested/Optimised for Other Models
- **CPU RAM**: ~6GB for Gemma 4-bit, ~16GB for FP16
- **Python**: 3.10+
- **OS**: Linux, macOS, Windows

---

## Limitations

- Gemma runs on CPU (20-40s per reflection)
- Dual-gen and multi-gen require GPU for Gemma (untested)
- Cross-platform testing limited to Linux/ROCm
- Experimental — test before production use

---

## Troubleshooting

**"python: not found"**
The agent will auto-detect your Python path. First run may need a retry. Subsequent runs use cached path.

**"Connection refused"**
Ensure your Qwen server is running. Check `--base-url` matches your server port.

**Agent gets stuck in a loop**
Type `/done` to force completion. Use `--max-steps 10` to limit iterations. **|OR|** 
*CTRL + C, reload the script, type `/load` to load all saved states, and then `/go` to restart!*

---

## Acknowledgements

Built with [Transformers](https://github.com/huggingface/transformers) and [Gemma](https://ai.google.dev/gemma).

---

## License

Apache 2.0 — Use it, break it, fix it, share it.