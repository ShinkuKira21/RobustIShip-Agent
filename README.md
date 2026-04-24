# RebustIShip-Agent
RebustIShip Agent is designed to connect to your local LLM (openai endpoint), and leverage Gemma-4-E4B model within CPU as a repair layer. (Experimental agent)

rebustIship/
├── .env                 # Fill this in for HF
├── agent.py              # Main agent script
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── LICENSE              # MIT License#

# 1. Start your LLM server
Option A: Ollama (easiest) (port 8000 is what the agent looks for)

```bash
ollama run qwen2.5-coder:7b
```

Option B: Run your personal server through custom scripts, or other means (using OpenAI Endpoint reinforcement).

# 2. Run the Agent

```bash
# Interactive mode (recommended)
python agent.py --apply --run --interactive

# One-shot mode
python agent.py --prompt "Create a Python script to organize my Downloads folder" --apply --run
```

# Commands (Interactive Mode)

## Command	Description
```bash
/exit	Exit the agent
/clear	Clear conversation history
/status	Show memory usage and saved fixes
/fixes	List all saved command fixes
/save	Save conversation to JSON file
/done	Force mark task as complete
/help	Show this help
```

# Configuration
```
# Basic usage with Ollama
python agent.py \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --cpu-model google/gemma-4-E4B-it \
  --interactive

# With custom CPU model
python agent.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B \
  --cpu-model google/gemma-4-E4B-it \
  --apply --run --interactive

# Auto-execute mode (skip confirmations - careful!)
python agent.py --apply --run --yes --interactive
```

# Arguments
## Argument	Default	Description
```bash
--base-url	http://127.0.0.1:8000/v1	GPU server URL
--model	huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated	Main model
--cpu-model	google/gemma-4-E4B-it	CPU validation model
--apply	False	Actually execute file writes
--run	False	Actually execute commands
--yes	False	Auto-confirm all actions
--interactive	False	Start interactive mode
--max-tokens	512	Max generation tokens
--temperature	0.2	Sampling temperature
--verbose	False	Show detailed output
--debug	False	Debug mode with extra logging
```

# How Fix Memory Works
## When a command fails, you can provide a natural language fix:
```bash
Your choice: c

Tell the CPU brain what to fix:
Original command: python -m py_compile script.py
Error: python: not found

Your instruction: use /opt/rocm-venv/bin/python instead

✅ CPU brain fixed: /opt/rocm-venv/bin/python -m py_compile script.py
💾 Remembered: use '/opt/rocm-venv/bin/python' instead of 'python'
```

# Limitations

- **CPU Model Memory**: The agent uses a 4-bit quantized version of Gemma-4-E4B requiring **~5.5-6 GB RAM**. You can swap to smaller models (Gemma-2-2B) or different quantization for less memory usage.

- **Repair Speed**: May take 5+ minutes if the main model's output is severely malformed. Repairs are highly effective for task continuation, but first-time failures can be slow.

- **Experimental**: Works on my machine™. Your mileage may vary. Test before relying on it for critical tasks.

- **File Reading**: No built-in `read_file` tool. Use system commands (`cat`, `head`, etc.) or add custom tools. Tested on Linux with ROCm.

# Troubleshooting
*"python: not found"*

**Use custom fix: use /full/path/to/python** *or activate your venv first*

*"Connection refused"*

**Ensure your LLM server is running**

**Check --base-url matches your server port**

*Agent gets stuck in loop*

**Type /done to force completion**

**Use --max-steps 10 to limit iterations**

# Contributing
Issues and PRs welcome! This is experimental, so feedback is valuable.

# Licence
Apache 2.0 - Use it, break it, fix it, share it.

# Acknowledgements
- Built with [https://huggingface.co/docs/transformers/en/index](Transformers)
- Validated by [https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/](Gemma)
