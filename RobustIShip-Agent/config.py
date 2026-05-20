"""Constants and configuration."""

VERSION = "v0.26"

BANNER = f"""
{'=' * 60}
🤖 RobustIShip {VERSION} — Modular Multi-Strategy Code Generation Agent
{'=' * 60}
"""

HELP_CMDS = "\n🎯 Commands: /plan, /go, /save, /load, /status, /fix, /clear, /fixes, /exit, /help"

ALLOWED_TOOLS = {"write_file", "edit_file", "run_command", "read_file", "grep_search"}

_REVIEWABLE_EXTS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.yaml', '.yml', '.sh', '.md'}

DANGEROUS_COMMANDS = (" rm ", " mkfs", " dd ", " shutdown", " reboot", ":(){", ">|", " chown ", " chmod 777", " wipe")