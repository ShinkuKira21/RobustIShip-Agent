"""Feature flags and quality thresholds for the orchestrator."""

from dataclasses import dataclass, field


@dataclass
class FeatureFlags:
    """Toggle features and set quality thresholds independently."""

    # Strategy toggles
    fast_path: bool = True
    quality_warn: bool = True
    dual_gen: bool = True
    multi_gen: bool = False
    gemma_takeover: bool = True
    research_path: bool = True

    # Independent features
    pre_flight: bool = True
    session_filter: bool = True
    retry_step: bool = True
    auto_pythonpath: bool = True

    # Development modes
    tdd: bool = False                 # Test-driven development: tests gate progress

    # Quality thresholds
    fast_path_threshold: int = 8       # Score ≥ this: accept silently
    warn_threshold: int = 7            # Score = this: warn, accept
    research_threshold: int = 5        # Score ≤ this: research (read more context)
    dual_gen_threshold: int = 6        # Score ≤ this: dual-gen
    multi_gen_threshold: int = 4       # Score ≤ this: multi-gen (if enabled)
    takeover_threshold: int = 2        # Score ≤ this: Gemma takeover

    # Timing
    dual_gen_timeout_qwen: int = 60
    dual_gen_timeout_gemma: int = 60
    multi_gen_timeout: int = 90

    # Debug
    log_events: bool = False          # Record events.jsonl and file snapshots

    @classmethod
    def production(cls):
        """Default production settings."""
        return cls()

    @classmethod
    def minimal(cls):
        """Minimal overhead — fast path only."""
        return cls(
            dual_gen=False,
            multi_gen=False,
            gemma_takeover=False,
            pre_flight=False,
            retry_step=False,
        )

    @classmethod
    def experimental(cls):
        """All experimental features enabled."""
        return cls(multi_gen=True)

    @classmethod
    def from_args(cls, args):
        """Build flags from CLI arguments."""
        flags = cls()
        if getattr(args, "minimal", False):
            return cls.minimal()
        if getattr(args, "experimental", False):
            flags = cls.experimental()
        if getattr(args, "no_dual_gen", False):
            flags.dual_gen = False
        if getattr(args, "no_pre_flight", False):
            flags.pre_flight = False
        if getattr(args, "no_retry_step", False):
            flags.retry_step = False
        if getattr(args, "no_session_filter", False):
            flags.session_filter = False
        if getattr(args, "dual_gen_threshold", None) is not None:
            flags.dual_gen_threshold = args.dual_gen_threshold
        if getattr(args, "takeover_threshold", None) is not None:
            flags.takeover_threshold = args.takeover_threshold
        if getattr(args, "tdd", False):
            flags.tdd = True
        if getattr(args, "log_events", False):
            flags.log_events = True
        
        return flags