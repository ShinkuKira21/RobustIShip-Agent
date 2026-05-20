"""Quality-based strategies — fast path through Gemma takeover."""

from orchestrator.strategies.fast_path import handle as fast_path
from orchestrator.strategies.warn_path import handle as warn_path
from orchestrator.strategies.dual_gen import handle as dual_gen
from orchestrator.strategies.multi_gen import handle as multi_gen
from orchestrator.strategies.gemma_takeover import handle as gemma_takeover