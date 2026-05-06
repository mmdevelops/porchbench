# examples/

Canonical artifact samples for the public schemas re-exported from
`porchbench` (`RunResult`, `Scorecard`, `RoutingAnalysis`, `SystemProfile`).
Use them to validate code that consumes porchbench output before you have
your own runs to test against.

## Loading

```python
import json
from pathlib import Path

from porchbench import RoutingAnalysis, RunResult, Scorecard, SystemProfile

examples = Path(__file__).parent / "examples"

profile = SystemProfile.model_validate_json(
    (examples / "system-profile.json").read_text(encoding="utf-8")
)
run = RunResult.model_validate_json(
    (examples / "run-result_coding-basics_ministral-3-8b.json").read_text(encoding="utf-8")
)
scorecard = Scorecard.model_validate_json(
    (examples / "scorecard_coding-basics_ministral-3-8b.json").read_text(encoding="utf-8")
)
routing = RoutingAnalysis.model_validate_json(
    (examples / "routing-analysis_tool-use_ministral-vs-granite.json").read_text(encoding="utf-8")
)
```

The CLI tools (`porchbench compare`, `porchbench evaluate`,
`porchbench analyze-routes`) all accept these files directly.

## Files

| File | Schema | Produced by |
|------|--------|-------------|
| `system-profile.json` | `SystemProfile` | `porchbench profile -m ministral-3:8b -m granite4.1:8b` |
| `run-result_coding-basics_ministral-3-8b.json` | `RunResult` | `porchbench run -s coding-basics -m ministral-3:8b -m granite4.1:8b --evaluate --eval-model devstral-small-2:24b` |
| `run-result_coding-basics_granite4.1-8b.json` | `RunResult` | (same command, granite output) |
| `scorecard_coding-basics_ministral-3-8b.json` | `Scorecard` | `--evaluate` post-phase from the run above |
| `scorecard_coding-basics_granite4.1-8b.json` | `Scorecard` | (same, granite scorecard) |
| `run-result_tool-use-discovery_ministral-3-8b.json` | `RunResult` | `porchbench run -s tool-use --strategies -m ministral-3:8b -m granite4.1:8b` |
| `run-result_tool-use-discovery_granite4.1-8b.json` | `RunResult` | (same command, granite output) |
| `routing-analysis_tool-use_ministral-vs-granite.json` | `RoutingAnalysis` | `porchbench analyze-routes` over the two `tool-use-discovery` results above |

The two `tool-use-discovery` runs exercise schema fields the `coding-basics`
runs don't reach: `PromptResult.strategy` (per-cell strategy tag),
`PromptResult.tool_use_metrics` (tool-call counts, transcript metadata),
`PromptResult.validation_passed` (sandbox validator outcome). They cover the
full 19-prompt tool-use suite × 4 strategies × 2 models = 152 cells per file.

All 2026-05-06 runs against Ollama on an RX 9070 XT (16 GB VRAM, ROCm
gfx1201). Ministral and granite picked as a paired comparison at the same
parameter scale (8B); devstral-small-2:24b picked as judge to keep the
evaluator off the model families under test (no self-preference bias).

The tool-use routing analysis spans the full 19-prompt tool-use suite
(`t1-*` mechanics, `t2-*` multi-step, `t3-*` error recovery) across
4 prompting strategies (`cot`, `direct`, `structured`, `universal`) per
model — 152 inference cells in total.

## Schema drift

`tests/test_examples_roundtrip.py` round-trips every file here through
its schema during CI. If a future schema change makes any example
uninstantiable, that test fails before the wheel ships.
