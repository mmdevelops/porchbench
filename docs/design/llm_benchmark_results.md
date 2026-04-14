# Local LLM Benchmark Results
**Hardware:** AMD Radeon RX 9070 XT (16GB VRAM) · Windows 11 · Ollama w/ ROCm HIP SDK workaround  
**Date:** April 2026

---

## Methodology

Three prompts of increasing complexity were used to assess each model across different dimensions:

### Prompt 1 — Pure Coding Task
> Write a Python function `parse_log_entries` that takes a list of raw log strings in the format `"[LEVEL] TIMESTAMP: MESSAGE"` and returns a dictionary grouping messages by level, with each level mapping to a list of `(timestamp, message)` tuples sorted by timestamp ascending. Handle malformed lines under an `"UNPARSEABLE"` key. Include a docstring with an example.

**Tests:** String parsing, data structures, sorting, error handling, documentation quality.

### Prompt 2 — Reasoning Task
> A distributed system has a bug where requests are sometimes processed twice. Given a sequence of log entries, diagnose what's going wrong and write a production-ready Python fix using Redis.

**Tests:** Log analysis, root cause diagnosis, distributed systems knowledge, Redis locking primitives.

### Prompt 3 — Cross-Domain Task (Coding + Biology)
> Write a Python function that analyzes a DNA string: GC content, reverse complement, all ORFs, and most frequent codon. Then answer: a researcher reports ~65% GC content — what are two biological implications and how does it affect ORF detection?

**Tests:** Bioinformatics algorithms, biochemistry domain knowledge, multi-requirement spec adherence.

### Prompt 4 — Cross-Domain Task (Coding + Security)
> Review a Flask password reset endpoint written by a junior developer. Identify every security vulnerability, explain why each is dangerous, and rewrite it to be production-safe.

**Tests:** Web security (OWASP), cryptography, credential management, secure coding practices.

---

## Results by Model

### qwen2:3b
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Mostly correct |
| **Reasoning task** | — not tested |
| **Cross-domain** | — not tested |
| **Speed** | Fast |
| **Reliability** | ✅ Consistent |

**Notes:**
- Hardcoded only `ERROR` and `INFO` levels — `WARN`/`DEBUG` would silently misroute to `UNPARSEABLE`
- Sorts on every loop iteration rather than once at the end (wasteful but not wrong)
- Shared `[` stripping bug with qwen2:7b — `level` ends up as `"[ERROR"` not `"ERROR"`
- Despite the bugs, produces runnable code that broadly solves the problem
- **Verdict:** Surprisingly capable for 3B. Good for simple boilerplate on constrained hardware.

---

### qwen2:7b
| Attribute | Result |
|---|---|
| **Coding task** | ❌ Broken |
| **Reasoning task** | — not tested |
| **Cross-domain** | — not tested |
| **Speed** | Fast |
| **Reliability** | ❌ Low |

**Notes:**
- Reaches for `datetime`, `defaultdict`, and dynamic dict — more sophisticated than the 3B
- But splits on spaces expecting 3 parts, leaving a trailing `:` on the timestamp and `[` on the level — fundamental parsing failure
- Classic "capability overhang" — bigger model introduces more subtle bugs by restructuring the problem
- **Verdict:** Worst performer in the benchmark. Avoid for anything requiring correctness.

---

### deepseek-r1:8b
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Good |
| **Reasoning task** | — not tested |
| **Cross-domain** | — not tested |
| **Speed** | ⚠️ Slow (thinking phase) |
| **Reliability** | ✅ Moderate |

**Notes:**
- Verbose "Approach / Solution / Explanation" structure is a DeepSeek-R1 signature
- Correctly strips `[` and `]` with `parts[0][1:-1]` — fixes the bug both Qwen2 models had
- But timestamp validation logic is convoluted and fragile (`startswith('20')` heuristic)
- Thinking overhead adds significant latency even for simple well-defined tasks
- **Verdict:** Solid but the thinking phase is overhead for simple tasks. Better reserved for ambiguous problems.

---

### deepseek-r1:14b
| Attribute | Result |
|---|---|
| **Coding task** | ⚠️ Wrong format |
| **Reasoning task** | ⚠️ Race condition + broken logging |
| **Cross-domain (bio)** | ❌ Incomplete — ignored 3 of 4 requirements |
| **Cross-domain (security)** | ⚠️ Found most issues but hallucinated API |
| **Speed** | ❌ Slowest (53–130 seconds) |
| **Reliability** | ❌ Low |

**Coding task issues:**
- Regex required milliseconds + Z suffix (`\.\d+Z`) — rejects every example in the prompt
- Renamed function to `parse_logs` instead of specified `parse_log_entries`
- Effectively solved a different problem than the one given

**Reasoning task issues:**
- `setnx` + `expire` as two separate commands — non-atomic, can deadlock if process crashes between them
- `self.log` defined as a method but called as `self.log.info()` — throws `AttributeError` at runtime
- Didn't diagnose the log timeline before jumping to code

**Bioinformatics task issues:**
- Only implemented ORF finding, ignored GC content, reverse complement, and most frequent codon
- ORF inner loop steps by 1 instead of 3 — fundamental in-frame error
- No reading frames 1 or 2, no reverse complement strand search
- Ignored the biology questions entirely
- 130 seconds for the least complete answer in the benchmark

**Security task:**
- Found the most vulnerabilities of any model (8 distinct issues)
- Suggested token hashing with `itsdangerous` — conceptually correct
- But `TimedJSONWebToken` does not exist in the library — confident hallucination that would fail on import
- Most knowledgeable on paper, but least reliable in practice

**Verdict:** Recurring pattern — finds more things, explains them well, then hallucinates a critical detail under complexity. Dangerous in a production context because failures are confident. The thinking phase fragments attention on multi-part problems rather than helping.

---

### gemma4:26b
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Excellent |
| **Reasoning task** | — not tested |
| **Cross-domain** | — not tested |
| **Speed** | Medium |
| **Reliability** | ✅ High |

**Notes:**
- `defaultdict(list)` — most Pythonic dict handling of any model
- Named regex capture groups (`?P<level>`, `?P<timestamp>`) — more readable than indexed groups
- Skips empty lines — small but thoughtful edge case
- `return dict(parsed_data)` — good practice converting defaultdict back to plain dict
- **One notable flaw:** Typo in docstring — `'20lar-01-15T08:32:11'` instead of `'2024-01-15T08:32:11'`
- Paradoxically scores slightly below gemma4:e4b despite being ~6x larger
- **Verdict:** Excellent but outperformed by the smaller MoE sibling. Good general-purpose model.

---

### qwen2.5-coder:14b (Q4)
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Excellent |
| **Reasoning task** | — not tested |
| **Cross-domain** | — not tested |
| **Speed** | ✅ Very fast (fastest of all models tested) |
| **Reliability** | ✅ High |

**Notes:**
- Strict regex: `[A-Z]+` for level, `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}` for timestamp
- Only model to include a proper `assert`-based unit test that runs and passes
- No pre-populated keys, no empty UNPARSEABLE in output
- `import re` inside the function — unconventional but harmless
- No type hints
- **Verdict:** Best speed/correctness ratio for pure coding. No fluff, straight to correct code.

---

### qwen2.5-coder:14b (Q8)
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Best pure coding |
| **Reasoning task** | ⚠️ Shallow — missed root cause diagnosis |
| **Cross-domain (bio)** | ⚠️ Correct code, factual biology error |
| **Cross-domain (security)** | ✅ Most deployable rewrite |
| **Speed** | ⚠️ Slower than Q4 (50–86 seconds) |
| **Reliability** | ✅ High |

**Coding task — best solution:**
- `logs.setdefault(level, []).append(...)` — most elegant dict handling in the benchmark
- `datetime.strptime` for proper timezone-aware sorting
- Whole function body in ~8 lines
- No input pre-population, no unnecessary keys

**Reasoning task issues:**
- Used mutex lock (releases after completion) rather than idempotency key (persists after completion)
- A late retry after lock release would reprocess — wrong tool for the problem
- Never diagnosed the log timeline

**Bioinformatics task:**
- All three reading frames + reverse complement ORF search ✅
- `Counter` for most frequent codon ✅
- Biology answer had one factual error: claimed high GC → higher mutation rate (backwards — GC-rich regions are more stable and typically have lower mutation rates)
- Missed secondary structure / hairpin loops and CpG islands

**Security task — best rewrite:**
- `environ.get("SMTP_PASS")` with no fallback — only model to get credential handling exactly right
- `smtp.starttls()` actually implemented (not just noted)
- `MIMEText` with proper message structure
- `app.logger.error` instead of print

**Verdict:** Go-to model for pure code generation. Fast, idiomatic, no hallucinations. But the coder finetune advantage disappears on tasks requiring domain reasoning before coding.

---

### gemma4:e4b ⭐
| Attribute | Result |
|---|---|
| **Coding task** | ✅ Excellent |
| **Reasoning task** | ✅ Best overall |
| **Cross-domain (bio)** | ✅ Best overall |
| **Cross-domain (security)** | ✅ Best overall |
| **Speed** | ✅ Fastest overall (20–45 seconds) |
| **Reliability** | ✅ Highest |

**Coding task:**
- `str.maketrans` / `.translate` for reverse complement — most elegant approach
- `dna_sequence.upper().replace('N', '')` — handles ambiguous bases, only model to do this
- `orfs = set()` — automatically deduplicates overlapping ORFs
- Inner loop correctly steps by 3 (in-frame)
- Type hints, docstring, empty sequence guard, case-insensitive input

**Reasoning task (distributed systems):**
- Only model to fully diagnose the logs before writing code
- Named the pattern correctly: "At-Least-Once Delivery" / "Timeout Ambiguity"
- Called out the specific timestamps: retry fired at 849ms before first attempt completed at ~1.1s
- `setex` — single atomic set+TTL, avoiding the race condition that broke r1:14b
- Namespaced Redis keys (`processed:request:{request_id}`)
- Marks success *after* business logic, not before
- Does NOT mark as processed on failure — explicitly implemented
- Included simulation scenarios demonstrating the fix working
- Summary of best practices that acknowledged the remaining TOCTOU limitation

**Bioinformatics task:**
- `str.maketrans` complement — Pythonic and correct
- Handles ambiguous 'N' bases
- Set-based ORF deduplication
- **Biology answer was the standout of the entire benchmark:**
  - Thermal stability + Tm ✅
  - CpG islands and epigenetic regulation — no other model mentioned this
  - Hairpin loops / secondary structure impeding RNA polymerase ✅
  - Methylation → gene silencing → computationally predicted but non-functional ORF
  - Correctly reframed: "the algorithm is fine, the biology is the constraint"

**Security task:**
- SQL injection, weak token, hardcoded credentials, rate limiting, user enumeration all caught
- Only model to explicitly note user enumeration by name and implement constant response
- No hallucinated APIs
- Coherent single rewrite (unlike r1:14b's fragmented snippets)
- Missed: SMTP TLS (noted but not implemented), token expiry, timing side-channel

**Verdict:** Best overall performer by a significant margin. Fastest, most reliable, strongest cross-domain reasoning. The MoE architecture appears to route effectively to specialized knowledge without the overhead of a large dense model.

---

## Summary Leaderboard

| Model | Params | Coding | Reasoning | Cross-domain | Speed | Reliability | **Overall** |
|---|---|---|---|---|---|---|---|
| **gemma4:e4b** | ~4B MoE | ✅ Excellent | ✅ Best | ✅ Best | ✅ Fastest | ✅ Highest | ⭐ **#1** |
| qwen2.5-coder:14b Q8 | 14B dense | ✅ Best | ⚠️ Shallow | ⚠️ Errors | ⚠️ Slow | ✅ High | **#2** |
| qwen2.5-coder:14b Q4 | 14B dense | ✅ Excellent | — | — | ✅ Fastest | ✅ High | **#3** |
| gemma4:26b | 26B dense | ✅ Excellent | — | — | Medium | ✅ High | **#4** |
| deepseek-r1:8b | 8B dense | ✅ Good | — | — | ⚠️ Slow | ✅ Moderate | **#5** |
| qwen2:3b | 3B dense | ⚠️ Basic | — | — | Fast | ✅ Consistent | **#6** |
| deepseek-r1:14b | 14B dense | ⚠️ Wrong spec | ❌ Bugs | ❌ Incomplete | ❌ Slowest | ❌ Low | **#7** |
| qwen2:7b | 7B dense | ❌ Broken | — | — | Fast | ❌ Low | **#8** |

---

## Key Takeaways

### 1. Parameter count is a poor predictor of quality
The 7B dense model was the worst performer. The nominally 4B MoE beat everything including the 14B and 26B dense models on reasoning and cross-domain tasks.

### 2. MoE architecture punches well above its weight
gemma4:e4b's expert routing appears to activate specialized knowledge per token without the VRAM and inference overhead of a large dense model. It fits comfortably in 16GB and outperforms models several times its size.

### 3. Coder finetuning has a narrow but real advantage
`qwen2.5-coder:14b Q8` produces the tightest, most idiomatic code on pure coding tasks. But the advantage disappears the moment the task requires understanding a problem before writing code.

### 4. Reasoning models need the right problems
DeepSeek-R1's thinking phase is real compute that needs genuinely ambiguous or hard problems to justify its latency cost. On well-specified tasks it's pure overhead — and on multi-part problems it appears to fragment attention rather than improve it.

### 5. Confident failures are worse than simple failures
r1:14b's hallucinated `TimedJSONWebToken` and broken `self.log` are more dangerous than qwen2:3b's hardcoded levels — the former passes code review, the latter is obviously incomplete.

### 6. No model caught the timing side-channel
Across both the security task and related reasoning tasks, no model identified response-time-based information leakage as a vulnerability. This appears to be a ceiling for security depth at this model size tier.

---

## Recommended Routing Architecture

```
User prompt
     │
     ▼
 Classifier (gemma4:e4b or heuristic)
     │
     ├──► Simple boilerplate / autocomplete
     │    └──► qwen2.5-coder:14b Q4  (speed priority)
     │
     ├──► Standard coding tasks
     │    └──► qwen2.5-coder:14b Q8  (quality priority)
     │
     └──► Reasoning / cross-domain / ambiguous
          └──► gemma4:e4b  (best all-around)
```

For most workloads, **gemma4:e4b as the default** with **qwen2.5-coder:14b Q8 for high-throughput pure coding** is the optimal configuration on 16GB VRAM hardware.

---

---

### qwen3.5:9b ⏳ (Pending GPU Support)
| Attribute | Result |
|---|---|
| **Coding task (thinking on)** | ✅ Good — 6m 50s |
| **Coding task (thinking off)** | ✅ Excellent — 96s |
| **Reasoning task** | — not tested (CPU only) |
| **Cross-domain** | — not tested (CPU only) |
| **Speed** | ❌ CPU-only on RDNA 4, unusable |
| **Reliability** | — insufficient data |

**Status:** GPU acceleration not working on RDNA 4 (gfx1201) with Ollama 0.20.5. Model runs 100% CPU despite ROCm fix that works for other models.

**Log evidence:**
```
offloading 0 repeating layers to GPU
offloaded 0/33 layers to GPU
```
No `gfx1201` or `ROCm` mention during initialization. 43-minute gap between GPU discovery and first load request suggests initialization hang. `OLLAMA_NEW_ENGINE=false` in current config — qwen3.5 architecture (`requires 0.17.1`) likely needs new engine path for proper ROCm initialization.

**Workarounds attempted:**
- `OLLAMA_NEW_ENGINE=1` — no effect
- `--num-gpu 99` — no effect  
- `/nothink` Modelfile with `SYSTEM` directive — thinking disable works via `/set nothink` in CLI session only
- `PARAMETER thinking false` in Modelfile — unsupported parameter in Ollama 0.20.5

**Code quality (CPU runs):**
- Optional milliseconds in regex `(?:\.\d+)?` — handles both `08:32:11` and `08:32:11.001` formats, best timestamp handling in benchmark
- `defaultdict(list)` + `return dict()` ✅
- UNPARSEABLE stores `(None, entry)` tuples for structural consistency ✅
- Comprehensive docstring ✅
- No hardcoded levels, sorts once ✅

**Thinking mode notes:**
- Enabled by default, adds ~6x latency
- `/set nothink` in Ollama CLI session disables it
- `think: false` works via Python ollama library or direct API call
- Modelfile `PARAMETER thinking` is not a supported directive in Ollama 0.20.5

**Verdict:** Output quality suggests this could challenge gemma4:e4b once GPU support lands. Revisit when Ollama adds proper RDNA 4 support for qwen3.5 architecture. Expected timeline: 2-4 weeks based on community issue activity.

---

## Hardware Notes

- **GPU:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201, 16GB GDDR6)
- **OS:** Windows 11
- **Ollama GPU fix:** ROCm HIP SDK 7.x — copy `gfx1201` rocblas libraries from `C:\Program Files\AMD\ROCm\7.x\bin\rocblas\library` into `C:\Users\<USER>\AppData\Local\Programs\Ollama\lib\ollama\rocm\rocblas\library` **(APPLIED — GPU acceleration working)**
- **Confirmed working log entry:** `library=ROCm compute=gfx1201 name=ROCm0 description="AMD Radeon RX 9070 XT" total="15.9 GiB"`
- Native Ollama Windows support for RDNA 4 is still pending (issue #10430) as of April 2026
