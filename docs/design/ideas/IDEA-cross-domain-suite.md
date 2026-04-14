# IDEA: Cross-Domain Benchmark Suite

**Status:** Idea
**Date:** 2026-04-13
**Related:** `docs/design/llm_benchmark_results.md` (PoC data), `docs/design/prelim/PRELIM-model-library.md`

## Motivation

The PoC benchmark revealed that cross-domain prompts — tasks requiring both
code implementation and genuine domain science knowledge — produced the largest
quality gaps between models. gemma4:e4b's biology answer (CpG islands, epigenetic
regulation, methylation-driven false ORFs) and security answer (no hallucinated
APIs, explicit user enumeration mitigation) were qualitatively different from
every other model's output, including models 3-6x its size.

The existing coding-basics suite has "cross-domain" prompts, but they're
software-engineering adjacent (auth design, SQL vs NoSQL, monolith vs micro).
These test architectural judgment, not domain knowledge transfer. The PoC showed
that a model can score well on software design prompts while completely failing
on biology or security — and vice versa.

A dedicated cross-domain suite would test the hypothesis that MoE architectures
(gemma4:e4b) route to specialized expert knowledge more effectively than dense
models at similar or larger parameter counts.

## Concept

A suite of 12-16 prompts, each requiring the model to:

1. **Implement** an algorithm or tool in Python (testable code)
2. **Answer** domain-specific reasoning questions that require knowledge
   beyond what's in the code

This two-part structure is what made the PoC prompts so effective. A model
can't pattern-match its way through — it must demonstrate both implementation
skill and domain understanding. The code portion grounds the evaluation
(correct or not), while the reasoning portion reveals depth.

### Candidate Domains

**Security (3-4 prompts)**
- Code review of vulnerable endpoints (SQL injection, XSS, CSRF, IDOR)
- Cryptographic implementation choices (hashing, token generation, key derivation)
- Threat modeling for a given architecture
- PoC showed: models hallucinate security APIs confidently (r1:14b's
  `TimedJSONWebToken`), miss timing side-channels universally

**Biology / Bioinformatics (3-4 prompts)**
- DNA/RNA sequence analysis + biological interpretation
- Protein structure prediction constraints + biochemistry reasoning
- Phylogenetic distance calculation + evolutionary biology implications
- PoC showed: gemma4:e4b was the only model to mention CpG islands,
  correctly frame ORF detection as biology-constrained, and handle
  ambiguous bases

**Mathematics / Statistics (3-4 prompts)**
- Statistical test selection and implementation for a given dataset
- Numerical methods with stability analysis (why naive approaches fail)
- Bayesian inference implementation + interpretation of posteriors
- Untested in PoC — would validate phi4:14b's claimed STEM advantage
  (80.4% MATH benchmark) against gemma4:e4b's generalist breadth

**Finance / Economics (2-3 prompts)**
- Time-value-of-money calculations with domain constraint reasoning
- Risk modeling (VaR, Monte Carlo) + interpretation of tail risk
- Market microstructure simulation + explanation of price formation
- Novel domain for most models — low contamination, tests genuine
  transfer rather than memorized patterns

### What makes a good cross-domain prompt

From the PoC, the prompts that best differentiated models shared traits:

1. **Multi-requirement** — 3-4 distinct deliverables, not just "write a function"
2. **Domain reasoning after implementation** — "now explain what this means
   biologically" forces the model to connect code output to domain knowledge
3. **Trap opportunities** — plausible-sounding but wrong answers exist (e.g.,
   "high GC → higher mutation rate" is backwards but sounds reasonable)
4. **Low contamination** — novel problem framing, not standard textbook exercises
5. **Verifiable code** — the implementation can be checked for correctness
   independently of the domain reasoning

### Domains intentionally excluded

- **Data science / ML** — every tutorial and course teaches pandas + sklearn
  pipelines. Extremely high contamination. Models score well by memorization
  rather than understanding.
- **Systems / infrastructure** — already well-covered by coding-basics
  (thread pool, event emitter, rate limiter) and routing-discovery suites.
- **Legal / medical** — hard to validate correctness without domain expertise.
  Incorrect legal or medical advice is also a liability concern even in
  benchmark context.
- **Creative writing** — subjective evaluation, doesn't fit the rubric-based
  scoring methodology.

## Strategic Value

1. **MoE vs dense architecture hypothesis** — the PoC suggests MoE models
   route to domain experts more effectively. A systematic cross-domain suite
   would validate or refute this with statistical rigor (CIs, paired tests).

2. **Routing discovery integration** — cross-domain prompts are prime
   candidates for routing. A security prompt might route to a different
   model than a biology prompt. The routing-discovery framework is already
   built to test this.

3. **Differentiation from existing benchmarks** — MMLU tests factual recall,
   HumanEval tests code generation. This suite tests the intersection:
   can a model reason in a domain AND implement correctly? Few public
   benchmarks test this combination.

4. **Practical value signal** — users choosing models for real work care about
   this exact capability: "can I trust this model on a bioinformatics task?"
   or "will it hallucinate security APIs?" These prompts answer that directly.

## System Connections

| System | Connection |
|--------|-----------|
| `suites/coding-basics.yaml` | Complementary — coding-basics tests software skills, this tests domain transfer |
| `suites/tool-use.yaml` | Could extend — cross-domain tasks with tool use (e.g., "use the calculator tool to compute bond prices") |
| `rubrics/cross-domain.yaml` | Direct — the cross-domain rubric (domain_knowledge at 30%) was designed for these prompts |
| `suites/routing-discovery.yaml` | Integration — cross-domain prompts would be strong candidates for routing strategy testing |
| Evaluator correctness hints | Critical — domain-specific `expected_answer` fields needed to anchor the judge on factual accuracy |

## Open Questions

1. **Who validates the domain answers?** The LLM judge (deepseek-r1:14b)
   may not have sufficient domain knowledge to evaluate a bioinformatics
   answer correctly. Options: very detailed `expected_answer` hints,
   domain-expert human review of the rubric, or using a frontier cloud
   model for cross-domain evaluation only.

2. **How many prompts per domain?** 3-4 gives enough signal per domain for
   within-domain comparison but may be too few for statistical rigor. The
   PoC had exactly 1 per domain — more is clearly needed, but diminishing
   returns set in quickly.

3. **Contamination control** — how do we ensure these prompts aren't in
   training data? Novel problem framing helps, but bioinformatics algorithms
   (GC content, ORF finding) are standard exercises. The key is making the
   domain reasoning questions novel, not the code.

4. **Should this be one suite or four?** One `cross-domain.yaml` with all
   domains is simpler operationally. Four separate suites (`security.yaml`,
   `biology.yaml`, etc.) allows per-domain routing discovery. Leaning toward
   one suite with domain as a tag/category for filtering.

5. **Integration with tool-use** — some cross-domain tasks naturally involve
   tools (file I/O for datasets, calculator for financial math). Should this
   suite include tool-use prompts, or keep it text-only and let tool-use.yaml
   handle that dimension separately?
