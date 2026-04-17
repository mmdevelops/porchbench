# Security Policy

## Reporting a Vulnerability

If you believe you've found a security issue in feral, please report it
privately to **mmdevelops@gmail.com** rather than opening a public issue.

Include:

- A description of the issue and its impact
- Steps to reproduce, or a minimal proof of concept
- The version or commit you tested against

Expect an initial acknowledgement within a week. Once confirmed, I'll work
on a fix and coordinate a disclosure timeline with you.

## Threat Model

feral is a benchmarking tool you run on your own machine, against models
you've chosen to install and run locally (or against API endpoints you've
authenticated to). The threat model assumes you trust the inputs you give
it: the suite YAML, the rubric YAML, the models being benchmarked, and the
inference endpoint.

### What the sandbox does (and does not) do

The `tool-use` suite executes model-generated code in a subprocess with a
temporary working directory. **This is a containment / cleanup helper, not
a security boundary.** The subprocess inherits your full environment
(including API keys), has unrestricted network access, and can read or
write any file your user account can. The only enforced limit is the
per-execution wall-clock timeout.

If you point feral at a model that emits hostile code, that code runs with
your privileges. That's the documented behavior, not a bug. Run feral in a
VM or container if you want hard isolation.

## In Scope

Bugs in feral itself that compromise a user running feral on inputs they
authored or trust:

- Remote code execution or file disclosure via crafted result/scorecard/
  rubric/suite files (parsing must remain safe — currently `yaml.safe_load`
  and Pydantic JSON validation, no `pickle`, no `eval`)
- feral writing or transmitting credentials, API keys, or other secrets
  to places the user didn't ask for (logs, result JSON, network calls)
- feral making network connections to hosts other than the configured
  inference endpoint
- Path traversal in feral's own output writers (results, scorecards,
  profiles) when the user passes trusted CLI flags

## Out of Scope

- Model output exfiltrating data, writing files outside the sandbox temp
  directory, or running arbitrary commands. The sandbox does not prevent
  this — see "What the sandbox does" above.
- Crashes, hangs, or resource exhaustion from inputs the reporter authored
  themselves (malicious YAML you wrote, malicious model you ran). If you
  control the input, it's not a vulnerability.
- Issues in dependencies (Ollama, the Anthropic SDK, pydantic, etc.).
  Report those upstream.

## Supported Versions

Only the latest release receives security fixes during the 0.x series.
