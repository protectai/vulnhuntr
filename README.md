<div align="center">

  <img width="250" src="https://github.com/user-attachments/assets/d1153ab4-df29-4955-ad49-1be7fad18bb3" alt="VulnHuntr Logo">

# VulnHuntr v2.0
### Next-Generation AI-Driven Static Vulnerability & Taint-Chain Auditor

**World's first autonomous AI-discovered 0day vulnerabilities — upgraded with multi-provider LLM engines, modern Python support, SARIF export, and expanded vulnerability taxonomy.**

</div>

---

## What's New in v2.0?

VulnHuntr v2.0 is an enhanced continuation of Protect AI's original VulnHuntr, addressing previous limitations and adding enterprise-grade features:

- 🚀 **Modern Python Support (3.10 to 3.14+)**: Removed the legacy Jedi 0.18 pin. Now runs cleanly on modern Python runtimes.
- 🧠 **Multi-Provider LLM Engine**:
  - **Google Gemini** (`gemini-2.5-pro`, `gemini-2.5-flash`) via `google-genai` with huge context windows and rapid reasoning.
  - **Anthropic Claude** (`claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`).
  - **OpenAI** (`gpt-4o`, `o3-mini`, `gpt-4.5`).
  - **Local Models** via Ollama / OpenAI-compatible local endpoints.
- ⚡ **Hybrid AST & Symbol Resolution**: Fallback AST parser using standard Python `ast` alongside modern Jedi 0.20+, ensuring call chain traversal never crashes or drops symbols.
- 🛡️ **Expanded Vulnerability Taxonomy (11 Classes)**:
  - Local File Inclusion (LFI)
  - Remote Code Execution (RCE)
  - Cross-Site Scripting (XSS)
  - Arbitrary File Overwrite (AFO)
  - SQL Injection (SQLI)
  - Server-Side Request Forgery (SSRF)
  - Insecure Direct Object Reference (IDOR)
  - **Server-Side Template Injection (SSTI)** *(New)*
  - **Insecure Deserialization** (`pickle`, `yaml`, `marshal`) *(New)*
  - **Authentication & Authorization Bypasses** *(New)*
  - **Path Traversal & Zip Slip** *(New)*
- 📋 **Defensive Remediation**: Every finding now includes concrete mitigation advice and secure coding recommendations.
- 📊 **Enterprise Output Formats**:
  - **SARIF 2.1.0**: Standard format for direct ingestion into GitHub Actions Code Scanning and IDEs.
  - **Interactive HTML Dashboard**: Self-contained dark-mode report with confidence badges, code traces, and remediation steps.
  - **JSON & Rich Terminal**: Formatted tables and statistics.

---

## Installation

### Prerequisites
- Python >= 3.10 (tested through Python 3.14)

### Install from Source
```bash
git clone https://github.com/<your-repo>/vulnhuntr.git
cd vulnhuntr
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

---

## Configuration & Environment Variables

Export the API key for your preferred provider before running:

```bash
# Google Gemini (Default)
export GEMINI_API_KEY="your-gemini-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Ollama / Local (Optional)
export OLLAMA_BASE_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_MODEL="llama3.2"
```

---

## Usage

```
usage: vulnhuntr [-h] -r ROOT [-a ANALYZE] [-l {gemini,claude,gpt,ollama}]
                 [-m MODEL] [-o OUTPUT] [-f {terminal,json,sarif,html}]
                 [--max-iterations MAX_ITERATIONS] [-v]

options:
  -h, --help            Show this help message and exit
  -r, --root ROOT       Path to the root directory of the project
  -a, --analyze ANALYZE
                        Specific path or file within the project to analyze
  -l, --llm {gemini,claude,gpt,ollama}
                        LLM provider to use (default: claude)
  -m, --model MODEL     Custom model identifier (e.g. gemini-2.5-pro, claude-3-7-sonnet-latest)
  -o, --output OUTPUT   Output path to save report file
  -f, --format {terminal,json,sarif,html}
                        Output format (default: terminal)
  --max-iterations MAX_ITERATIONS
                        Maximum context retrieval iterations (default: 7)
  -v, --verbosity       Increase output verbosity (-v for INFO, -vv for DEBUG)
```

### Examples

#### 1. Scan with Google Gemini and Generate an HTML Report
```bash
vulnhuntr -r /path/to/target/repo/ -l gemini -m gemini-2.5-pro -f html -o report.html
```

#### 2. Scan for CI/CD with SARIF Export (GitHub Code Scanning)
```bash
vulnhuntr -r . -l claude -f sarif -o results.sarif
```

#### 3. Analyze a Specific Suspicious API File
```bash
vulnhuntr -r /path/to/target/repo/ -a server/api/routes.py -l gpt
```

---

## GitHub Actions CI/CD Integration

You can easily run VulnHuntr in your GitHub Actions pipeline:

```yaml
name: VulnHuntr Security Audit

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install VulnHuntr
        run: |
          pip install git+https://github.com/<your-repo>/vulnhuntr.git

      - name: Run VulnHuntr Scan
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          vulnhuntr -r . -l gemini -f sarif -o results.sarif

      - name: Upload SARIF Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: results.sarif
```

---

## Logic Flow

1. **Endpoint Discovery**: Automatically scans project files for remote user-input decorators and web framework routes (FastAPI, Flask, Django, Tornado, aiohttp, Starlette, etc.).
2. **Architecture Ingestion**: Summarizes project README to contextualize remote attack surfaces.
3. **Initial Static Taint Audit**: Inspects potential user-controlled inputs to dangerous function sinks.
4. **Iterative Call-Chain Traversal**:
   - The LLM requests missing definitions (classes, helper methods, middleware).
   - The **Hybrid AST + Jedi Engine** extracts the exact source code across the repository.
   - Repeats recursively until the complete user-input to server-processing chain is evaluated.
5. **Validation & Proof of Concept**: Assesses whether sanitizers or filters can be bypassed, assigns a confidence score (0–10), and produces an exploit PoC + defensive mitigation plan.
6. **Reporting**: Formats findings to Rich terminal, SARIF 2.1.0, or standalone HTML.

---

## Attribution & License

- **Original Project**: Created by Dan McInerney and Marcello Salvati at Protect AI ([protectai/vulnhuntr](https://github.com/protectai/vulnhuntr)).
- **Version 2.0 Enhancements**: Maintained by the community.
- **License**: Released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
