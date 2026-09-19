import json
import html
from typing import List, Dict, Any
from pathlib import Path

CWE_MAPPING: Dict[str, Dict[str, str]] = {
    "RCE": {"cwe": "CWE-78", "name": "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')"},
    "SQLI": {"cwe": "CWE-89", "name": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')"},
    "XSS": {"cwe": "CWE-79", "name": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"},
    "LFI": {"cwe": "CWE-22", "name": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal / Local File Inclusion')"},
    "AFO": {"cwe": "CWE-73", "name": "External Control of File Name or Path ('Arbitrary File Overwrite')"},
    "SSRF": {"cwe": "CWE-918", "name": "Server-Side Request Forgery (SSRF)"},
    "IDOR": {"cwe": "CWE-639", "name": "Authorization Bypass Through User-Controlled Key ('Insecure Direct Object Reference')"},
    "SSTI": {"cwe": "CWE-1336", "name": "Improper Neutralization of Special Elements in Template Engine"},
    "DESERIALIZATION": {"cwe": "CWE-502", "name": "Deserialization of Untrusted Data"},
    "AUTH_BYPASS": {"cwe": "CWE-287", "name": "Improper Authentication / Authorization Bypass"},
    "PATH_TRAVERSAL": {"cwe": "CWE-22", "name": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')"},
}


def generate_sarif(findings: List[Dict[str, Any]], repo_root: str = ".") -> Dict[str, Any]:
    """Generates standard OASIS SARIF v2.1.0 output compatible with GitHub Code Scanning."""
    rules = []
    results = []
    seen_rules = set()

    for item in findings:
        report = item.get("report", {})
        file_path = str(item.get("file_path", "unknown"))
        rel_path = file_path
        try:
            rel_path = str(Path(file_path).relative_to(Path(repo_root))).replace("\\", "/")
        except Exception:
            pass

        vuln_types = report.get("vulnerability_types", [])
        confidence = report.get("confidence_score", 0)
        level = "error" if confidence >= 8 else "warning" if confidence >= 5 else "note"

        for vtype in vuln_types:
            rule_id = f"VULN-{vtype}"
            cwe_info = CWE_MAPPING.get(vtype, {"cwe": "CWE-20", "name": "Improper Input Validation"})

            if rule_id not in seen_rules:
                seen_rules.add(rule_id)
                rules.append({
                    "id": rule_id,
                    "name": vtype,
                    "shortDescription": {"text": f"Potential {vtype}: {cwe_info['name']}"},
                    "fullDescription": {"text": f"Remotely exploitable {vtype} detected through taint chain analysis."},
                    "help": {
                        "text": f"Vulnerability: {vtype}\nCWE: {cwe_info['cwe']}\nRemediation:\n{report.get('remediation', 'Review input validation and sanitization.')}",
                        "markdown": f"### Remediation\n\n{report.get('remediation', 'Review input validation and sanitization.')}\n\n**CWE**: [{cwe_info['cwe']}](https://cwe.mitre.org/data/definitions/{cwe_info['cwe'].split('-')[-1]}.html)"
                    },
                    "properties": {
                        "tags": ["security", "ai-sast", vtype.lower()],
                        "precision": "high" if confidence >= 8 else "medium",
                        "problem.severity": "error" if confidence >= 8 else "warning"
                    }
                })

            result_entry = {
                "ruleId": rule_id,
                "level": level,
                "message": {
                    "text": f"[{vtype} Confidence: {confidence}/10] {report.get('analysis', 'Vulnerability detected')}"
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": rel_path,
                            "uriBaseId": "%SRCROOT%"
                        },
                        "region": {
                            "startLine": 1,
                            "startColumn": 1
                        }
                    }
                }],
                "properties": {
                    "confidence_score": confidence,
                    "proof_of_concept": report.get("poc", ""),
                    "remediation": report.get("remediation", ""),
                    "scratchpad": report.get("scratchpad", "")
                }
            }
            results.append(result_entry)

    sarif_doc = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "VulnHuntr",
                    "version": "0.2.0",
                    "informationUri": "https://github.com/protectai/vulnhuntr",
                    "rules": rules
                }
            },
            "results": results
        }]
    }
    return sarif_doc


def generate_html_report(findings: List[Dict[str, Any]], summary_stats: Dict[str, Any]) -> str:
    """Generates a standalone, beautiful HTML dashboard report."""
    total_findings = len(findings)
    high_conf = sum(1 for f in findings if f.get("report", {}).get("confidence_score", 0) >= 8)
    med_conf = sum(1 for f in findings if 5 <= f.get("report", {}).get("confidence_score", 0) < 8)

    cards_html = []
    for idx, item in enumerate(findings, 1):
        report = item.get("report", {})
        file_path = html.escape(str(item.get("file_path", "Unknown File")))
        confidence = report.get("confidence_score", 0)
        conf_badge = "badge-danger" if confidence >= 8 else "badge-warning" if confidence >= 5 else "badge-info"
        vtypes = " ".join([f"<span class='badge'>{html.escape(t)}</span>" for t in report.get("vulnerability_types", [])])
        analysis = html.escape(report.get("analysis", "No analysis provided"))
        scratchpad = html.escape(report.get("scratchpad", "No reasoning provided"))
        poc = html.escape(report.get("poc", "None"))
        remediation = html.escape(report.get("remediation", "Apply strict input validation and least-privilege operations."))

        card = f"""
        <div class="finding-card">
            <div class="card-header">
                <div>
                    <span class="file-title">{file_path}</span>
                    <div class="tag-row">{vtypes}</div>
                </div>
                <div class="score-box {conf_badge}">
                    Confidence: {confidence}/10
                </div>
            </div>
            <div class="card-body">
                <p><strong>Analysis:</strong> {analysis}</p>
                <details class="collapsible">
                    <summary>Step-by-Step Reasoning Trace</summary>
                    <pre class="code-block">{scratchpad}</pre>
                </details>
                <div class="section-title">Proof of Concept / Exploit Vector</div>
                <pre class="code-block poc-block">{poc}</pre>
                <div class="section-title">Defensive Mitigation & Remediation</div>
                <div class="remediation-box">{remediation}</div>
            </div>
        </div>
        """
        cards_html.append(card)

    findings_content = "\n".join(cards_html) if cards_html else "<div class='empty-state'>No high-risk vulnerabilities identified in scanned files.</div>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VulnHuntr v2.0 Security Audit Report</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-bright: #f0f6fc;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --danger: #f85149;
            --warning: #d29922;
            --success: #3fb950;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            color: var(--text-bright);
            margin: 0 0 8px 0;
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: 700;
            color: var(--text-bright);
        }}
        .stat-danger {{ color: var(--danger); }}
        .stat-warning {{ color: var(--warning); }}
        .stat-success {{ color: var(--success); }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }}
        .finding-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.02);
        }}
        .file-title {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
            font-size: 15px;
            font-weight: 600;
            color: var(--accent);
        }}
        .tag-row {{
            margin-top: 6px;
            display: flex;
            gap: 6px;
        }}
        .badge {{
            background: #21262d;
            border: 1px solid var(--border-color);
            color: var(--text-bright);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .score-box {{
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
        }}
        .badge-danger {{ background: rgba(248, 81, 73, 0.15); color: var(--danger); border: 1px solid var(--danger); }}
        .badge-warning {{ background: rgba(210, 153, 34, 0.15); color: var(--warning); border: 1px solid var(--warning); }}
        .badge-info {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid var(--accent); }}
        .card-body {{
            padding: 20px;
        }}
        .section-title {{
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin: 16px 0 8px 0;
        }}
        .code-block {{
            background: #090d13;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 13px;
            overflow-x: auto;
            color: #e6edf3;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .poc-block {{
            border-left: 3px solid var(--danger);
        }}
        .remediation-box {{
            background: rgba(63, 185, 80, 0.08);
            border: 1px solid rgba(63, 185, 80, 0.3);
            border-left: 3px solid var(--success);
            padding: 12px;
            border-radius: 6px;
            color: #d2f8d2;
            font-size: 14px;
        }}
        .collapsible summary {{
            cursor: pointer;
            color: var(--accent);
            font-size: 13px;
            user-select: none;
            margin: 12px 0;
        }}
        .empty-state {{
            text-align: center;
            padding: 40px;
            background: var(--card-bg);
            border-radius: 8px;
            border: 1px dashed var(--border-color);
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ VulnHuntr v2.0 Security Report</h1>
            <div class="subtitle">AI-Driven Static Code Analysis & Remote Taint Chain Audit</div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_findings}</div>
                <div class="stat-label">Total Vulnerabilities</div>
            </div>
            <div class="stat-card">
                <div class="stat-number stat-danger">{high_conf}</div>
                <div class="stat-label">High Confidence (>=8/10)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number stat-warning">{med_conf}</div>
                <div class="stat-label">Investigate (5-7/10)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number stat-success">{summary_stats.get("files_scanned", 0)}</div>
                <div class="stat-label">Files Scanned</div>
            </div>
        </div>

        {findings_content}
    </div>
</body>
</html>
"""
    return html_template
