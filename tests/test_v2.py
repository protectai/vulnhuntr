import pytest
from pathlib import Path
from vulnhuntr.symbol_finder import SymbolExtractor
from vulnhuntr.reporting import generate_sarif, generate_html_report
from vulnhuntr.prompts import VULN_SPECIFIC_BYPASSES_AND_PROMPTS
from vulnhuntr.__main__ import VulnType, Response


def test_vulnerability_types():
    assert len(VULN_SPECIFIC_BYPASSES_AND_PROMPTS) == 11
    assert "SSTI" in VULN_SPECIFIC_BYPASSES_AND_PROMPTS
    assert "DESERIALIZATION" in VULN_SPECIFIC_BYPASSES_AND_PROMPTS
    assert "AUTH_BYPASS" in VULN_SPECIFIC_BYPASSES_AND_PROMPTS
    assert "PATH_TRAVERSAL" in VULN_SPECIFIC_BYPASSES_AND_PROMPTS
    assert VulnType.SSTI == "SSTI"


def test_ast_symbol_extraction():
    se = SymbolExtractor(".")
    # Search for SymbolExtractor class itself in symbol_finder.py
    file_path = Path("vulnhuntr/symbol_finder.py")
    res = se.ast_search("SymbolExtractor", [file_path])
    assert res is not None
    assert res["name"] == "SymbolExtractor"
    assert "class SymbolExtractor" in res["source"]


def test_sarif_generation():
    mock_findings = [{
        "file_path": "vulnhuntr/test_app.py",
        "report": {
            "scratchpad": "Analyzing route parameters.",
            "analysis": "Critical Remote Code Execution found in route handler.",
            "poc": "curl http://localhost:8000/eval?cmd=id",
            "remediation": "Do not pass user input directly to eval() or exec().",
            "confidence_score": 9,
            "vulnerability_types": ["RCE"]
        }
    }]
    sarif = generate_sarif(mock_findings, repo_root=".")
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1
    assert len(sarif["runs"][0]["results"]) == 1
    assert sarif["runs"][0]["results"][0]["ruleId"] == "VULN-RCE"
    assert sarif["runs"][0]["results"][0]["level"] == "error"


def test_html_report_generation():
    mock_findings = [{
        "file_path": "vulnhuntr/test_app.py",
        "report": {
            "scratchpad": "Analyzing route parameters.",
            "analysis": "Arbitrary File Read detected.",
            "poc": "curl http://localhost:8000/view?file=../../../../etc/passwd",
            "remediation": "Validate path with os.path.realpath and restrict to base dir.",
            "confidence_score": 8,
            "vulnerability_types": ["LFI"]
        }
    }]
    html = generate_html_report(mock_findings, {"files_scanned": 1})
    assert "<!DOCTYPE html>" in html
    assert "VulnHuntr v2.0" in html
    assert "Confidence: 8/10" in html
    assert "Defensive Mitigation & Remediation" in html
