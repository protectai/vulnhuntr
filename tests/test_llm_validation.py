"""Tests for LLM response validation with invalid enum handling."""
import pytest
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List
from enum import Enum

# Import the LLM class and test with a mock response model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vulnhuntr.LLMs import LLM, LLMError


# Mock VulnType enum matching the real one
class MockVulnType(str, Enum):
    LFI = "LFI"
    RCE = "RCE"
    SSRF = "SSRF"
    AFO = "AFO"
    SQLI = "SQLI"
    XSS = "XSS"
    IDOR = "IDOR"


# Mock Response model matching the real one
class MockResponse(BaseModel):
    scratchpad: str = Field(description="Analysis process")
    analysis: str = Field(description="Final analysis")
    poc: str = Field(description="Proof-of-concept")
    confidence_score: int = Field(description="0-10 confidence")
    vulnerability_types: List[MockVulnType] = Field(description="Vulnerability types")


class TestLLMValidation:
    """Test suite for LLM response validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = LLM(system_prompt="test")

    def test_valid_response_passes(self):
        """Test that a valid response with correct enum values passes validation."""
        valid_json = json.dumps({
            "scratchpad": "Testing analysis",
            "analysis": "Found RCE vulnerability",
            "poc": "curl -X POST",
            "confidence_score": 8,
            "vulnerability_types": ["RCE", "SQLI"]
        })

        result = self.llm._validate_response(valid_json, MockResponse)
        assert isinstance(result, MockResponse)
        assert result.vulnerability_types == [MockVulnType.RCE, MockVulnType.SQLI]
        assert result.confidence_score == 8

    def test_invalid_enum_filtered_out(self):
        """Test that invalid enum values like 'DoS' are filtered out gracefully."""
        invalid_json = json.dumps({
            "scratchpad": "Testing analysis",
            "analysis": "Found DoS vulnerability",
            "poc": "N/A",
            "confidence_score": 5,
            "vulnerability_types": ["DoS"]
        })

        result = self.llm._validate_response(invalid_json, MockResponse)
        assert isinstance(result, MockResponse)
        assert result.vulnerability_types == []  # DoS filtered out
        assert result.confidence_score == 5

    def test_mixed_valid_invalid_enums(self):
        """Test that valid enums are kept while invalid ones are filtered."""
        mixed_json = json.dumps({
            "scratchpad": "Testing analysis",
            "analysis": "Found multiple vulnerabilities",
            "poc": "exploit code",
            "confidence_score": 7,
            "vulnerability_types": ["RCE", "DoS", "SQLI", "CSRF", "XSS"]
        })

        result = self.llm._validate_response(mixed_json, MockResponse)
        assert isinstance(result, MockResponse)
        # Only RCE, SQLI, XSS should remain (DoS and CSRF filtered)
        assert set(result.vulnerability_types) == {MockVulnType.RCE, MockVulnType.SQLI, MockVulnType.XSS}
        assert len(result.vulnerability_types) == 3

    def test_all_invalid_enums_returns_empty_list(self):
        """Test that when all enum values are invalid, we get an empty list."""
        all_invalid_json = json.dumps({
            "scratchpad": "Testing analysis",
            "analysis": "Found vulnerabilities",
            "poc": "N/A",
            "confidence_score": 3,
            "vulnerability_types": ["DoS", "CSRF", "BufferOverflow"]
        })

        result = self.llm._validate_response(all_invalid_json, MockResponse)
        assert isinstance(result, MockResponse)
        assert result.vulnerability_types == []

    def test_wrapped_json_with_think_tags(self):
        """Test extraction of JSON from response wrapped in <think> tags."""
        wrapped_response = """<think>

        Let me analyze this code...

        </think>

        {"scratchpad": "Analysis process", "analysis": "Found RCE", "poc": "test", "confidence_score": 9, "vulnerability_types": ["RCE"]}"""

        result = self.llm._validate_response(wrapped_response, MockResponse)
        assert isinstance(result, MockResponse)
        assert result.vulnerability_types == [MockVulnType.RCE]
        assert result.confidence_score == 9

    def test_json_with_markdown_code_block(self):
        """Test extraction of JSON from markdown code blocks."""
        markdown_response = """Here's my analysis:

```json
{"scratchpad": "Analysis", "analysis": "Found SQLI", "poc": "' OR 1=1", "confidence_score": 10, "vulnerability_types": ["SQLI", "XSS"]}
```
"""

        result = self.llm._validate_response(markdown_response, MockResponse)
        assert isinstance(result, MockResponse)
        assert set(result.vulnerability_types) == {MockVulnType.SQLI, MockVulnType.XSS}

    def test_malformed_json_raises_error(self):
        """Test that truly malformed JSON still raises an error."""
        malformed = "This is not JSON at all, no braces here!"

        with pytest.raises(LLMError, match="Validation failed"):
            self.llm._validate_response(malformed, MockResponse)

    def test_missing_required_fields_raises_error(self):
        """Test that JSON missing required fields raises an error."""
        incomplete_json = json.dumps({
            "scratchpad": "Test",
            "vulnerability_types": ["RCE"]
            # Missing: analysis, poc, confidence_score
        })

        with pytest.raises(LLMError, match="Validation failed"):
            self.llm._validate_response(incomplete_json, MockResponse)

    def test_prefill_handling_with_invalid_enums(self):
        """Test that prefill is properly prepended and invalid enums filtered."""
        self.llm.prefill = '{"scratchpad": "'
        partial_json = 'Analysis here", "analysis": "Found issues", "poc": "test", "confidence_score": 6, "vulnerability_types": ["AFO", "DoS"]}'

        result = self.llm._validate_response(partial_json, MockResponse)
        assert isinstance(result, MockResponse)
        assert result.vulnerability_types == [MockVulnType.AFO]  # DoS filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
