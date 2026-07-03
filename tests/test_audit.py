"""
Unit tests for AuditAgent.

Tests cover:
    - Scoring logic (aggregate, severity, traffic lights)
    - Reporter output generation
    - Prompt template resolution
    - Config file loading
    - Stage pipeline orchestration (with mock LLM)
    - AuditResult container methods

Run with:
    pytest tests/test_audit.py -v
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from audit_agent.scoring import AuditScorer
from audit_agent.reporter import Reporter
from audit_agent.prompts import (
    get_stage_prompt,
    STAGE_ORDER,
    STAGE_LABELS,
    get_prior_stage_context,
)
from audit_agent.stages import BaseStage, STAGE_CLASSES, get_stage_class


# ==============================================================================
# Sample Data Fixtures
# ==============================================================================

SAMPLE_COHERENCE_OUTPUT = {
    "stage": "coherence_check",
    "overall_score": 72,
    "findings": [
        {
            "finding_id": "COH-1",
            "issue_type": "contradiction",
            "severity": "high",
            "severity_score": 75,
            "location": "Line 12 vs. Line 28",
            "evidence": "emissions will reduce by 40%",
            "description": "Two different emission reduction figures stated without reconciliation.",
            "suggested_fix": "Clarify and reconcile figures.",
        },
        {
            "finding_id": "COH-2",
            "issue_type": "undefined_term",
            "severity": "medium",
            "severity_score": 45,
            "location": "Line 8",
            "evidence": "double dividend",
            "description": "Term 'double dividend' used without definition.",
            "suggested_fix": "Define the term.",
        },
    ],
}

SAMPLE_CLAIMS_OUTPUT = {
    "stage": "claim_extraction",
    "overall_score": 55,
    "total_claims": 3,
    "claims": [
        {
            "claim_id": "CLM-1",
            "statement": "carbon tax will reduce emissions by 22% within five years",
            "location": "Line 9",
            "domain": "economics",
            "claim_type": "quantitative_prediction",
            "verifiability_score": 45,
            "support_level": "asserted",
            "citation_provided": "none",
            "flags": ["uncited_statistic"],
        },
    ],
}

SAMPLE_ASSUMPTIONS_OUTPUT = {
    "stage": "assumption_surfacing",
    "overall_score": 60,
    "assumptions": [
        {
            "assumption_id": "ASM-1",
            "assumption": "Firms will pass carbon costs to consumers",
            "assumption_type": "behavioral",
            "impact_if_wrong": "critical",
            "severity": "high",
            "severity_score": 70,
        },
    ],
}

SAMPLE_STAKEHOLDER_OUTPUT = {
    "stage": "stakeholder_analysis",
    "overall_score": 50,
    "stakeholders": [
        {
            "stakeholder_id": "SH-1",
            "stakeholder": "Low-income households",
            "impact_direction": "negative",
            "impact_magnitude": "large",
            "addressed_in_doc": "partially",
            "severity": "high",
            "severity_score": 72,
        },
    ],
}

SAMPLE_METHODOLOGY_OUTPUT = {
    "stage": "methodology_review",
    "overall_score": 40,
    "methods_identified": [
        {
            "method_id": "MTH-1",
            "claim_reference": "CLM-1",
            "method_identified": "CGE modeling (implied)",
            "method_status": "implied",
            "concerns": [
                {
                    "concern_type": "model_specification",
                    "severity": "high",
                    "severity_score": 78,
                    "description": "No model structure described.",
                },
            ],
            "overall_method_quality": "inadequate",
        },
    ],
}

SAMPLE_BIAS_OUTPUT = {
    "stage": "bias_detection",
    "overall_score": 55,
    "findings": [
        {
            "finding_id": "BIA-1",
            "bias_type": "cherry_picking",
            "severity": "high",
            "severity_score": 72,
            "location": "Lines 18-22",
            "evidence": "studies show that carbon pricing is the single most effective tool",
            "description": "Cites only pro-carbon-tax research without acknowledging mixed evidence.",
            "reframing_suggestion": "Acknowledge range of findings.",
        },
    ],
}

COMPLETE_STAGE_OUTPUTS = {
    "coherence_check": SAMPLE_COHERENCE_OUTPUT,
    "claim_extraction": SAMPLE_CLAIMS_OUTPUT,
    "assumption_surfacing": SAMPLE_ASSUMPTIONS_OUTPUT,
    "stakeholder_analysis": SAMPLE_STAKEHOLDER_OUTPUT,
    "methodology_review": SAMPLE_METHODOLOGY_OUTPUT,
    "bias_detection": SAMPLE_BIAS_OUTPUT,
}


# ==============================================================================
# Scoring Tests
# ==============================================================================

class TestAuditScorer(unittest.TestCase):
    """Tests for the AuditScorer class."""

    def setUp(self) -> None:
        self.scorer = AuditScorer()

    def test_aggregate_score_valid(self) -> None:
        """Test aggregate scoring with valid stage outputs."""
        result = self.scorer.aggregate_score(COMPLETE_STAGE_OUTPUTS)

        self.assertIn("composite_score", result)
        self.assertIn("pass", result)
        self.assertIn("per_stage_scores", result)
        self.assertIn("weighted_contributions", result)

        # Score should be between 0 and 100
        self.assertGreaterEqual(result["composite_score"], 0)
        self.assertLessEqual(result["composite_score"], 100)

        # All six stages should be scored
        self.assertEqual(len(result["per_stage_scores"]), 6)
        for stage_key in STAGE_ORDER:
            self.assertIn(stage_key, result["per_stage_scores"])

    def test_aggregate_score_missing_stage_raises(self) -> None:
        """Test that missing stage raises ValueError."""
        incomplete = {"coherence_check": SAMPLE_COHERENCE_OUTPUT}
        with self.assertRaises(ValueError):
            self.scorer.aggregate_score(incomplete)

    def test_severity_distribution(self) -> None:
        """Test severity histogram computation."""
        dist = self.scorer.severity_distribution(COMPLETE_STAGE_OUTPUTS)

        self.assertIn("critical", dist)
        self.assertIn("high", dist)
        self.assertIn("medium", dist)
        self.assertIn("low", dist)

        total = sum(dist.values())
        self.assertGreater(total, 0)

    def test_red_flag_summary(self) -> None:
        """Test that red flags only contain critical/high severity items."""
        flags = self.scorer.red_flag_summary(COMPLETE_STAGE_OUTPUTS)

        for flag in flags:
            sev = flag.get("severity", "")
            sev_score = flag.get("severity_score", 50)
            self.assertTrue(
                sev in ("critical", "high") or sev_score >= 65,
                f"Expected critical or high, got {sev} ({sev_score})"
            )

    def test_traffic_light_map(self) -> None:
        """Test traffic light generation."""
        traffic = self.scorer.traffic_light_map(COMPLETE_STAGE_OUTPUTS)

        self.assertEqual(len(traffic), 6)
        for stage_key in STAGE_ORDER:
            self.assertIn(stage_key, traffic)
            self.assertIn(traffic[stage_key], ("green", "yellow", "red"))

    def test_pass_threshold_passes(self) -> None:
        """Test pass threshold with a high score."""
        passed, msg = self.scorer.pass_threshold_check(85.0)
        self.assertTrue(passed)
        self.assertIn("PASS", msg)

    def test_pass_threshold_fails(self) -> None:
        """Test pass threshold with a low score."""
        passed, msg = self.scorer.pass_threshold_check(45.0)
        self.assertFalse(passed)
        self.assertIn("FAIL", msg)

    def test_custom_weights(self) -> None:
        """Test scorer with custom weights."""
        custom = {
            "coherence_check": 0.30,
            "claim_extraction": 0.10,
            "assumption_surfacing": 0.10,
            "stakeholder_analysis": 0.10,
            "methodology_review": 0.30,
            "bias_detection": 0.10,
        }
        scorer = AuditScorer(weights=custom)
        result = scorer.aggregate_score(COMPLETE_STAGE_OUTPUTS)
        self.assertGreaterEqual(result["composite_score"], 0)

    def test_invalid_weights_raises(self) -> None:
        """Test that non-summing weights raise ValueError."""
        bad_weights = {
            "coherence_check": 0.50,
            "claim_extraction": 0.30,
            "assumption_surfacing": 0.10,
            "stakeholder_analysis": 0.05,
            "methodology_review": 0.05,
            "bias_detection": 0.00,  # Total = 1.00 but missing bias
        }
        # Test with mismatched stages
        bad_weights2 = {
            "coherence_check": 0.25,
            "claim_extraction": 0.25,
            "assumption_surfacing": 0.25,
            "stakeholder_analysis": 0.25,
            "methodology_review": 0.25,
            "bias_detection": 0.25,  # Total = 1.50
        }
        with self.assertRaises(ValueError):
            AuditScorer(weights=bad_weights2)

    def test_empty_findings_full_score(self) -> None:
        """Test that empty findings yield 100% score."""
        empty_outputs = {
            k: {"stage": k, "overall_score": 100,
                k.replace("_check", "").replace("_extraction", "").replace(
                    "_surfacing", "").replace("_analysis", "").replace(
                    "_review", "").replace("_detection", ""):
                []}
            for k in STAGE_ORDER
        }
        # Rebuild with consistent key
        clean = {}
        for k in STAGE_ORDER:
            clean[k] = {"stage": k, "overall_score": 100}
        result = self.scorer.aggregate_score(clean)
        self.assertGreaterEqual(result["composite_score"], 90)

    def test_summary_string(self) -> None:
        """Test the one-line summary generation."""
        summary = self.scorer.summary(COMPLETE_STAGE_OUTPUTS)
        self.assertIsInstance(summary, str)
        self.assertIn("Score:", summary)


# ==============================================================================
# Reporter Tests
# ==============================================================================

class TestReporter(unittest.TestCase):
    """Tests for the Reporter class."""

    def setUp(self) -> None:
        self.scorer = AuditScorer()
        self.scoring = self.scorer.aggregate_score(COMPLETE_STAGE_OUTPUTS)
        self.reporter = Reporter(
            document_name="Test Document",
            audit_timestamp="2026-01-01T00:00:00",
        )
        self.audit_data = {
            "stage_outputs": COMPLETE_STAGE_OUTPUTS,
            "scoring": self.scoring,
        }

    def test_generate_report(self) -> None:
        """Test text report generation."""
        report = self.reporter.generate_report(self.audit_data, self.scorer)
        self.assertIsInstance(report, str)
        self.assertIn("AuditAgent", report)
        self.assertIn("Test Document", report)
        self.assertIn("综合评分", report)

    def test_executive_summary(self) -> None:
        """Test executive summary generation."""
        summary = self.reporter.executive_summary(self.audit_data)
        self.assertIsInstance(summary, str)
        self.assertIn("Test Document", summary)
        # Should be under max_length
        self.assertLessEqual(len(summary), 500)

    def test_executive_summary_max_length(self) -> None:
        """Test executive summary respects max_length."""
        summary = self.reporter.executive_summary(self.audit_data, max_length=100)
        self.assertLessEqual(len(summary), 100)

    def test_action_items(self) -> None:
        """Test action items extraction."""
        items = self.reporter.action_items(self.audit_data)
        self.assertIsInstance(items, list)
        for item in items:
            self.assertIn("stage", item)
            self.assertIn("finding_id", item)
            self.assertIn("severity", item)
            self.assertIn("issue", item)
            self.assertIn("recommendation", item)

    def test_generate_html(self) -> None:
        """Test HTML report generation."""
        html = self.reporter.generate_html(self.audit_data, self.scorer)
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("AuditAgent Report", html)
        self.assertIn("Severity Distribution", html)
        # Should be self-contained (no external CDN references)
        self.assertNotIn("<link rel=", html.lower())

    def test_export_json(self) -> None:
        """Test JSON export."""
        json_str = self.reporter.export_json(self.audit_data)
        self.assertIsInstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertIn("audit_agent_version", parsed)
        self.assertIn("scoring", parsed)
        self.assertIn("stage_outputs", parsed)

    def test_reporter_without_scorer(self) -> None:
        """Test reporter works without a scorer."""
        report = self.reporter.generate_report(self.audit_data)
        self.assertIn("AuditAgent", report)

    def test_escape_html(self) -> None:
        """Test HTML escaping."""
        text = '<script>alert("XSS")</script>'
        escaped = Reporter._escape_html(text)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;", escaped)
        self.assertIn("&quot;", escaped)


# ==============================================================================
# Prompts Tests
# ==============================================================================

class TestPrompts(unittest.TestCase):
    """Tests for the prompts module."""

    def test_get_all_stage_prompts(self) -> None:
        """Test that all stage prompts are retrievable."""
        for stage_key in STAGE_ORDER:
            prompt = get_stage_prompt(stage_key)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 200,
                f"Prompt for {stage_key} seems too short ({len(prompt)} chars)")
            self.assertIn("DOCUMENT TO AUDIT:", prompt)

    def test_get_invalid_stage_raises(self) -> None:
        """Test that invalid stage name raises ValueError."""
        with self.assertRaises(ValueError):
            get_stage_prompt("nonexistent_stage")

    def test_prior_stage_context(self) -> None:
        """Test prior stage context injection."""
        context = get_prior_stage_context(
            "一致性检查 (Coherence Check)",
            SAMPLE_COHERENCE_OUTPUT,
        )
        self.assertIn("CONTEXT FROM PREVIOUS", context)
        self.assertIn("COH-1", context)

    def test_stage_labels_have_all_stages(self) -> None:
        """Test that all stages have labels."""
        for stage_key in STAGE_ORDER:
            self.assertIn(stage_key, STAGE_LABELS)
            self.assertIsInstance(STAGE_LABELS[stage_key], str)


# ==============================================================================
# Stages Tests
# ==============================================================================

class TestStages(unittest.TestCase):
    """Tests for the stage classes."""

    def test_get_stage_class_valid(self) -> None:
        """Test stage class retrieval by index."""
        cls = get_stage_class(0)
        self.assertTrue(issubclass(cls, BaseStage))

    def test_get_stage_class_index_error(self) -> None:
        """Test out-of-range index raises IndexError."""
        with self.assertRaises(IndexError):
            get_stage_class(-1)
        with self.assertRaises(IndexError):
            get_stage_class(99)

    def test_all_stages_have_stage_name(self) -> None:
        """Test that all stage classes have a stage_name."""
        for cls in STAGE_CLASSES:
            self.assertTrue(hasattr(cls, "stage_name"))
            self.assertIsInstance(cls.stage_name, str)
            self.assertGreater(len(cls.stage_name), 0)

    @patch("audit_agent.stages.CoherenceCheck._call")
    def test_coherence_check_run(self, mock_call: MagicMock) -> None:
        """Test CoherenceCheck stage execution with mock LLM."""
        mock_call.return_value = json.dumps(SAMPLE_COHERENCE_OUTPUT)
        mock_client = MagicMock()

        from audit_agent.stages import CoherenceCheck
        stage = CoherenceCheck(mock_client)
        result = stage.run("test document", model_name="test-model")

        self.assertEqual(result["overall_score"], 72)
        self.assertEqual(len(result["findings"]), 2)

    @patch("audit_agent.stages.ClaimExtractor._call")
    def test_stage_with_prior_context(self, mock_call: MagicMock) -> None:
        """Test that prior context is passed between stages."""
        mock_call.return_value = json.dumps(SAMPLE_CLAIMS_OUTPUT)
        mock_client = MagicMock()

        from audit_agent.stages import ClaimExtractor
        stage = ClaimExtractor(mock_client)
        result = stage.run(
            "test document",
            prior_stage_output={"coherence_check": SAMPLE_COHERENCE_OUTPUT},
            model_name="test-model",
        )
        self.assertIn("claims", result)


# ==============================================================================
# Core / AuditResult Tests
# ==============================================================================

class TestAuditResult(unittest.TestCase):
    """Tests for the AuditResult container class."""

    def setUp(self) -> None:
        from audit_agent.core import AuditResult
        self.scorer = AuditScorer()
        self.scoring = self.scorer.aggregate_score(COMPLETE_STAGE_OUTPUTS)
        self.reporter = Reporter("Test Doc", "2026-01-01T00:00:00")
        self.result = AuditResult(
            document_name="Test Doc",
            document_text="Test content",
            stage_outputs=COMPLETE_STAGE_OUTPUTS,
            scoring=self.scoring,
            scorer=self.scorer,
            reporter=self.reporter,
        )

    def test_composite_score_property(self) -> None:
        """Test composite_score property."""
        self.assertIsInstance(self.result.composite_score, float)

    def test_passed_property(self) -> None:
        """Test passed property."""
        self.assertIsInstance(self.result.passed, bool)

    def test_executive_summary(self) -> None:
        """Test executive summary on result."""
        summary = self.result.executive_summary()
        self.assertIn("Test Doc", summary)

    def test_red_flags(self) -> None:
        """Test red flags extraction from result."""
        flags = self.result.red_flags()
        self.assertIsInstance(flags, list)

    def test_traffic_lights(self) -> None:
        """Test traffic lights from result."""
        lights = self.result.traffic_lights()
        self.assertEqual(len(lights), 6)

    def test_action_items(self) -> None:
        """Test action items from result."""
        items = self.result.action_items()
        self.assertIsInstance(items, list)


# ==============================================================================
# AuditAgent Tests
# ==============================================================================

class TestAuditAgent(unittest.TestCase):
    """Tests for the AuditAgent class."""

    @patch("audit_agent.core.OpenAI")
    @patch("audit_agent.core.AuditAgent._load_rules")
    def test_agent_initialization(
        self, mock_load_rules: MagicMock, mock_openai: MagicMock
    ) -> None:
        """Test agent initialization with default config."""
        mock_load_rules.return_value = {
            "llm": {
                "local": {
                    "base_url": "http://localhost:11434/v1",
                    "model_name": "qwen2.5:7b",
                    "max_tokens": 4096,
                    "temperature": 0.0,
                },
            },
            "scoring": {
                "coherence_weight": 0.20,
                "claims_weight": 0.15,
                "assumptions_weight": 0.15,
                "stakeholders_weight": 0.15,
                "methodology_weight": 0.20,
                "bias_weight": 0.15,
                "pass_threshold": 70,
            },
            "severity_thresholds": {
                "critical": 85,
                "high": 65,
                "medium": 40,
                "low": 15,
            },
        }

        from audit_agent.core import AuditAgent
        agent = AuditAgent(model="local")

        self.assertEqual(agent.model, "local")
        self.assertEqual(agent.model_name, "qwen2.5:7b")
        self.assertIsNotNone(agent.scorer)

    def test_read_document_file(self) -> None:
        """Test reading document from file."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Test document content")
            f.flush()
            temp_path = f.name

        try:
            from audit_agent.core import AuditAgent
            with patch("audit_agent.core.OpenAI"), \
                 patch.object(AuditAgent, "_load_rules", return_value={}):
                agent = AuditAgent(model="local")
                name, text = agent._read_document(temp_path)
                self.assertIn("tmp", name)  # Temporary file name
                self.assertEqual(text, "Test document content")
        finally:
            os.unlink(temp_path)

    def test_read_document_raw_text(self) -> None:
        """Test handling of raw text input."""
        from audit_agent.core import AuditAgent
        with patch("audit_agent.core.OpenAI"), \
             patch.object(AuditAgent, "_load_rules", return_value={}):
            agent = AuditAgent(model="local")
            name, text = agent._read_document("inline text content")
            self.assertEqual(name, "Inline Document")
            self.assertEqual(text, "inline text content")

    def test_list_stages(self) -> None:
        """Test stage listing."""
        from audit_agent.core import AuditAgent
        with patch("audit_agent.core.OpenAI"), \
             patch.object(AuditAgent, "_load_rules", return_value={}):
            agent = AuditAgent(model="local")
            stages = agent.list_stages()
            self.assertEqual(len(stages), 6)
            for stage in stages:
                self.assertIn("key", stage)
                self.assertIn("label", stage)
                self.assertIn("description", stage)

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model parameter raises ValueError."""
        from audit_agent.core import AuditAgent
        with patch("audit_agent.core.OpenAI"), \
             patch.object(AuditAgent, "_load_rules", return_value={}):
            with self.assertRaises(ValueError):
                AuditAgent(model="invalid")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
