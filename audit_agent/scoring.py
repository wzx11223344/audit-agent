"""
Quantitative Audit Scoring System.

The AuditScorer computes composite scores from the structured outputs of all
six audit stages. It provides multiple views of audit quality: aggregate scores,
severity distributions, red flag summaries, and traffic-light visualizations.

Scoring Philosophy:
    - Higher scores = better document quality.
    - Stage scores are inverted from problem counts (a document with many
      critical issues scores low; a clean document scores high).
    - Weighted composite provides the final pass/fail decision.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

from audit_agent.prompts import STAGE_LABELS


class AuditScorer:
    """Computes and aggregates scores across all six audit stages.

    The scorer takes the structured JSON output from each stage, computes
    per-stage scores, and produces a weighted composite score with detailed
    breakdowns and visualizations.

    Attributes:
        weights: Dict mapping stage keys to their weights in the composite.
        pass_threshold: Minimum composite score to pass the audit.
        severity_thresholds: Score boundaries for severity levels.
    """

    # Default weights matching config/audit_rules.yaml
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "coherence_check": 0.20,
        "claim_extraction": 0.15,
        "assumption_surfacing": 0.15,
        "stakeholder_analysis": 0.15,
        "methodology_review": 0.20,
        "bias_detection": 0.15,
    }

    # Severity score thresholds
    DEFAULT_SEVERITY_THRESHOLDS: Dict[str, int] = {
        "critical": 85,
        "high": 65,
        "medium": 40,
        "low": 15,
    }

    # Human-readable stage key mapping
    STAGE_KEY_MAP: Dict[str, str] = {
        "coherence_check": "coherence_check",
        "claim_extraction": "claim_extraction",
        "assumption_surfacing": "assumption_surfacing",
        "stakeholder_analysis": "stakeholder_analysis",
        "methodology_review": "methodology_review",
        "bias_detection": "bias_detection",
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        pass_threshold: float = 70.0,
        severity_thresholds: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize the scorer with configurable weights and thresholds.

        Args:
            weights: Dict mapping stage keys to weight values. Must sum to 1.0.
            pass_threshold: Minimum composite score (0-100) to pass.
            severity_thresholds: Dict mapping severity labels to minimum scores.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.pass_threshold = pass_threshold
        self.severity_thresholds = (
            severity_thresholds or self.DEFAULT_SEVERITY_THRESHOLDS.copy()
        )
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that stage weights are properly configured."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Stage weights must sum to 1.0, got {total:.3f}. "
                f"Current weights: {self.weights}"
            )

        expected_stages = set(self.DEFAULT_WEIGHTS.keys())
        actual_stages = set(self.weights.keys())
        if actual_stages != expected_stages:
            missing = expected_stages - actual_stages
            extra = actual_stages - expected_stages
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing stages: {missing}")
            if extra:
                msg_parts.append(f"Unknown stages: {extra}")
            raise ValueError(". ".join(msg_parts))

    def _classify_severity(self, score: float) -> str:
        """Classify a numeric score into a severity label.

        Args:
            score: Numeric score (0-100, higher = more severe problem).

        Returns:
            Severity label: critical, high, medium, low, or info.
        """
        if score >= self.severity_thresholds.get("critical", 85):
            return "critical"
        elif score >= self.severity_thresholds.get("high", 65):
            return "high"
        elif score >= self.severity_thresholds.get("medium", 40):
            return "medium"
        elif score >= self.severity_thresholds.get("low", 15):
            return "low"
        else:
            return "info"

    def _extract_findings(
        self, stage_output: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract the list of findings from a stage's structured output.

        Different stages use different keys for their findings list.
        This method normalizes extraction across stages.

        Args:
            stage_output: The parsed JSON output from a stage.

        Returns:
            A list of finding/item dicts, or empty list if none found.
        """
        for key in ("findings", "claims", "assumptions", "stakeholders",
                     "methods_identified", "concerns"):
            if key in stage_output and isinstance(stage_output[key], list):
                return stage_output[key]
        return []

    def _compute_stage_score(
        self, stage_output: Dict[str, Any]
    ) -> float:
        """Compute a normalized stage score from findings.

        Uses the stage's self-reported overall_score if available, otherwise
        computes it from the severity distribution of findings.

        Args:
            stage_output: The parsed JSON output from a stage.

        Returns:
            Normalized score (0-100, higher = better quality).
        """
        if "overall_score" in stage_output:
            score = float(stage_output["overall_score"])
            return max(0.0, min(100.0, score))

        # Fallback: compute from finding severities
        findings = self._extract_findings(stage_output)
        if not findings:
            return 100.0

        total_penalty = 0.0
        for finding in findings:
            sev_score = finding.get("severity_score", 50)
            severity = finding.get("severity", "medium")

            # Weight by severity
            severity_multiplier = {
                "critical": 1.0,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.15,
                "info": 0.05,
            }.get(severity, 0.4)

            total_penalty += sev_score * severity_multiplier

        # Normalize: fewer findings = less penalty
        max_penalty = len(findings) * 100
        if max_penalty == 0:
            return 100.0

        raw_score = 100.0 - (total_penalty / max_penalty * 100.0)
        return max(0.0, min(100.0, raw_score))

    def aggregate_score(
        self, stage_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute the weighted composite score across all stages.

        Args:
            stage_results: Dict mapping stage keys to their parsed JSON outputs.

        Returns:
            Dict with composite_score, per_stage_scores, weight_details, and
            pass/fail status.

        Raises:
            ValueError: If required stages are missing from stage_results.
        """
        per_stage_scores: Dict[str, float] = {}
        weight_details: Dict[str, float] = {}

        for stage_key in self.weights:
            if stage_key not in stage_results:
                raise ValueError(
                    f"Missing stage result: '{stage_key}'. "
                    f"Available: {list(stage_results.keys())}"
                )
            score = self._compute_stage_score(stage_results[stage_key])
            per_stage_scores[stage_key] = score
            weight_details[stage_key] = score * self.weights[stage_key]

        composite_score = sum(weight_details.values())

        return {
            "composite_score": round(composite_score, 1),
            "pass": composite_score >= self.pass_threshold,
            "pass_threshold": self.pass_threshold,
            "per_stage_scores": per_stage_scores,
            "weighted_contributions": {
                k: round(v, 1) for k, v in weight_details.items()
            },
        }

    def severity_distribution(
        self, stage_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """Compute a histogram of issue severities across all stages.

        Args:
            stage_results: Dict mapping stage keys to their parsed JSON outputs.

        Returns:
            Dict with counts for critical, high, medium, low, and info.
        """
        distribution: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for stage_key, output in stage_results.items():
            findings = self._extract_findings(output)
            for finding in findings:
                sev_score = finding.get("severity_score", 50)
                severity = self._classify_severity(sev_score)
                distribution[severity] = distribution.get(severity, 0) + 1

        return distribution

    def red_flag_summary(
        self, stage_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract only critical and high severity findings across all stages.

        Args:
            stage_results: Dict mapping stage keys to their parsed JSON outputs.

        Returns:
            List of finding dicts with severity 'critical' or 'high',
            sorted by severity_score descending.
        """
        red_flags: List[Dict[str, Any]] = []

        for stage_key, output in stage_results.items():
            findings = self._extract_findings(output)
            for finding in findings:
                sev_score = finding.get("severity_score", 50)
                severity = self._classify_severity(sev_score)
                if severity in ("critical", "high"):
                    finding_with_stage = dict(finding)
                    finding_with_stage["_stage"] = stage_key
                    finding_with_stage["_stage_label"] = STAGE_LABELS.get(
                        stage_key, stage_key
                    )
                    red_flags.append(finding_with_stage)

        red_flags.sort(
            key=lambda f: f.get("severity_score", 50), reverse=True
        )
        return red_flags

    def traffic_light_map(
        self, stage_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate a green/yellow/red assessment per section.

        Args:
            stage_results: Dict mapping stage keys to their parsed JSON outputs.

        Returns:
            Dict mapping stage keys to traffic light colors:
            'green' (>=80), 'yellow' (50-79), 'red' (<50).
        """
        traffic_map: Dict[str, str] = {}

        for stage_key, output in stage_results.items():
            score = self._compute_stage_score(output)
            if score >= 80:
                traffic_map[stage_key] = "green"
            elif score >= 50:
                traffic_map[stage_key] = "yellow"
            else:
                traffic_map[stage_key] = "red"

        return traffic_map

    def pass_threshold_check(
        self, score: float, threshold: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Check if a score passes the audit threshold.

        Args:
            score: Composite score to check.
            threshold: Override threshold (default: self.pass_threshold).

        Returns:
            Tuple of (passed: bool, message: str).
        """
        threshold = threshold if threshold is not None else self.pass_threshold
        passed = score >= threshold

        if passed:
            msg = (
                f"PASS: Composite score {score:.1f} meets the "
                f"threshold of {threshold:.0f}."
            )
        else:
            msg = (
                f"FAIL: Composite score {score:.1f} is below the "
                f"threshold of {threshold:.0f}."
            )

        return passed, msg

    def summary(self, stage_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a human-readable one-line scoring summary.

        Args:
            stage_results: Dict mapping stage keys to their parsed JSON outputs.

        Returns:
            A formatted summary string.
        """
        aggregate = self.aggregate_score(stage_results)
        dist = self.severity_distribution(stage_results)
        traffic = self.traffic_light_map(stage_results)

        green_count = sum(1 for v in traffic.values() if v == "green")
        yellow_count = sum(1 for v in traffic.values() if v == "yellow")
        red_count = sum(1 for v in traffic.values() if v == "red")

        passed, status_msg = self.pass_threshold_check(
            aggregate["composite_score"]
        )

        parts = [
            f"[{'PASS' if passed else 'FAIL'}] Score: {aggregate['composite_score']:.1f}/100",
            f"Traffic Lights: {green_count}G {yellow_count}Y {red_count}R",
            f"Issues: {dist['critical']} critical, {dist['high']} high, "
            f"{dist['medium']} medium, {dist['low']} low",
        ]
        return " | ".join(parts)
