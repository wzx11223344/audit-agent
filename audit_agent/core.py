"""
Main Audit Pipeline — AuditAgent Core.

The AuditAgent class is the central orchestrator that chains together the six
audit stages, manages LLM backends, and coordinates scoring and reporting.

Usage:
    from audit_agent import AuditAgent

    agent = AuditAgent(model="local")
    result = agent.audit("path/to/document.txt")
    print(result.executive_summary())
    result.export_html("output/report.html")
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import yaml
from openai import OpenAI

from audit_agent.stages import (
    CoherenceCheck,
    ClaimExtractor,
    AssumptionSurfacer,
    StakeholderAnalyzer,
    MethodologyReviewer,
    BiasDetector,
    STAGE_CLASSES,
)
from audit_agent.scoring import AuditScorer
from audit_agent.reporter import Reporter
from audit_agent.prompts import STAGE_LABELS, STAGE_DESCRIPTIONS, STAGE_ORDER

logger = logging.getLogger(__name__)


class AuditResult:
    """Container for the complete result of an audit run.

    This object provides convenient access to scoring, stage outputs,
    and report generation methods.

    Attributes:
        document_name: Name of the audited document.
        document_text: Full text of the audited document.
        stage_outputs: Dict mapping stage keys to their parsed JSON outputs.
        scoring: Dict with composite_score, per_stage_scores, etc.
        audit_timestamp: ISO-format timestamp of the audit.
    """

    def __init__(
        self,
        document_name: str,
        document_text: str,
        stage_outputs: Dict[str, Dict[str, Any]],
        scoring: Dict[str, Any],
        scorer: AuditScorer,
        reporter: Reporter,
    ) -> None:
        """Initialize the audit result container.

        Args:
            document_name: Name/title of the audited document.
            document_text: Full text content.
            stage_outputs: Parsed outputs from all six stages.
            scoring: Composite scoring results.
            scorer: The AuditScorer instance.
            reporter: Configured Reporter instance.
        """
        self.document_name = document_name
        self.document_text = document_text
        self.stage_outputs = stage_outputs
        self.scoring = scoring
        self._scorer = scorer
        self._reporter = reporter

    @property
    def composite_score(self) -> float:
        """The weighted composite audit score (0-100)."""
        return self.scoring.get("composite_score", 0.0)

    @property
    def passed(self) -> bool:
        """Whether the document passed the audit threshold."""
        return self.scoring.get("pass", False)

    def executive_summary(self, max_length: int = 500) -> str:
        """Generate a 1-page executive summary.

        Args:
            max_length: Maximum character length.

        Returns:
            Concise summary string.
        """
        return self._reporter.executive_summary(
            self._to_audit_data(), max_length
        )

    def red_flags(self) -> List[Dict[str, Any]]:
        """Get all critical and high severity findings.

        Returns:
            List of finding dicts sorted by severity.
        """
        return self._scorer.red_flag_summary(self.stage_outputs)

    def action_items(self) -> List[Dict[str, str]]:
        """Extract actionable improvement recommendations.

        Returns:
            List of action item dicts.
        """
        return self._reporter.action_items(self._to_audit_data())

    def traffic_lights(self) -> Dict[str, str]:
        """Get per-stage traffic light assessment.

        Returns:
            Dict mapping stage keys to 'green', 'yellow', or 'red'.
        """
        return self._scorer.traffic_light_map(self.stage_outputs)

    def generate_report(self) -> str:
        """Generate a complete text report.

        Returns:
            Formatted text report.
        """
        return self._reporter.generate_report(
            self._to_audit_data(), self._scorer
        )

    def generate_html(self) -> str:
        """Generate a self-contained HTML report.

        Returns:
            Complete HTML string.
        """
        return self._reporter.generate_html(
            self._to_audit_data(), self._scorer
        )

    def export_html(self, output_path: str) -> str:
        """Save HTML report to file.

        Args:
            output_path: Path for the HTML file.

        Returns:
            The output path.
        """
        return self._reporter.save_html_report(
            self._to_audit_data(), output_path, self._scorer
        )

    def export_json(self, output_path: str) -> str:
        """Save machine-readable JSON report to file.

        Args:
            output_path: Path for the JSON file.

        Returns:
            The output path.
        """
        return self._reporter.save_json_export(
            self._to_audit_data(), output_path
        )

    def export_text_report(self, output_path: str) -> str:
        """Save text report to file.

        Args:
            output_path: Path for the text file.

        Returns:
            The output path.
        """
        return self._reporter.save_text_report(
            self._to_audit_data(), output_path, self._scorer
        )

    def _to_audit_data(self) -> Dict[str, Any]:
        """Convert to the dict format expected by Reporter."""
        return {
            "stage_outputs": self.stage_outputs,
            "scoring": self.scoring,
        }


class AuditAgent:
    """Structured multi-stage LLM-powered document audit engine.

    This is NOT a generic chatbot. It follows a fixed 6-stage methodology
    with strict output schemas, quantitative scoring, and deterministic
    behavior (temperature=0.0).

    Attributes:
        model: The backend mode ('local' or 'api').
        client: The OpenAI-compatible client.
        model_name: The LLM model identifier.
        rules: Loaded audit rules from YAML config.
        scorer: Configured AuditScorer instance.
    """

    def __init__(
        self,
        model: str = "local",
        rules_path: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the audit agent.

        Args:
            model: Backend mode — 'local' (Ollama) or 'api' (OpenAI compatible).
            rules_path: Path to audit_rules.yaml (default: config/audit_rules.yaml).
            api_base: API base URL (overrides config).
            api_key: API key (overrides config).
            model_name: Model name (overrides config).
        """
        self.model = model

        # Load configuration
        self.rules = self._load_rules(rules_path)

        # Configure LLM
        llm_config = self.rules.get("llm", {})
        if model not in ("local", "api"):
            raise ValueError(f"Model must be 'local' or 'api', got '{model}'")
        mode_config = llm_config.get(model, {})

        base_url = api_base or mode_config.get("base_url", "http://localhost:11434/v1")
        key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.model_name = model_name or mode_config.get("model_name", "qwen2.5:7b")

        self.client = OpenAI(base_url=base_url, api_key=key)

        # Configure scorer
        scoring_config = self.rules.get("scoring", {})
        weights = {
            "coherence_check": scoring_config.get("coherence_weight", 0.20),
            "claim_extraction": scoring_config.get("claims_weight", 0.15),
            "assumption_surfacing": scoring_config.get("assumptions_weight", 0.15),
            "stakeholder_analysis": scoring_config.get("stakeholders_weight", 0.15),
            "methodology_review": scoring_config.get("methodology_weight", 0.20),
            "bias_detection": scoring_config.get("bias_weight", 0.15),
        }
        pass_threshold = scoring_config.get("pass_threshold", 70)

        sev_config = self.rules.get("severity_thresholds", {})
        severity_thresholds = {
            "critical": sev_config.get("critical", 85),
            "high": sev_config.get("high", 65),
            "medium": sev_config.get("medium", 40),
            "low": sev_config.get("low", 15),
        }

        self.scorer = AuditScorer(
            weights=weights,
            pass_threshold=pass_threshold,
            severity_thresholds=severity_thresholds,
        )

        logger.info(
            "AuditAgent initialized: model=%s, backend=%s, base_url=%s",
            model, self.model_name, base_url,
        )

    def _load_rules(self, rules_path: Optional[str] = None) -> Dict[str, Any]:
        """Load audit rules from YAML configuration file.

        Args:
            rules_path: Path to YAML file. Defaults to config/audit_rules.yaml
                        relative to the project root.

        Returns:
            Parsed configuration dictionary.
        """
        if rules_path is None:
            # Find config relative to this file
            project_root = Path(__file__).parent.parent
            rules_path = str(project_root / "config" / "audit_rules.yaml")

        if not os.path.exists(rules_path):
            logger.warning(
                "Rules file not found at '%s', using defaults", rules_path
            )
            return {}

        with open(rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _read_document(self, document_input: str) -> Tuple[str, str]:
        """Read document text from a file path or return raw text.

        Args:
            document_input: Either a file path or raw document text.

        Returns:
            Tuple of (document_name, document_text).
        """
        if os.path.isfile(document_input):
            path = Path(document_input)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return path.stem, text
        else:
            return "Inline Document", document_input

    def audit(self, document_path_or_text: str) -> AuditResult:
        """Run the full 6-stage audit pipeline on a document.

        This is the main entry point. It chains all six audit stages in order,
        passes context between them, computes scores, and returns a complete
        AuditResult.

        Args:
            document_path_or_text: Path to a document file or raw text.

        Returns:
            AuditResult containing all stage outputs, scores, and report
            generation methods.
        """
        doc_name, doc_text = self._read_document(document_path_or_text)
        logger.info("Starting audit of '%s' (%d chars)", doc_name, len(doc_text))

        return self._run_pipeline(doc_name, doc_text, verbose=False)

    def stage_by_stage(
        self, document_path_or_text: str, verbose: bool = False
    ) -> AuditResult:
        """Run the audit pipeline in interactive/debug mode.

        Unlike `audit()`, this method logs detailed progress for each stage
        and allows inspection of intermediate outputs.

        Args:
            document_path_or_text: Path to a document or raw text.
            verbose: If True, log stage outputs as they complete.

        Returns:
            AuditResult with all stage outputs and scores.
        """
        doc_name, doc_text = self._read_document(document_path_or_text)
        logger.info(
            "Starting stage-by-stage audit of '%s' (%d chars, verbose=%s)",
            doc_name, len(doc_text), verbose,
        )
        return self._run_pipeline(doc_name, doc_text, verbose=verbose)

    def _run_pipeline(
        self, doc_name: str, doc_text: str, verbose: bool = False
    ) -> AuditResult:
        """Execute the complete audit pipeline.

        Args:
            doc_name: Document name for reporting.
            doc_text: Full document text.
            verbose: Enable detailed stage logging.

        Returns:
            AuditResult with all outputs.
        """
        stage_outputs: Dict[str, Dict[str, Any]] = {}
        prior_output: Optional[Dict[str, Any]] = None

        stage_instances = [cls(self.client) for cls in STAGE_CLASSES]

        for i, (stage_key, stage_instance) in enumerate(
            zip(STAGE_ORDER, stage_instances)
        ):
            label = STAGE_LABELS.get(stage_key, stage_key)

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  Stage {i + 1}/6: {label}")
                print(f"{'=' * 60}")

            t_start = time.time()
            try:
                output = stage_instance.run(
                    document=doc_text,
                    prior_stage_output=prior_output,
                    model_name=self.model_name,
                )
                stage_outputs[stage_key] = output
                prior_output = {stage_key: output}

                elapsed = time.time() - t_start
                score = output.get("overall_score", "N/A")
                logger.info(
                    "  [%d/6] %s: score=%s (%.1fs)",
                    i + 1, label, score, elapsed,
                )

                if verbose:
                    print(f"  Score: {score}/100")
                    print(f"  Time: {elapsed:.1f}s")
                    findings = self._count_findings(output)
                    print(f"  Findings: {findings}")

            except Exception as e:
                logger.error(
                    "Stage '%s' failed: %s", stage_key, str(e)
                )
                stage_outputs[stage_key] = {
                    "stage": stage_key,
                    "overall_score": 0,
                    "error": str(e),
                    "findings": [],
                }
                if verbose:
                    print(f"  ERROR: {str(e)}")

        # Compute scores
        scoring = self.scorer.aggregate_score(stage_outputs)
        logger.info(
            "Audit complete: composite=%.1f, pass=%s",
            scoring["composite_score"], scoring["pass"],
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Composite Score: {scoring['composite_score']:.1f}/100")
            print(f"  Result: {'PASS' if scoring['pass'] else 'FAIL'}")
            print(f"{'=' * 60}")
            print(self.scorer.summary(stage_outputs))

        # Build reporter and result
        reporter = Reporter(
            document_name=doc_name,
            audit_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        return AuditResult(
            document_name=doc_name,
            document_text=doc_text,
            stage_outputs=stage_outputs,
            scoring=scoring,
            scorer=self.scorer,
            reporter=reporter,
        )

    @staticmethod
    def _count_findings(output: Dict[str, Any]) -> str:
        """Count findings in a stage output for verbose display.

        Args:
            output: Stage output dict.

        Returns:
            A string like "3 findings" or "5 claims".
        """
        for key in ("findings", "claims", "assumptions", "stakeholders"):
            if key in output and isinstance(output[key], list):
                count = len(output[key])
                return f"{count} {key}"
        return "0 findings"

    @staticmethod
    def list_stages() -> List[Dict[str, str]]:
        """List all audit stages with their descriptions.

        Returns:
            List of dicts with 'key', 'label', and 'description'.
        """
        return [
            {
                "key": key,
                "label": STAGE_LABELS[key],
                "description": STAGE_DESCRIPTIONS.get(key, ""),
            }
            for key in STAGE_ORDER
        ]


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    """Command-line entry point for AuditAgent.

    Usage:
        python -m audit_agent.core --input policy.txt --output report.html
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="AuditAgent — 结构化文档审计引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m audit_agent.core --input policy.txt
  python -m audit_agent.core --input policy.txt --output report.html --verbose
  python -m audit_agent.core --input policy.txt --model api --api-key sk-xxx
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to document to audit"
    )
    parser.add_argument(
        "--output", "-o", help="Output path for HTML report"
    )
    parser.add_argument(
        "--model", "-m", default="local",
        choices=["local", "api"], help="LLM backend mode"
    )
    parser.add_argument(
        "--api-base", help="API base URL (for 'api' mode)"
    )
    parser.add_argument(
        "--api-key", help="API key (for 'api' mode)"
    )
    parser.add_argument(
        "--model-name", help="Model name (overrides config)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose stage-by-stage output"
    )
    parser.add_argument(
        "--config", "-c", help="Path to audit_rules.yaml"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Initialize agent
    agent = AuditAgent(
        model=args.model,
        rules_path=args.config,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
    )

    # Run audit
    print(f"\nAuditAgent v1.0.0")
    print(f"Document: {args.input}")
    print(f"Model: {agent.model_name}")
    print()

    result = agent.audit(args.input)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Composite Score: {result.composite_score:.1f}/100")
    print(f"  Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"{'=' * 60}\n")
    print(result.executive_summary())

    # Print red flags
    flags = result.red_flags()
    if flags:
        print(f"\n--- RED FLAGS ({len(flags)}) ---")
        for f in flags[:10]:  # Top 10
            fid = f.get("finding_id", f.get("claim_id", "?"))
            sev = f.get("severity", "?")
            desc = f.get("description", f.get("statement", ""))[:120]
            print(f"  [{sev.upper()}] {fid}: {desc}")

    # Save report
    if args.output:
        result.export_html(args.output)
        print(f"\nReport saved to: {args.output}")
    else:
        # Default output
        default_output = Path(args.input).stem + "_audit_report.html"
        result.export_html(default_output)
        print(f"\nReport saved to: {default_output}")

    # Also export JSON
    json_output = Path(args.input).stem + "_audit_data.json"
    result.export_json(json_output)
    print(f"Data export saved to: {json_output}")


if __name__ == "__main__":
    main()
