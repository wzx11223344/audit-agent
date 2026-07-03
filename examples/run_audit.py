#!/usr/bin/env python3
"""
Demo script: Run a complete audit on the sample carbon tax policy document.

This script demonstrates the full AuditAgent workflow:
    1. Load the sample policy document
    2. Run the 6-stage audit pipeline
    3. Print the executive summary
    4. Save structured reports (HTML + JSON) to the output directory

Usage:
    python examples/run_audit.py                    # Local Ollama
    python examples/run_audit.py --model api --api-key sk-xxx  # OpenAI API
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from audit_agent import AuditAgent
from audit_agent.prompts import STAGE_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AuditAgent demo on sample policy document.",
    )
    parser.add_argument(
        "--model", "-m", default="local",
        choices=["local", "api"],
        help="LLM backend: 'local' (Ollama) or 'api' (OpenAI compatible)"
    )
    parser.add_argument(
        "--api-base",
        help="API base URL (default: http://localhost:11434/v1 for local, "
             "https://api.openai.com/v1 for api)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (not needed for local Ollama)"
    )
    parser.add_argument(
        "--model-name",
        help="Model name (default: qwen2.5:7b for local, gpt-4o for api)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose stage-by-stage output"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for reports (default: examples/output)"
    )

    args = parser.parse_args()

    # Resolve paths
    examples_dir = Path(__file__).resolve().parent
    sample_path = examples_dir / "sample_policy.txt"
    output_dir = args.output_dir or str(examples_dir / "output")

    if not sample_path.exists():
        print(f"Error: Sample document not found at {sample_path}")
        sys.exit(1)

    print("=" * 60)
    print("  AuditAgent — Demo Run")
    print("=" * 60)
    print(f"  Document: {sample_path}")
    print(f"  Model mode: {args.model}")
    print(f"  Output dir: {output_dir}")
    print()

    # Initialize agent
    agent = AuditAgent(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
    )

    print(f"  Backend: {agent.model_name}")
    print(f"  Base URL: {agent.client.base_url}")
    print()

    # Run audit
    if args.verbose:
        result = agent.stage_by_stage(str(sample_path), verbose=True)
    else:
        print("Running audit pipeline (6 stages)...")
        result = agent.audit(str(sample_path))

    # === Print Results ===

    print()
    print("=" * 60)
    print("  AUDIT RESULTS")
    print("=" * 60)
    print(f"  Composite Score: {result.composite_score:.1f}/100")
    print(f"  Result: {'PASS' if result.passed else 'FAIL'}")
    print()

    # Executive summary
    print("--- EXECUTIVE SUMMARY ---")
    print(result.executive_summary())
    print()

    # Per-stage scores
    print("--- PER-STAGE SCORES ---")
    per_stage = result.scoring.get("per_stage_scores", {})
    traffic = result.traffic_lights()
    for stage_key, label in STAGE_LABELS.items():
        score = per_stage.get(stage_key, "N/A")
        light = traffic.get(stage_key, "green")
        emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(light, "")
        print(f"  {emoji} {label}: {score}/100")
    print()

    # Red flags
    flags = result.red_flags()
    if flags:
        print(f"--- RED FLAGS ({len(flags)} critical/high findings) ---")
        for f in flags[:10]:  # Show top 10
            fid = f.get("finding_id") or f.get("claim_id") or f.get("assumption_id") or "?"
            stage = f.get("_stage_label", "")
            sev = f.get("severity", "?")
            desc = (
                f.get("description")
                or f.get("statement")
                or f.get("assumption")
                or ""
            )[:150]
            print(f"  [{sev.upper()}] [{stage}] {fid}: {desc}")
        if len(flags) > 10:
            print(f"  ... and {len(flags) - 10} more")
    else:
        print("  No critical or high severity findings.")
    print()

    # Action items
    actions = result.action_items()
    if actions:
        print(f"--- ACTION ITEMS (top {min(5, len(actions))}) ---")
        for item in actions[:5]:
            print(f"  [{item['severity'].upper()}] [{item['stage']}] {item['issue'][:100]}")
        print()

    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "audit_report.html")
    json_path = os.path.join(output_dir, "audit_data.json")
    txt_path = os.path.join(output_dir, "audit_report.txt")

    result.export_html(html_path)
    result.export_json(json_path)
    result.export_text_report(txt_path)

    print("--- OUTPUT FILES ---")
    print(f"  HTML Report:  {html_path}")
    print(f"  JSON Data:    {json_path}")
    print(f"  Text Report:  {txt_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
