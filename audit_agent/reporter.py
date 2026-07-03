"""
Structured Report Generator for AuditAgent.

Generates audit reports in multiple formats: structured text, executive summary,
action items, self-contained HTML, and machine-readable JSON.

Design Goals:
    - Self-contained HTML with inline CSS (no external dependencies).
    - Color-coded severity indicators.
    - Traffic-light visual dashboard.
    - Export options for integration with other tools.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from audit_agent.prompts import STAGE_LABELS, STAGE_DESCRIPTIONS


# CSS color scheme matching config/audit_rules.yaml
COLOR_SCHEME: Dict[str, str] = {
    "critical": "#DC3545",
    "high": "#FD7E14",
    "medium": "#FFC107",
    "low": "#198754",
    "info": "#6C757D",
    "green": "#28A745",
    "yellow": "#FFC107",
    "red": "#DC3545",
}

# Traffic light emoji
TRAFFIC_EMOJI: Dict[str, str] = {
    "green": "\U0001f7e2",   # green circle
    "yellow": "\U0001f7e1",  # yellow circle
    "red": "\U0001f534",     # red circle
}


class Reporter:
    """Generates structured audit reports in multiple formats.

    The Reporter takes the complete audit result (stage outputs + scoring)
    and produces human-readable and machine-readable reports.
    """

    def __init__(
        self,
        document_name: str = "Untitled Document",
        audit_timestamp: Optional[str] = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            document_name: Name/title of the audited document.
            audit_timestamp: ISO-format timestamp (default: now).
        """
        self.document_name = document_name
        self.audit_timestamp = audit_timestamp or datetime.now().isoformat()

    def generate_report(
        self,
        audit_result: Dict[str, Any],
        scorer: Any = None,
    ) -> str:
        """Generate a complete structured text report.

        Args:
            audit_result: Complete audit data with stage_outputs and scoring.
            scorer: The AuditScorer instance used (for traffic lights, etc.).

        Returns:
            Formatted text report string.
        """
        score_data = audit_result.get("scoring", {})
        stage_outputs = audit_result.get("stage_outputs", {})
        composite = score_data.get("composite_score", "N/A")
        passed = score_data.get("pass", False)

        lines = []
        lines.append("=" * 72)
        lines.append("  AuditAgent — 结构化文档审计报告")
        lines.append("=" * 72)
        lines.append(f"  文档: {self.document_name}")
        lines.append(f"  审计时间: {self.audit_timestamp}")
        lines.append(f"  综合评分: {composite}/100")
        lines.append(f"  审计结果: {'通过 (PASS)' if passed else '未通过 (FAIL)'}")
        lines.append("=" * 72)
        lines.append("")

        # Per-stage scores
        lines.append("-" * 72)
        lines.append("  各阶段评分")
        lines.append("-" * 72)
        per_stage = score_data.get("per_stage_scores", {})
        weighted = score_data.get("weighted_contributions", {})

        if scorer:
            traffic = scorer.traffic_light_map(stage_outputs)
        else:
            traffic = {}

        for stage_key in STAGE_LABELS:
            label = STAGE_LABELS[stage_key]
            score = per_stage.get(stage_key, "N/A")
            weight_contrib = weighted.get(stage_key, "N/A")
            light = traffic.get(stage_key, "green")
            light_emoji = TRAFFIC_EMOJI.get(light, "")
            lines.append(
                f"  {light_emoji} {label} ({stage_key}): "
                f"{score}/100 (加权贡献: {weight_contrib})"
            )
        lines.append("")

        # Severity distribution
        if scorer:
            dist = scorer.severity_distribution(stage_outputs)
            lines.append("-" * 72)
            lines.append("  问题严重度分布")
            lines.append("-" * 72)
            lines.append(
                f"  严重 (Critical): {dist.get('critical', 0)}  |  "
                f"高危 (High): {dist.get('high', 0)}  |  "
                f"中等 (Medium): {dist.get('medium', 0)}  |  "
                f"低危 (Low): {dist.get('low', 0)}  |  "
                f"信息 (Info): {dist.get('info', 0)}"
            )
            lines.append("")

        # Stage-by-stage detailed findings
        for stage_key in STAGE_LABELS:
            label = STAGE_LABELS[stage_key]
            desc = STAGE_DESCRIPTIONS.get(stage_key, "")
            lines.append("-" * 72)
            lines.append(f"  {label} — {desc}")
            lines.append("-" * 72)

            output = stage_outputs.get(stage_key, {})

            # Extract findings list
            findings = self._get_findings(output)
            if not findings:
                lines.append("  未发现明显问题。")
                lines.append("")
                continue

            for i, finding in enumerate(findings, 1):
                fid = finding.get(
                    "finding_id", finding.get("claim_id",
                    finding.get("assumption_id", finding.get("stakeholder_id",
                    finding.get("method_id", f"ITEM-{i}"))))
                )
                severity = finding.get("severity", "medium")
                sev_score = finding.get("severity_score", 50)
                lines.append(f"  [{severity.upper()}] {fid} (severity={sev_score})")

                # Description or statement
                desc_text = (
                    finding.get("description")
                    or finding.get("statement")
                    or finding.get("assumption")
                    or finding.get("issue_type", "")
                )
                lines.append(f"    {desc_text}")

                # Location
                location = finding.get("location", "")
                if location:
                    lines.append(f"    位置: {location}")

                # Evidence
                evidence = finding.get("evidence", "")
                if evidence:
                    lines.append(f'    原文: "{evidence}"')

                # Suggested fix or alternative
                fix = finding.get("suggested_fix") or finding.get(
                    "suggested_alternative") or finding.get(
                    "reframing_suggestion") or finding.get(
                    "suggested_qualification")
                if fix:
                    lines.append(f"    建议: {fix}")

                # Concerns list (for methodology)
                concerns = finding.get("concerns", [])
                if isinstance(concerns, list):
                    for c in concerns:
                        if isinstance(c, dict):
                            c_sev = c.get("severity", "medium")
                            c_desc = c.get("description", str(c))
                            lines.append(f"    [{c_sev.upper()}] {c_desc}")

                lines.append("")

        # Summary
        lines.append("=" * 72)
        lines.append("  审计结束")
        lines.append("=" * 72)

        return "\n".join(lines)

    def executive_summary(
        self,
        audit_result: Dict[str, Any],
        max_length: int = 500,
    ) -> str:
        """Generate a 1-page executive summary for busy readers.

        Args:
            audit_result: Complete audit data.
            max_length: Maximum characters for the summary.

        Returns:
            Concise executive summary string.
        """
        score_data = audit_result.get("scoring", {})
        stage_outputs = audit_result.get("stage_outputs", {})
        composite = score_data.get("composite_score", "N/A")
        passed = score_data.get("pass", False)

        parts = []
        parts.append(
            f"文档 '{self.document_name}' 审计综合得分为 {composite}/100，"
            f"审计结果: {'通过' if passed else '未通过'}。"
        )

        # Count critical and high issues
        critical_count = 0
        high_count = 0
        for output in stage_outputs.values():
            for finding in self._get_findings(output):
                sev = finding.get("severity", "")
                sev_score = finding.get("severity_score", 50)
                if sev == "critical" or sev_score >= 85:
                    critical_count += 1
                elif sev == "high" or sev_score >= 65:
                    high_count += 1

        if critical_count or high_count:
            parts.append(
                f"共发现 {critical_count} 个严重问题和 {high_count} 个高危问题。"
            )
        else:
            parts.append("未发现严重或高危问题。")

        # Per-stage highlights
        stage_highlights = []
        for stage_key in STAGE_LABELS:
            score = score_data.get("per_stage_scores", {}).get(stage_key)
            if score is not None:
                label = STAGE_LABELS[stage_key]
                if score < 50:
                    stage_highlights.append(f"{label}({score}/100)")
        if stage_highlights:
            parts.append(
                f"需要重点关注的领域: {', '.join(stage_highlights)}。"
            )

        summary = "".join(parts)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def action_items(
        self, audit_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract actionable improvements from audit findings.

        Args:
            audit_result: Complete audit data.

        Returns:
            List of dicts with 'stage', 'finding_id', 'severity',
            'issue', and 'recommendation' keys.
        """
        items: List[Dict[str, str]] = []
        stage_outputs = audit_result.get("stage_outputs", {})

        for stage_key, output in stage_outputs.items():
            findings = self._get_findings(output)
            for finding in findings:
                sev_score = finding.get("severity_score", 50)
                if sev_score < 40:
                    continue  # Skip low-severity for action items

                fid = finding.get(
                    "finding_id", finding.get("claim_id",
                    finding.get("assumption_id", finding.get("stakeholder_id",
                    finding.get("method_id", "ITEM"))))
                )
                desc = (
                    finding.get("description")
                    or finding.get("statement")
                    or finding.get("assumption")
                    or ""
                )
                fix = (
                    finding.get("suggested_fix")
                    or finding.get("suggested_alternative")
                    or finding.get("reframing_suggestion")
                    or finding.get("suggested_qualification")
                    or ""
                )

                items.append({
                    "stage": STAGE_LABELS.get(stage_key, stage_key),
                    "finding_id": fid,
                    "severity": finding.get("severity", "medium"),
                    "issue": desc,
                    "recommendation": fix,
                })

        # Sort by severity_score if available
        items.sort(
            key=lambda x: (
                0 if x["severity"] == "critical"
                else 1 if x["severity"] == "high"
                else 2
            )
        )

        return items

    def generate_html(
        self,
        audit_result: Dict[str, Any],
        scorer: Any = None,
    ) -> str:
        """Generate a self-contained HTML audit report with color-coded sections.

        Args:
            audit_result: Complete audit data.
            scorer: The AuditScorer instance.

        Returns:
            Complete HTML string with inline CSS.
        """
        score_data = audit_result.get("scoring", {})
        stage_outputs = audit_result.get("stage_outputs", {})
        composite = score_data.get("composite_score", "N/A")
        passed = score_data.get("pass", False)

        # Traffic lights
        if scorer:
            traffic = scorer.traffic_light_map(stage_outputs)
            dist = scorer.severity_distribution(stage_outputs)
        else:
            traffic = {}
            dist = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

        # Build stage sections
        stage_html = ""
        for stage_key in STAGE_LABELS:
            label = STAGE_LABELS[stage_key]
            desc = STAGE_DESCRIPTIONS.get(stage_key, "")
            light = traffic.get(stage_key, "green")
            light_color = COLOR_SCHEME.get(light, "#6C757D")
            score = score_data.get("per_stage_scores", {}).get(stage_key, "N/A")
            light_emoji = TRAFFIC_EMOJI.get(light, "")

            output = stage_outputs.get(stage_key, {})
            findings = self._get_findings(output)

            findings_html = ""
            if findings:
                for finding in findings:
                    fid = finding.get("finding_id", finding.get("claim_id",
                        finding.get("assumption_id", finding.get("stakeholder_id",
                        finding.get("method_id", "ITEM")))))
                    sev = finding.get("severity", "medium")
                    sev_color = COLOR_SCHEME.get(sev, COLOR_SCHEME["medium"])
                    sev_score = finding.get("severity_score", 50)

                    desc_text = (
                        finding.get("description")
                        or finding.get("statement")
                        or finding.get("assumption")
                        or ""
                    )
                    location = finding.get("location", "")
                    evidence = finding.get("evidence", "")
                    fix = (
                        finding.get("suggested_fix")
                        or finding.get("suggested_alternative")
                        or finding.get("reframing_suggestion")
                        or finding.get("suggested_qualification")
                        or ""
                    )

                    findings_html += f"""
                    <div class="finding" style="border-left: 4px solid {sev_color};">
                        <div class="finding-header">
                            <span class="badge" style="background: {sev_color};">{sev.upper()}</span>
                            <strong>{fid}</strong>
                            <span class="sev-score">Severity: {sev_score}</span>
                        </div>
                        <p class="finding-desc">{self._escape_html(desc_text)}</p>
                        {f'<p class="finding-location"><strong>Location:</strong> {self._escape_html(location)}</p>' if location else ''}
                        {f'<blockquote class="finding-evidence">{self._escape_html(evidence)}</blockquote>' if evidence else ''}
                        {f'<p class="finding-fix"><strong>Recommendation:</strong> {self._escape_html(fix)}</p>' if fix else ''}
                    </div>"""
            else:
                findings_html = '<p class="no-findings">No significant issues found.</p>'

            stage_html += f"""
            <div class="stage-section">
                <h3 style="color: {light_color};">
                    {light_emoji} {label}
                    <span class="stage-score">Score: {score}/100</span>
                </h3>
                <p class="stage-desc">{desc}</p>
                {findings_html}
            </div>"""

        # Severity summary bars
        total_issues = sum(dist.values()) or 1
        sev_bars = ""
        for sev_name in ["critical", "high", "medium", "low", "info"]:
            count = dist.get(sev_name, 0)
            pct = count / total_issues * 100
            color = COLOR_SCHEME.get(sev_name, "#6C757D")
            sev_bars += f"""
            <div class="sev-bar-item">
                <span class="sev-label">{sev_name.capitalize()}</span>
                <div class="sev-bar-track">
                    <div class="sev-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
                </div>
                <span class="sev-count">{count}</span>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AuditAgent Report — {self._escape_html(self.document_name)}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 960px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); overflow: hidden; }}
    .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff; padding: 48px 40px; }}
    .header h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
    .header .meta {{ opacity: 0.8; font-size: 14px; }}
    .score-banner {{ display: flex; align-items: center; gap: 24px; margin-top: 24px; }}
    .score-circle {{ width: 100px; height: 100px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: 700; }}
    .score-circle.pass {{ background: rgba(40,167,69,0.2); color: #28a745; border: 3px solid #28a745; }}
    .score-circle.fail {{ background: rgba(220,53,69,0.2); color: #dc3545; border: 3px solid #dc3545; }}
    .score-circle .score-val {{ font-size: 36px; }}
    .score-circle .score-label {{ font-size: 12px; text-transform: uppercase; }}
    .pass-fail {{ font-size: 20px; font-weight: 600; padding: 8px 20px; border-radius: 8px; }}
    .pass-fail.pass {{ background: rgba(40,167,69,0.15); color: #28a745; }}
    .pass-fail.fail {{ background: rgba(220,53,69,0.15); color: #dc3545; }}

    .content {{ padding: 40px; }}
    .section-title {{ font-size: 22px; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #eee; }}

    .sev-summary {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 32px; }}
    .sev-bar-item {{ display: flex; align-items: center; gap: 12px; }}
    .sev-label {{ width: 80px; font-size: 13px; font-weight: 600; text-transform: uppercase; }}
    .sev-bar-track {{ flex: 1; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
    .sev-bar-fill {{ height: 100%; border-radius: 10px; transition: width 0.3s; }}
    .sev-count {{ width: 40px; text-align: right; font-weight: 600; }}

    .stage-section {{ margin-bottom: 32px; padding: 24px; background: #fafafa; border-radius: 8px; border: 1px solid #eee; }}
    .stage-section h3 {{ font-size: 18px; margin-bottom: 4px; }}
    .stage-score {{ font-size: 14px; font-weight: normal; opacity: 0.7; margin-left: 12px; }}
    .stage-desc {{ font-size: 13px; color: #666; margin-bottom: 16px; }}

    .finding {{ margin-bottom: 16px; padding: 16px 20px; background: #fff; border-radius: 6px; }}
    .finding-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .badge {{ color: #fff; padding: 2px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
    .sev-score {{ font-size: 12px; color: #999; margin-left: auto; }}
    .finding-desc {{ font-size: 14px; margin-bottom: 8px; }}
    .finding-location {{ font-size: 12px; color: #888; }}
    .finding-evidence {{ margin: 8px 0; padding: 8px 16px; border-left: 3px solid #ddd; background: #f9f9f9; font-style: italic; font-size: 13px; color: #555; }}
    .finding-fix {{ font-size: 13px; color: #0d6efd; }}
    .no-findings {{ color: #28a745; font-style: italic; }}

    .footer {{ padding: 24px 40px; background: #fafafa; border-top: 1px solid #eee; font-size: 12px; color: #999; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>AuditAgent Report</h1>
        <div class="meta">Document: {self._escape_html(self.document_name)} | {self.audit_timestamp}</div>
        <div class="score-banner">
            <div class="score-circle {'pass' if passed else 'fail'}">
                <span class="score-val">{composite}</span>
                <span class="score-label">/100</span>
            </div>
            <div class="pass-fail {'pass' if passed else 'fail'}">
                {'PASS' if passed else 'FAIL'}
            </div>
        </div>
    </div>

    <div class="content">
        <h2 class="section-title">Severity Distribution</h2>
        <div class="sev-summary">
            {sev_bars}
        </div>

        <h2 class="section-title">Stage-by-Stage Findings</h2>
        {stage_html}
    </div>

    <div class="footer">
        Generated by AuditAgent v1.0.0 | Structured Document Audit Engine
    </div>
</div>
</body>
</html>"""

        return html

    def export_json(self, audit_result: Dict[str, Any]) -> str:
        """Export the complete audit result as formatted JSON.

        Args:
            audit_result: Complete audit data.

        Returns:
            Formatted JSON string.
        """
        export_data = {
            "audit_agent_version": "1.0.0",
            "document_name": self.document_name,
            "audit_timestamp": self.audit_timestamp,
            "scoring": audit_result.get("scoring", {}),
            "stage_outputs": audit_result.get("stage_outputs", {}),
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def save_text_report(
        self,
        audit_result: Dict[str, Any],
        output_path: str,
        scorer: Any = None,
    ) -> str:
        """Save the text report to a file.

        Args:
            audit_result: Complete audit data.
            output_path: Path to save the text report.
            scorer: The AuditScorer instance.

        Returns:
            The output path.
        """
        report = self.generate_report(audit_result, scorer)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return output_path

    def save_html_report(
        self,
        audit_result: Dict[str, Any],
        output_path: str,
        scorer: Any = None,
    ) -> str:
        """Save the HTML report to a file.

        Args:
            audit_result: Complete audit data.
            output_path: Path to save the HTML report.
            scorer: The AuditScorer instance.

        Returns:
            The output path.
        """
        html = self.generate_html(audit_result, scorer)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def save_json_export(
        self,
        audit_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Save the JSON export to a file.

        Args:
            audit_result: Complete audit data.
            output_path: Path to save the JSON file.

        Returns:
            The output path.
        """
        json_str = self.export_json(audit_result)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_findings(output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract findings list from a stage output dict."""
        for key in ("findings", "claims", "assumptions", "stakeholders",
                     "methods_identified"):
            if key in output and isinstance(output[key], list):
                return output[key]
        return []

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
