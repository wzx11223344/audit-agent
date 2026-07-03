"""
AuditAgent — 结构化多阶段文档审计引擎

A structured multi-stage document audit engine powered by LLMs.
Designed for rigorous audit of policy documents, research reports,
and institutional texts. Not a generic chatbot.

Usage:
    from audit_agent import AuditAgent

    agent = AuditAgent(model="local")
    result = agent.audit("path/to/document.txt")
    print(result.executive_summary())
    result.export_html("output/report.html")
"""

__version__ = "1.0.0"
__author__ = "AuditAgent Team"

from audit_agent.core import AuditAgent, AuditResult
from audit_agent.stages import (
    CoherenceCheck,
    ClaimExtractor,
    AssumptionSurfacer,
    StakeholderAnalyzer,
    MethodologyReviewer,
    BiasDetector,
)
from audit_agent.scoring import AuditScorer
from audit_agent.reporter import Reporter

__all__ = [
    "AuditAgent",
    "AuditResult",
    "CoherenceCheck",
    "ClaimExtractor",
    "AssumptionSurfacer",
    "StakeholderAnalyzer",
    "MethodologyReviewer",
    "BiasDetector",
    "AuditScorer",
    "Reporter",
]
