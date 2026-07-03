"""
Audit Stages — 6 structured audit stages for the AuditAgent pipeline.

Each stage is an independent audit module with a specific focus, output schema,
and scoring rubric. Stages are designed to be chained together in a fixed order,
with each stage building on the context from previous stages.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from audit_agent.prompts import get_stage_prompt, get_prior_stage_context

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Abstract base class for all audit stages.

    Each stage has:
    - A unique stage_name that maps to its prompt template.
    - A _call method that invokes the LLM with structured context.
    - An optional _parse method to validate/transform LLM output.
    """

    stage_name: str = ""

    def __init__(self, llm_client: Any) -> None:
        """Initialize the stage with an LLM client.

        Args:
            llm_client: An OpenAI-compatible client instance.
        """
        self.llm_client = llm_client

    def run(
        self,
        document: str,
        prior_stage_output: Optional[Dict[str, Any]] = None,
        model_name: str = "qwen2.5:7b",
    ) -> Dict[str, Any]:
        """Execute this audit stage on a document.

        Args:
            document: The full document text to audit.
            prior_stage_output: Structured output from the previous stage,
                used to provide context and avoid duplication.
            model_name: The LLM model to use for this stage.

        Returns:
            Parsed structured output as a dictionary.

        Raises:
            RuntimeError: If the LLM call fails or output cannot be parsed.
        """
        logger.info(
            "Running stage: %s (model=%s)", self.stage_name, model_name
        )

        # Build the prompt
        base_prompt = get_stage_prompt(self.stage_name)
        # Use replace() instead of format() because prompts contain
        # literal JSON with curly braces that conflict with str.format().
        full_prompt = base_prompt.replace("{document}", document)

        # Inject prior stage context if available
        if prior_stage_output is not None:
            prior_label = {
                "coherence_check": "一致性检查 (Coherence Check)",
                "claim_extraction": "主张提取 (Claim Extraction)",
                "assumption_surfacing": "假设浮出 (Assumption Surfacing)",
                "stakeholder_analysis": "利益相关者分析 (Stakeholder Analysis)",
                "methodology_review": "方法论审查 (Methodology Review)",
            }.get(list(prior_stage_output.keys())[0] if len(prior_stage_output) == 1 else "", "")

            if prior_label:
                context = get_prior_stage_context(prior_label, prior_stage_output)
                full_prompt += "\n\n" + context

        # Call the LLM
        try:
            result = self._call(full_prompt, model_name)
        except Exception as e:
            logger.error(
                "LLM call failed for stage %s: %s", self.stage_name, str(e)
            )
            raise RuntimeError(
                f"Stage '{self.stage_name}' LLM call failed: {str(e)}"
            ) from e

        # Parse and validate
        try:
            parsed = self._parse(result)
        except Exception as e:
            logger.error(
                "Parse failed for stage %s: %s\nRaw output: %s",
                self.stage_name,
                str(e),
                result[:500],
            )
            raise RuntimeError(
                f"Stage '{self.stage_name}' output parse failed: {str(e)}"
            ) from e

        logger.info(
            "Stage %s completed. Overall score: %s",
            self.stage_name,
            parsed.get("overall_score", "N/A"),
        )
        return parsed

    @abstractmethod
    def _call(self, prompt: str, model_name: str) -> str:
        """Invoke the LLM with the stage prompt.

        Args:
            prompt: The complete prompt to send to the LLM.
            model_name: The model to use.

        Returns:
            Raw string response from the LLM.
        """
        ...

    def _parse(self, raw_output: str) -> Dict[str, Any]:
        """Parse the LLM's raw output into structured JSON.

        Subclasses can override this for stage-specific validation.

        Args:
            raw_output: Raw text output from the LLM.

        Returns:
            Parsed and validated dictionary.
        """
        # Strip markdown code fences if present
        text = raw_output.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)


class CoherenceCheck(BaseStage):
    """Stage 1: Logical Coherence Check.

    Scans for internal contradictions, circular reasoning, undefined terms,
    inconsistent usage, and logical gaps in the document.
    """

    stage_name = "coherence_check"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class ClaimExtractor(BaseStage):
    """Stage 2: Systematic Claim Extraction.

    Extracts every factual, scientific, statistical, and econometric claim
    from the document, assessing each claim's verifiability and support level.
    """

    stage_name = "claim_extraction"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class AssumptionSurfacer(BaseStage):
    """Stage 3: Implicit Assumption Surfacing.

    Identifies unstated assumptions, missing preconditions, and absent caveats
    that the document's arguments depend on.
    """

    stage_name = "assumption_surfacing"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class StakeholderAnalyzer(BaseStage):
    """Stage 4: Stakeholder Impact Analysis.

    Maps all stakeholders affected by the policy, assesses document coverage,
    and identifies omitted groups or concerns.
    """

    stage_name = "stakeholder_analysis"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class MethodologyReviewer(BaseStage):
    """Stage 5: Methodology and Statistical Rigor Review.

    Evaluates empirical methods, identification strategies, sample adequacy,
    and overall scientific validity of the document's quantitative claims.
    """

    stage_name = "methodology_review"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class BiasDetector(BaseStage):
    """Stage 6: Systematic Bias Detection.

    Detects framing bias, cherry-picking, false equivalency, motivated reasoning,
    linguistic manipulation, and omission bias in the document.
    """

    stage_name = "bias_detection"

    def _call(self, prompt: str, model_name: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


# Ordered list of stage classes for the pipeline
STAGE_CLASSES: List[type] = [
    CoherenceCheck,
    ClaimExtractor,
    AssumptionSurfacer,
    StakeholderAnalyzer,
    MethodologyReviewer,
    BiasDetector,
]


def get_stage_class(stage_index: int) -> type:
    """Get the stage class by its 0-based index in the pipeline.

    Args:
        stage_index: 0-based index (0 = CoherenceCheck, 5 = BiasDetector).

    Returns:
        The stage class.

    Raises:
        IndexError: If stage_index is out of range.
    """
    if stage_index < 0 or stage_index >= len(STAGE_CLASSES):
        raise IndexError(
            f"Stage index {stage_index} out of range. "
            f"Valid range: 0-{len(STAGE_CLASSES) - 1}"
        )
    return STAGE_CLASSES[stage_index]
