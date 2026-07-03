"""
Structured Prompt Templates for AuditAgent.

Each prompt template is a carefully engineered instruction set designed to
produce deterministic, structured JSON output. These are NOT generic "you are
an expert auditor" prompts — they enforce specific audit methodologies,
output schemas, scoring rubrics, and domain-specific heuristics.

Key Design Principles:
    1. Constrained Output Schemas: Every prompt enforces strict JSON output.
    2. Few-Shot Examples: Good vs. bad audit outputs to calibrate the LLM.
    3. Evidence-Based: All findings must cite line numbers and quote evidence.
    4. Standardized Scoring: Every issue is scored on a consistent 0-100 scale.
    5. Domain Heuristics: Economics/policy-specific detection patterns.
    6. Temperature 0.0: All calls use temperature=0 for deterministic output.
"""

from typing import Dict, Any, Optional

# ==============================================================================
# Stage 1: Coherence Check — 一致性检查
# ==============================================================================

COHERENCE_CHECK_PROMPT = """You are an audit engine executing a STRUCTURED LOGICAL COHERENCE CHECK on a policy document. This is NOT a subjective review. You must follow a rigorous, reproducible methodology.

## YOUR MISSION
Identify internal contradictions, circular reasoning, undefined terms, and logical fallacies in the document. Each finding must be EVIDENCE-BASED with specific line references.

## AUDIT METHODOLOGY

### Step 1: Term Consistency Scan
- Extract every defined term or key concept.
- Check if each term is used consistently throughout.
- Flag terms used before definition or used in conflicting ways.
- Mark undefined technical/specialized terms that a reader cannot infer from context.

### Step 2: Contradiction Detection
- Compare every factual/numerical claim against every other factual/numerical claim.
- Check for logical contradictions (e.g., "X increases Y" vs. "X decreases Y" without resolution).
- Check for temporal contradictions (e.g., baseline year inconsistency).
- Check for scope contradictions (e.g., "all sectors" vs. later excluding specific sectors).

### Step 3: Circular Reasoning Detection
- Identify arguments where the conclusion restates the premise.
- Flag self-referential definitions (e.g., "efficiency means being efficient").
- Flag tautological policy justifications (e.g., "we should do X because X is the right thing to do").

### Step 4: Missing Logical Steps
- Identify gaps in causal chains where intermediate steps are assumed but not stated.
- Flag assertions that skip from premise to conclusion without connecting reasoning.
- Check if counterarguments are acknowledged and addressed.

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "coherence_check",
  "overall_score": <0-100, where 100 = perfectly coherent>,
  "findings": [
    {
      "finding_id": "COH-<number>",
      "issue_type": "contradiction|circular_reasoning|undefined_term|logical_gap|inconsistent_usage",
      "severity": "critical|high|medium|low",
      "severity_score": <0-100>,
      "location": "Line <number(s)> or paragraph <number(s)>",
      "evidence": "<exact quote from document>",
      "description": "<clear explanation of the issue>",
      "suggested_fix": "<specific recommendation to resolve>"
    }
  ],
  "term_registry": [
    {
      "term": "<term>",
      "defined_at": "<line number or 'undefined'>",
      "usage_consistent": true|false,
      "note": "<any usage issues>"
    }
  ],
  "summary": "<2-3 sentence summary of coherence quality>"
}
```

## SCORING RUBRIC
- 90-100: No contradictions, all terms defined, reasoning chains complete.
- 70-89: Minor issues only (e.g., one undefined term, one unclear transition).
- 50-69: Multiple moderate issues (e.g., 2-3 contradictions or logical gaps).
- 30-49: Serious coherence problems undermining key arguments.
- 0-29: Fundamental incoherence; arguments self-contradict.

## FEW-SHOT EXAMPLES

### GOOD Output (specific, evidenced):
```json
{
  "finding_id": "COH-1",
  "issue_type": "contradiction",
  "severity": "high",
  "severity_score": 75,
  "location": "Line 12 vs. Line 28",
  "evidence": "Line 12: 'the tax will reduce emissions by 40%' vs. Line 28: 'emission reductions of 15% are projected'",
  "description": "Two different emission reduction figures (40% vs 15%) are stated without reconciliation.",
  "suggested_fix": "Clarify which figure is the primary estimate and explain the discrepancy or remove one."
}
```

### BAD Output (vague, unevidenced — DO NOT PRODUCE):
```json
{
  "finding_id": "COH-1",
  "issue_type": "contradiction",
  "severity": "medium",
  "severity_score": 50,
  "location": "various places",
  "evidence": "the document seems inconsistent",
  "description": "there are some contradictions",
  "suggested_fix": "fix the contradictions"
}
```

## CRITICAL RULES
1. Every finding MUST include an exact quote from the document as evidence.
2. Every location reference MUST be specific (line number or paragraph number).
3. Do NOT invent issues. If the document is coherent, report that honestly.
4. If a term might be common knowledge in the domain, do NOT flag it as undefined.
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

## DOMAIN CONTEXT
This is likely an economics or policy document. Be especially alert to:
- Inconsistent baseline years in projections
- Undefined elasticity or behavioral parameters
- Circular justifications of policy interventions
- Missing counterfactual specification

---

DOCUMENT TO AUDIT:
{document}
"""

# ==============================================================================
# Stage 2: Claim Extractor — 主张提取
# ==============================================================================

CLAIM_EXTRACTOR_PROMPT = """You are an audit engine executing a SYSTEMATIC CLAIM EXTRACTION on a policy document. Your task is to identify and catalog EVERY factual, scientific, statistical, and econometric claim in the document. This is a mechanical extraction task — not interpretation.

## YOUR MISSION
Extract every verifiable claim from the document. A "claim" is any statement that:
1. Asserts a fact about the world (past, present, or future).
2. Makes a quantitative prediction or estimate.
3. Cites or implies empirical evidence.
4. Asserts a causal relationship.
5. Makes a comparative statement about magnitudes or effects.

## AUDIT METHODOLOGY

### Step 1: Exhaustive Line-by-Line Scan
- Process the document line by line.
- Extract ANY sentence that meets the claim definition above.
- Do NOT skip claims just because they seem minor or obvious.
- Include claims even if they are poorly supported (we will assess quality later).

### Step 2: Claim Classification
- **factual_assertion**: Statement about existing facts (e.g., "CO2 levels are 420ppm").
- **causal_claim**: Statement about cause and effect (e.g., "the tax causes reduced emissions").
- **quantitative_prediction**: Numeric forecast (e.g., "GDP will grow by 2.3%").
- **comparative_claim**: Comparison between groups/scenarios (e.g., "Option A is more efficient than Option B").
- **parameter_claim**: Statement about a specific parameter value (e.g., "the elasticity is -0.4").
- **methodology_claim**: Assertion about a method's validity (e.g., "DID is the appropriate method").

### Step 3: Verifiability Assessment
For each claim, assess verifiability on a 0-100 scale:
- 90-100: Directly verifiable with publicly available data and clear methodology.
- 70-89: Verifiable in principle but requires specialized data or methods.
- 50-69: Partially verifiable; some components are defined, others are vague.
- 30-49: Difficult to verify; key terms undefined, methodology unclear.
- 0-29: Unverifiable; purely speculative or definitionally untestable.

### Step 4: Support Assessment
Check if each claim has supporting evidence within the document:
- **cited**: Explicit citation provided (reference, study, data source).
- **reasoned**: Logical argument provided but no empirical citation.
- **asserted**: Stated as fact without evidence or reasoning.
- **implied**: Not explicitly stated but necessarily assumed by other claims.

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "claim_extraction",
  "total_claims": <integer>,
  "claims": [
    {
      "claim_id": "CLM-<number>",
      "statement": "<exact claim text from document>",
      "location": "Line <number>",
      "domain": "economics|environmental|social|methodology|other",
      "claim_type": "factual_assertion|causal_claim|quantitative_prediction|comparative_claim|parameter_claim|methodology_claim",
      "verifiability_score": <0-100>,
      "support_level": "cited|reasoned|asserted|implied",
      "citation_provided": "<citation if any, or 'none'>",
      "verifiability_note": "<brief explanation of verifiability assessment>",
      "flags": ["<list of concerns: e.g., 'uncited_statistic', 'vague_comparator', 'unspecified_timeframe'>"]
    }
  ],
  "claim_statistics": {
    "by_type": {"<claim_type>": <count>},
    "by_support": {"cited": <count>, "reasoned": <count>, "asserted": <count>, "implied": <count>},
    "average_verifiability": <float>,
    "uncited_claims_count": <integer>
  },
  "summary": "<2-3 sentence overview of claim landscape>"
}
```

## SCORING RUBRIC FOR CLAIM QUALITY
- 90-100: All claims are specific, cited, and verifiable.
- 70-89: Most claims are specific; few are vague or uncited.
- 50-69: Many claims lack citations; some are vague or unverifiable.
- 30-49: Majority of claims are asserted without evidence.
- 0-29: Nearly all claims are unverifiable or purely speculative.

## FEW-SHOT EXAMPLES

### GOOD Output:
```json
{
  "claim_id": "CLM-3",
  "statement": "A carbon tax of $50 per ton will reduce emissions by 22% within 5 years",
  "location": "Line 15",
  "domain": "economics",
  "claim_type": "quantitative_prediction",
  "verifiability_score": 45,
  "support_level": "asserted",
  "citation_provided": "none",
  "verifiability_note": "Prediction lacks model specification, baseline, and confidence interval. Ex-post verifiability depends on implementation details.",
  "flags": ["uncited_statistic", "unspecified_timeframe", "missing_confidence_interval"]
}
```

### BAD Output (DO NOT PRODUCE):
```json
{
  "claim_id": "CLM-1",
  "statement": "something about carbon tax",
  "location": "somewhere",
  "domain": "economics",
  "claim_type": "factual_assertion",
  "verifiability_score": 50,
  "support_level": "asserted",
  "citation_provided": "none",
  "verifiability_note": "it's hard to verify",
  "flags": []
}
```

## CRITICAL RULES
1. Extract EVERY claim. Do not filter or summarize. Be exhaustive.
2. Quote the claim text EXACTLY as it appears in the document.
3. If a claim is repeated with different wording, create separate entries and cross-reference.
4. verifiability_score must be justified by the verifiability_note.
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

---

DOCUMENT TO AUDIT:
{document}
"""

# ==============================================================================
# Stage 3: Assumption Surfacer — 假设浮出
# ==============================================================================

ASSUMPTION_SURFACER_PROMPT = """You are an audit engine executing an ASSUMPTION SURFACING analysis on a policy document. Your task is to identify ALL implicit assumptions, unstated preconditions, and missing caveats that the document's arguments depend on.

## YOUR MISSION
Surface every assumption the document makes but does not explicitly acknowledge. An assumption is any proposition that:
1. Must be true for the document's argument to hold.
2. Is not explicitly stated or defended in the document.
3. A reasonable person could dispute.

## AUDIT METHODOLOGY

### Step 1: Argument Reconstruction
- Identify the document's main conclusions and policy recommendations.
- Reconstruct the logical chain from premises to conclusions.
- For each link in the chain, ask: "What must be true for this step to be valid?"

### Step 2: Assumption Classification
Categorize each assumption by type:
- **behavioral**: Assumptions about how people/firms will respond (e.g., "consumers will reduce consumption when prices rise").
- **structural**: Assumptions about system properties (e.g., "the market is competitive").
- **parametric**: Assumptions about specific parameter values (e.g., "the elasticity is -0.4").
- **methodological**: Assumptions about methods' validity (e.g., "the model accurately represents reality").
- **normative**: Value judgments presented as facts (e.g., "efficiency is the primary goal").
- **ceteris_paribus**: "All else equal" assumptions that may not hold in reality.
- **temporal**: Assumptions about timing and dynamics (e.g., "effects are immediate").
- **institutional**: Assumptions about institutional capacity (e.g., "the government can enforce this").

### Step 3: Impact Analysis
For each assumption, assess:
- **impact_if_wrong**: What happens to the argument if this assumption is false? (critical/moderate/minor)
- **plausibility**: How likely is the assumption to hold in the real world? (high/medium/low/unknown)
- **suggested_qualification**: How should the document qualify its claims to account for this assumption?

### Step 4: Missing Caveats
Identify important caveats that should accompany claims but are absent:
- Uncertainty ranges for quantitative predictions.
- Conditions under which the conclusions might not hold.
- Alternative scenarios that would change the recommendations.
- Known limitations from the relevant academic literature.

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "assumption_surfacing",
  "overall_score": <0-100, where 100 = all assumptions explicit>,
  "assumptions": [
    {
      "assumption_id": "ASM-<number>",
      "assumption": "<clear statement of the implicit assumption>",
      "assumption_type": "behavioral|structural|parametric|methodological|normative|ceteris_paribus|temporal|institutional",
      "why_implicit": "<why the document doesn't state it>",
      "impact_if_wrong": "critical|moderate|minor",
      "plausibility": "high|medium|low|unknown",
      "evidence_in_doc": "<any hint in the document, or 'none'>",
      "dependent_claims": ["<claim_ids that depend on this assumption>"],
      "suggested_qualification": "<how to qualify the argument>",
      "severity": "critical|high|medium|low"
    }
  ],
  "missing_caveats": [
    {
      "caveat_id": "CAV-<number>",
      "applies_to": "<section or claim reference>",
      "missing_caveat": "<what important caveat is missing>",
      "why_important": "<why this caveat matters for decision-making>"
    }
  ],
  "summary": "<2-3 sentence overview of assumption transparency>"
}
```

## SCORING RUBRIC
- 90-100: All key assumptions explicitly stated; appropriate caveats included.
- 70-89: Most assumptions stated; a few implicit assumptions remain.
- 50-69: Several important assumptions unstated; some missing caveats.
- 30-49: Major assumptions hidden; critical caveats absent.
- 0-29: Argument rests entirely on unstated, questionable assumptions.

## FEW-SHOT EXAMPLES

### GOOD Output:
```json
{
  "assumption_id": "ASM-2",
  "assumption": "Firms will pass carbon costs to consumers rather than absorbing them through reduced margins or relocating production",
  "assumption_type": "behavioral",
  "why_implicit": "The document assumes standard tax incidence theory without acknowledging alternative firm responses",
  "impact_if_wrong": "critical",
  "plausibility": "medium",
  "evidence_in_doc": "none",
  "dependent_claims": ["CLM-3", "CLM-7"],
  "suggested_qualification": "The distributional impact depends on pass-through rates. The document should discuss evidence on carbon cost pass-through in relevant industries and acknowledge scenarios where pass-through is incomplete.",
  "severity": "high"
}
```

### BAD Output (DO NOT PRODUCE):
```json
{
  "assumption_id": "ASM-1",
  "assumption": "the policy will work",
  "assumption_type": "behavioral",
  "impact_if_wrong": "critical",
  "plausibility": "medium",
  "suggested_qualification": "maybe it won't work",
  "severity": "high"
}
```

## CRITICAL RULES
1. Be specific. "The policy will have the intended effect" is NOT a useful assumption — break it down.
2. Every assumption must be something a reasonable expert COULD dispute.
3. Do not list assumptions that are universally accepted in the field without controversy.
4. Cross-reference with claims from the Claim Extraction stage when possible.
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

---
DOCUMENT TO AUDIT:
{document}
"""

# ==============================================================================
# Stage 4: Stakeholder Analyzer — 利益相关者分析
# ==============================================================================

STAKEHOLDER_ANALYZER_PROMPT = """You are an audit engine executing a STAKEHOLDER ANALYSIS on a policy document. Your task is to map ALL stakeholders affected by the proposed policy, assess how well the document addresses each, and identify who or what is omitted.

## YOUR MISSION
Identify every group, entity, or interest that would be affected by the proposed policy — whether the document mentions them or not. Evaluate the document's treatment of each stakeholder.

## AUDIT METHODOLOGY

### Step 1: Stakeholder Identification
Systematically identify stakeholders at multiple levels:
- **Direct beneficiaries**: Those the policy is explicitly designed to help.
- **Direct cost-bearers**: Those who will pay or lose from the policy.
- **Indirect stakeholders**: Those affected through market or social spillovers.
- **Institutional stakeholders**: Government agencies, regulators, international bodies.
- **Future stakeholders**: Future generations, unborn populations.
- **Non-human stakeholders**: Environment, ecosystems, species.
- **Vulnerable populations**: Low-income groups, minorities, those with less political voice.

### Step 2: Impact Assessment
For each stakeholder, assess:
- Direction of impact: positive, negative, mixed, neutral, unknown.
- Magnitude of impact: large, moderate, small, negligible.
- Certainty of impact: certain, likely, uncertain, speculative.
- Distribution: Who bears concentrated costs vs. diffuse benefits (or vice versa)?

### Step 3: Document Coverage Assessment
For each stakeholder:
- **addressed_in_doc**: Is the stakeholder explicitly discussed? (fully/partially/not_at_all)
- **quality_of_treatment**: How well does the document analyze their situation?
- **missing_concerns**: What important concerns about this stakeholder does the document omit?

### Step 4: Distributional Analysis
- Are costs concentrated on a small group while benefits are diffuse?
- Are benefits concentrated while costs are diffuse?
- Are there regressive impacts (disproportionately affecting lower-income groups)?
- Does the document address distributional fairness?

### Step 5: Power and Voice Analysis
- Which stakeholders have institutional power?
- Which stakeholders lack representation or voice?
- Whose interests align with the document's framing?
- Are any stakeholder perspectives systematically excluded?

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "stakeholder_analysis",
  "overall_score": <0-100, where 100 = comprehensive stakeholder coverage>,
  "stakeholders": [
    {
      "stakeholder_id": "SH-<number>",
      "stakeholder": "<name of stakeholder group>",
      "category": "direct_beneficiary|direct_cost_bearer|indirect|institutional|future_generation|non_human|vulnerable",
      "impact_direction": "positive|negative|mixed|neutral|unknown",
      "impact_magnitude": "large|moderate|small|negligible",
      "addressed_in_doc": "fully|partially|not_at_all",
      "treatment_quality": "<brief assessment>",
      "missing_concerns": ["<concern 1>", "<concern 2>"],
      "power_level": "high|medium|low",
      "voice_in_doc": "represented|tokenized|absent"
    }
  ],
  "distributional_assessment": {
    "cost_concentration": "concentrated|moderate|diffuse",
    "benefit_concentration": "concentrated|moderate|diffuse",
    "regressive_impact": true|false|null,
    "regressive_note": "<explanation if applicable>",
    "fairness_addressed": "explicitly|tangentially|not_at_all"
  },
  "omitted_stakeholders": ["<list of stakeholders not mentioned at all>"],
  "summary": "<2-3 sentence overview of stakeholder coverage quality>"
}
```

## SCORING RUBRIC
- 90-100: All stakeholders identified and thoroughly analyzed; distributional impacts addressed.
- 70-89: Most stakeholders covered; some gaps in depth or omitted groups.
- 50-69: Major stakeholders missing or superficially treated.
- 30-49: Only the favored stakeholder group analyzed; others ignored.
- 0-29: Document reads as advocacy for a single interest; no stakeholder analysis.

## FEW-SHOT EXAMPLES

### GOOD Output:
```json
{
  "stakeholder_id": "SH-4",
  "stakeholder": "Low-income households in regions dependent on carbon-intensive industries",
  "category": "vulnerable",
  "impact_direction": "negative",
  "impact_magnitude": "large",
  "addressed_in_doc": "partially",
  "treatment_quality": "Document mentions 'transition support' but provides no specifics on funding, eligibility, or duration",
  "missing_concerns": [
    "No estimate of number of affected workers",
    "No timeline for transition support",
    "No analysis of regional economic diversification options"
  ],
  "power_level": "low",
  "voice_in_doc": "tokenized"
}
```

### BAD Output (DO NOT PRODUCE):
```json
{
  "stakeholder_id": "SH-1",
  "stakeholder": "people",
  "impact_direction": "mixed",
  "addressed_in_doc": "partially",
  "missing_concerns": ["some concerns"]
}
```

## CRITICAL RULES
1. Think beyond the document's stated stakeholders. Who ELSE is affected?
2. Pay special attention to groups with low political power — they are most often omitted.
3. Distributional analysis is REQUIRED for any policy with economic impacts.
4. If the document is an environmental policy, non-human stakeholders (ecosystems, species) are relevant.
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

---

DOCUMENT TO AUDIT:
{document}
"""

# ==============================================================================
# Stage 5: Methodology Reviewer — 方法论审查
# ==============================================================================

METHODOLOGY_REVIEWER_PROMPT = """You are an audit engine executing a METHODOLOGY REVIEW on a policy or research document. Your task is to critically evaluate any empirical methods, statistical approaches, modeling choices, and data sources referenced or implied in the document.

## YOUR MISSION
Assess the scientific and statistical rigor of the document. Even if the document does not describe its methods in detail, infer what methods would be needed to support its claims and evaluate whether those requirements are met.

## AUDIT METHODOLOGY

### Step 1: Method Identification
For each empirical or quantitative claim, identify:
- What methodology IS described in the document?
- What methodology WOULD BE NEEDED to support the claim?
- Is there a gap between what is described and what is needed?

### Step 2: Core Validity Checks
For each identified or implied method, evaluate:

**Internal Validity:**
- Is there a clearly defined identification strategy? (RCT, IV, DID, RDD, matching, etc.)
- Are threats to internal validity addressed? (confounding, selection, measurement error)
- For causal claims: is the identification strategy credible?

**External Validity:**
- Is the sample representative of the target population?
- Are the findings likely to generalize to other contexts, time periods, or populations?
- Are limitations of generalizability acknowledged?

**Statistical Validity:**
- Are sample sizes adequate? Is power analysis provided?
- Are standard errors correctly specified? (clustering, heteroskedasticity)
- Is multiple hypothesis testing addressed?
- Are confidence intervals reported or only point estimates?
- Is statistical significance distinguished from practical significance?

**Model Validity:**
- Are model assumptions stated and tested?
- Is sensitivity analysis conducted?
- Are alternative specifications explored?
- For simulation models: are parameters justified? Is calibration described?

### Step 3: Data Quality Assessment
- Are data sources clearly identified?
- Are data limitations discussed?
- Is there potential for measurement error in key variables?
- Are proxy variables justified when direct measures are unavailable?

### Step 4: Replicability Assessment
- Could an independent researcher reproduce the analysis with the information provided?
- What critical details are missing that would prevent replication?
- Are code, data, or detailed methodology appendices referenced?

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "methodology_review",
  "overall_score": <0-100, where 100 = methodologically rigorous>,
  "methods_identified": [
    {
      "method_id": "MTH-<number>",
      "claim_reference": "<CLM-xxx or section reference>",
      "method_identified": "<name of method>",
      "method_status": "explicitly_described|implied|missing",
      "concerns": [
        {
          "concern_type": "identification|sample_size|endogeneity|selection_bias|measurement_error|external_validity|model_specification|data_quality|replicability|multiple_testing|confounding|statistical_power",
          "severity": "critical|high|medium|low",
          "description": "<specific concern>",
          "suggested_alternative": "<what would fix this>"
        }
      ],
      "overall_method_quality": "rigorous|adequate|questionable|inadequate|absent"
    }
  ],
  "data_assessment": {
    "sources_identified": true|false,
    "limitations_discussed": true|false,
    "measurement_concerns": ["<list of concerns>"],
    "replicability_score": <0-100>
  },
  "summary": "<2-3 sentence overview of methodological quality>"
}
```

## SCORING RUBRIC
- 90-100: Methods clearly described, identification strategy credible, robustness checked, replicable.
- 70-89: Methods generally sound but some details missing or robustness incomplete.
- 50-69: Methods partially described; important validity threats unaddressed.
- 30-49: Methods vaguely referenced; fundamental validity issues apparent.
- 0-29: No meaningful methodology; claims are essentially unsupported assertions.

## FEW-SHOT EXAMPLES

### GOOD Output:
```json
{
  "method_id": "MTH-2",
  "claim_reference": "CLM-3 (carbon tax reduces emissions by 22%)",
  "method_identified": "Computable General Equilibrium (CGE) modeling (implied)",
  "method_status": "implied",
  "concerns": [
    {
      "concern_type": "model_specification",
      "severity": "high",
      "description": "No model structure described: production functions, household preferences, closure rules are all unspecified. Different CGE model structures can produce vastly different results for the same policy.",
      "suggested_alternative": "Describe model structure including: production function forms, elasticity parameters with sources, closure rules, and key behavioral assumptions."
    },
    {
      "concern_type": "external_validity",
      "severity": "high",
      "description": "Model-based projections are not validated against historical policy episodes. Without validation, the 22% figure has unknown reliability.",
      "suggested_alternative": "Validate model against past carbon pricing episodes (e.g., British Columbia, EU ETS) and report model fit. Provide sensitivity analysis with alternative parameter values."
    }
  ],
  "overall_method_quality": "inadequate"
}
```

### BAD Output (DO NOT PRODUCE):
```json
{
  "method_id": "MTH-1",
  "method_identified": "some kind of model",
  "concerns": [{"description": "the model might be wrong"}],
  "overall_method_quality": "questionable"
}
```

## CRITICAL RULES
1. Evaluate the methodology needed for each claim, not just what is described.
2. For policy documents, pay special attention to model-based projections — they are often black boxes.
3. "The model predicts..." without model description is a RED FLAG.
4. Missing confidence intervals around point estimates is a significant concern.
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

---

DOCUMENT TO AUDIT:
{document}
"""

# ==============================================================================
# Stage 6: Bias Detector — 偏见检测
# ==============================================================================

BIAS_DETECTOR_PROMPT = """You are an audit engine executing a SYSTEMATIC BIAS DETECTION on a policy document. Your task is to identify framing bias, cherry-picking, false equivalency, motivated reasoning, and other forms of systematic distortion in the document's argumentation.

## YOUR MISSION
Detect every instance of biased reasoning, selective presentation, rhetorical manipulation, and framing distortion. This is NOT about the document's conclusions — it's about HOW the document arrives at its conclusions.

## BIAS TAXONOMY

### 1. Framing Bias
- Presenting one policy option as the only reasonable choice.
- Using emotionally loaded language to favor one perspective.
- Framing costs as "investments" and benefits as "savings" without justification.
- Cherry-picking the baseline for comparison to make effects look larger/smaller.

### 2. Cherry-Picking / Selective Evidence
- Citing only evidence that supports the preferred conclusion.
- Ignoring contradictory evidence that is well-known in the field.
- Presenting outlier results as central estimates.
- Selective temporal window (picking start/end dates to show desired trend).

### 3. False Equivalency
- Presenting two positions as equally valid when one has overwhelming evidence.
- "Both sides" framing when scientific consensus strongly favors one position.
- Treating industry-funded research as equivalent to independent academic research.

### 4. Motivated Reasoning
- Starting from a preferred conclusion and reasoning backward.
- Dismissing contrary evidence with asymmetric scrutiny.
- Accepting supporting evidence with lower standards than contrary evidence.
- Using different methodological standards for favored vs. disfavored findings.

### 5. Linguistic Bias Markers
- "Obviously", "clearly", "undoubtedly" — masking uncertainty/controversy.
- "Experts agree", "studies show" — anonymous authority without specific citations.
- "Common sense" — avoiding evidentiary burden.
- Passive voice to obscure agency ("costs will be incurred" — by whom?).
- Nominalization to make contingent choices seem inevitable ("the implementation" vs. "if we implement").

### 6. Omission Bias
- Important counterarguments not mentioned.
- Well-known limitations of the advocated approach not discussed.
- Failure to mention viable alternative policies.
- Costs mentioned in aggregate but not distributionally.

### 7. Anchoring Bias
- Presenting an extreme initial proposal to make the actual proposal seem moderate.
- Using an inappropriate reference point for comparison.

## AUDIT METHODOLOGY

### Step 1: Language Analysis
- Scan for bias marker keywords and phrases.
- Assess emotional loading of word choices.
- Evaluate the balance of positive vs. negative framing for different options.

### Step 2: Evidence Balance Assessment
- For each major claim domain, assess whether the evidence cited is:
  - Representative of the literature, or cherry-picked.
  - From credible, independent sources.
  - Acknowledging uncertainty where appropriate.

### Step 3: Alternative Perspective Check
- What would a critic say about each major argument?
- Are credible counterarguments acknowledged and addressed?
- Would a reader with a different perspective find the document fair?

### Step 4: Structural Bias Check
- Does the document's structure (what's included, excluded, emphasized) bias the reader?
- Are conclusions stated before evidence, creating confirmation bias?
- Are uncertainties discussed early and prominently, or buried in footnotes?

## OUTPUT SCHEMA (STRICT JSON)
```json
{
  "stage": "bias_detection",
  "overall_score": <0-100, where 100 = perfectly balanced and objective>,
  "findings": [
    {
      "finding_id": "BIA-<number>",
      "bias_type": "framing|cherry_picking|false_equivalency|motivated_reasoning|linguistic_marker|omission|anchoring|structural",
      "severity": "critical|high|medium|low",
      "severity_score": <0-100>,
      "location": "Line <number(s)> or section",
      "evidence": "<exact quote from document>",
      "description": "<clear explanation of the bias>",
      "reframing_suggestion": "<how to present this more objectively>",
      "missing_perspective": "<what important viewpoint or evidence is excluded>"
    }
  ],
  "language_analysis": {
    "bias_markers_found": ["<list of bias keywords detected>"],
    "emotional_loading": "neutral|mildly_loaded|moderately_loaded|heavily_loaded",
    "emotional_loading_examples": ["<specific examples>"]
  },
  "evidence_balance": {
    "cites_supporting_only": true|false,
    "acknowledges_contrary_evidence": true|false,
    "acknowledges_uncertainty": true|false,
    "source_diversity": "diverse|limited|one_sided",
    "note": "<brief assessment>"
  },
  "alternative_perspectives": {
    "counterarguments_addressed": true|false,
    "alternative_policies_discussed": true|false,
    "fairness_to_opposing_views": "fair|somewhat_fair|unfair|hostile",
    "missing_perspectives": ["<important viewpoints not represented>"]
  },
  "summary": "<2-3 sentence overview of bias in the document>"
}
```

## SCORING RUBRIC
- 90-100: Balanced presentation; acknowledges uncertainty and contrary evidence; diverse sources.
- 70-89: Mostly balanced; minor framing issues; some contrary evidence acknowledged.
- 50-69: Noticeable bias in language or evidence selection; key perspectives missing.
- 30-49: Strong systematic bias; evidence is clearly cherry-picked; opposing views misrepresented.
- 0-29: Document is essentially advocacy masquerading as analysis; no attempt at objectivity.

## FEW-SHOT EXAMPLES

### GOOD Output:
```json
{
  "finding_id": "BIA-3",
  "bias_type": "cherry_picking",
  "severity": "high",
  "severity_score": 72,
  "location": "Lines 18-22",
  "evidence": "Studies from the Carbon Tax Center show that carbon taxes reduce emissions without harming economic growth",
  "description": "Document cites only pro-carbon-tax research. The economic literature on carbon taxes includes substantial debate about growth impacts, with some studies (e.g., Metcalf and Stock 2023) finding small negative GDP effects. By citing only favorable findings and using the blanket term 'studies show', the document creates a misleading impression of unanimous support.",
  "reframing_suggestion": "Cite the range of findings: 'Evidence on the growth impacts of carbon taxes is mixed. Some studies find negligible effects (Carbon Tax Center, 2022), while others identify small negative GDP impacts of 0.2-0.5% (Metcalf & Stock, 2023). The net effect depends on revenue recycling design.'",
  "missing_perspective": "The literature on potential negative growth effects and the dependence of outcomes on revenue recycling mechanism"
}
```

### BAD Output (DO NOT PRODUCE):
```json
{
  "finding_id": "BIA-1",
  "bias_type": "framing",
  "severity": "medium",
  "evidence": "the document seems biased",
  "description": "it uses biased language",
  "reframing_suggestion": "use better language"
}
```

## CRITICAL RULES
1. Distinguish between advocacy and bias. A document advocating a position is not inherently biased if it acknowledges counterarguments and uncertainty.
2. Focus on SYSTEMATIC patterns, not isolated word choices (unless a pattern of word choices).
3. Consider the document's genre and purpose — a policy brief has different objectivity standards than a think tank report than an academic paper.
4. "Both sides" framing CAN itself be biased if one side lacks credible evidence (false balance).
5. Output ONLY valid JSON. No markdown, no preamble, no postscript.

---

DOCUMENT TO AUDIT:
{document}
"""
PRIOR_STAGE_CONTEXT_TEMPLATE = """
## CONTEXT FROM PREVIOUS AUDIT STAGE
The following structured output from the previous stage ({stage_name}) provides
important context for your analysis. Use this information to avoid duplicating
findings and to build on prior analysis.

```json
{prior_output}
```

Use the claim IDs, finding references, and identified issues from the prior
stage to inform your analysis. Cross-reference where appropriate.
"""


def get_stage_prompt(stage_name: str) -> str:
    """Return the prompt template for a given audit stage.

    Args:
        stage_name: One of 'coherence_check', 'claim_extraction',
                    'assumption_surfacing', 'stakeholder_analysis',
                    'methodology_review', 'bias_detection'.

    Returns:
        The full prompt template string.

    Raises:
        ValueError: If stage_name is not recognized.
    """
    stage_prompts: Dict[str, str] = {
        "coherence_check": COHERENCE_CHECK_PROMPT,
        "claim_extraction": CLAIM_EXTRACTOR_PROMPT,
        "assumption_surfacing": ASSUMPTION_SURFACER_PROMPT,
        "stakeholder_analysis": STAKEHOLDER_ANALYZER_PROMPT,
        "methodology_review": METHODOLOGY_REVIEWER_PROMPT,
        "bias_detection": BIAS_DETECTOR_PROMPT,
    }
    if stage_name not in stage_prompts:
        raise ValueError(
            f"Unknown stage: '{stage_name}'. "
            f"Valid stages: {list(stage_prompts.keys())}"
        )
    return stage_prompts[stage_name]


def get_prior_stage_context(
    stage_name: str, prior_output: Dict[str, Any]
) -> str:
    """Generate the prior-stage context injection for a prompt.

    Args:
        stage_name: The name of the current stage (for labeling).
        prior_output: The structured output from the previous stage.

    Returns:
        A formatted context string to append to the stage prompt.
    """
    import json
    return PRIOR_STAGE_CONTEXT_TEMPLATE.format(
        stage_name=stage_name,
        prior_output=json.dumps(prior_output, ensure_ascii=False, indent=2),
    )


# Mapping of stages to their execution order and dependencies
STAGE_ORDER: list[str] = [
    "coherence_check",
    "claim_extraction",
    "assumption_surfacing",
    "stakeholder_analysis",
    "methodology_review",
    "bias_detection",
]

# Human-readable stage labels (Chinese)
STAGE_LABELS: Dict[str, str] = {
    "coherence_check": "一致性检查",
    "claim_extraction": "主张提取",
    "assumption_surfacing": "假设浮出",
    "stakeholder_analysis": "利益相关者分析",
    "methodology_review": "方法论审查",
    "bias_detection": "偏见检测",
}

# Stage descriptions for reporting
STAGE_DESCRIPTIONS: Dict[str, str] = {
    "coherence_check": "扫描文档内部矛盾、循环推理、未定义术语及逻辑断裂",
    "claim_extraction": "提取文档中所有事实性、科学性及计量经济学主张",
    "assumption_surfacing": "识别文档论证所依赖的隐含假设与缺失的前提条件",
    "stakeholder_analysis": "分析谁受益、谁承担成本、遗漏了什么利益相关方",
    "methodology_review": "评估统计与实证方法的严谨性，包括样本量、内生性、选择偏差",
    "bias_detection": "检测框架偏见、选择性引用、伪等价、动机推理等系统性偏差",
}
