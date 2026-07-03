# AuditAgent — 结构化文档审计引擎

**不是通用聊天机器人，而是遵循严格多阶段方法论的审计系统。**

AuditAgent 是一个基于 LLM 的文档审计引擎，专为经济政策文件、研究报告、制度文档等严肃文本设计。它通过 6 个固定审计阶段，对文档进行系统性、可量化的质量评估。

---

## 核心理念

通用"AI 总结"工具的问题在于：输出不可控、评判标准模糊、无法复现。AuditAgent 采用**固定方法论管线**——每个阶段有明确的审计目标、评分标准和结构化输出格式。

```
                    +-------------------+
                    |   原始文档输入     |
                    +--------+----------+
                             |
                             v
              +--------------+--------------+
              |       AuditAgent.core       |
              |   (主审计管线 / 编排引擎)    |
              +--------------+--------------+
                             |
          +------------------+------------------+
          |      |      |      |      |      |
          v      v      v      v      v      v
    +---------+ +--------+ +----------+ +------------+ +--------------+ +----------+
    |Stage 1  | |Stage 2 | |Stage 3   | |Stage 4     | |Stage 5       | |Stage 6  |
    |一致性   | |主张    | |假设      | |利益相关者  | |方法论        | |偏见     |
    |检查     | |提取    | |浮出      | |分析        | |审查          | |检测     |
    +---------+ +--------+ +----------+ +------------+ +--------------+ +----------+
          |      |      |      |      |      |
          +------+------+------+------+------+
                             |
                             v
              +--------------+--------------+
              |     AuditScorer (评分)      |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |    Reporter (报告生成)      |
              |  Text Report / HTML / JSON  |
              +-----------------------------+
```

---

## 六个审计阶段

| 阶段 | 名称 | 中文标签 | 审计目标 |
|------|------|----------|----------|
| 1 | CoherenceCheck | 一致性检查 | 内部矛盾、循环推理、未定义术语 |
| 2 | ClaimExtractor | 主张提取 | 所有事实/科学/计量经济学主张 |
| 3 | AssumptionSurfacer | 假设浮出 | 隐含假设与缺失的前提条件 |
| 4 | StakeholderAnalyzer | 利益相关者分析 | 谁受益？谁承担成本？遗漏了什么？ |
| 5 | MethodologyReviewer | 方法论审查 | 统计方法、样本量、内生性、外部有效性 |
| 6 | BiasDetector | 偏见检测 | 框架偏见、选择性引用、伪等价、动机推理 |

---

## 评分体系

| 维度 | 权重 | 说明 |
|------|------|------|
| 一致性 (Coherence) | 20% | 逻辑自洽程度 |
| 方法论 (Methodology) | 20% | 实证方法严谨性 |
| 主张 (Claims) | 15% | 主张可验证性 |
| 假设 (Assumptions) | 15% | 假设透明度 |
| 利益相关者 (Stakeholders) | 15% | 利益分析均衡性 |
| 偏见 (Bias) | 15% | 偏见控制程度 |

最终得分 0-100，70 分以上为通过。

---

## 快速开始

### 安装

```bash
cd audit-agent
pip install -r requirements.txt
```

### 使用本地 Ollama

```bash
# 确保 Ollama 正在运行
ollama serve

# 拉取推荐模型
ollama pull qwen2.5:7b

# 运行审计
python examples/run_audit.py
```

### 使用 OpenAI 兼容 API

```python
from audit_agent import AuditAgent

agent = AuditAgent(
    model="api",
    api_base="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o"
)

result = agent.audit("path/to/document.txt")
print(result.executive_summary())
result.export_html("output/report.html")
```

### 命令行使用

```bash
python -m audit_agent.core --input policy.txt --output report.html --verbose
```

---

## 项目结构

```
audit-agent/
├── README.md
├── requirements.txt
├── audit_agent/
│   ├── __init__.py
│   ├── core.py              # 主审计管线
│   ├── stages.py            # 6 个审计阶段
│   ├── prompts.py           # 结构化提示模板
│   ├── scoring.py           # 定量审计评分
│   └── reporter.py          # 报告生成（文本/HTML/JSON）
├── config/
│   └── audit_rules.yaml     # 可配置审计规则
├── examples/
│   ├── sample_policy.txt    # 示例碳税政策文档
│   └── run_audit.py         # 演示脚本
└── tests/
    └── test_audit.py        # 单元测试
```

---

## 配置说明

编辑 `config/audit_rules.yaml` 可自定义：

- **评分权重**：调整六个维度的权重分配
- **偏见关键词**：添加领域特定的偏见信号词
- **严重度阈值**：自定义 critical/high/medium/low 的分数区间
- **领域模式**：为经济学、医疗、法律等不同领域配置模式规则

---

## 技术架构

- **LLM 后端**：使用 OpenAI 兼容 API，支持任何兼容端点（OpenAI、Ollama、vLLM 等）
- **温度设置**：所有 LLM 调用使用 `temperature=0.0`，确保确定性输出
- **严格 JSON 输出**：所有阶段提示要求 `response_format: {"type": "json_object"}`
- **无状态设计**：每次审计独立运行，不依赖外部数据库

---

## License

MIT
