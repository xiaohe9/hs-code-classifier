# hs-code-classifier

> 跨境电商物流 · 智能商品归类（HS Code）Agent
> RAG + ReAct Agent + 三层防幻觉 + 评测体系 ｜ 技术验证项目

[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20ReAct-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

## 项目背景

跨境物流报关中，商品归类（HS Code 归类）错误会导致海关查验、罚款、扣货甚至企业信用降级。
人工归类一票需 5-15 分钟且依赖个人经验；而直接问大模型会产生**无依据的幻觉编码**——这是零容忍场景。

本项目验证一套可落地的技术方案：**检索税则原文（RAG）+ Agent 主动追问澄清 + 确定性规则护栏 + 全链路评测**，
输出"编码 + 置信度 + 可复核依据链"，低置信度自动转人工，而不是硬答。

> 作者是国际物流行业在役 AI 工程师（生产系统连续运行 54 天零故障），本项目为针对智能报关场景的
> 工程化技术验证，架构复用自已开源的生产级项目 [flight-mcp-server](https://github.com/xiaohe9/flight-mcp-server)。

## 核心能力

- 🔍 **税则 RAG 检索**：bge-m3 向量化 + 向量/BM25 混合检索 + 重排序，条文按"品目-类章注"天然结构切分
- 🤖 **ReAct Agent 归类决策**：信息不完备时**主动追问**（追问由候选编码的区分特征驱动，非通用寒暄）
- 🛡️ **三层防幻觉**：禁限品正则硬拦截 → RAG 交叉验证（结论必须映射回条文原文）→ 置信度阈值兜底转人工
- 📏 **评测体系**：50 组标注测试集，Top-1/Top-3 准确率、防幻觉拦截率、P95 延迟，CI 自动回归
- 🐳 **开箱即用**：Docker Compose 一键部署，本地模型（Ollama）零 API 成本运行

## 系统架构

```
商品描述（自然语言）
      │
      ▼
┌──────────────────────────────┐
│  LangGraph ReAct Agent       │  ← 信息完备性检查 → 主动追问澄清
│  （工作记忆 + SQLite 归类历史）│
└──────────────────────────────┘
      │ MCP Tool 调用（JSON-RPC 2.0）
      ▼
┌────────────┬─────────────────┬──────────────┐
│ 税则检索    │ 归类规则引擎      │ 依据链生成     │
│ ChromaDB   │ 禁限品拦截/总规则 │ 结构化输出     │
│ 混合检索    │ 决策树(确定性)   │ +编码存在性校验│
└────────────┴─────────────────┴──────────────┘
      │
      ▼
输出：{ hs_code, 置信度, 条文原文, 类章注引用, 总规则依据, 备选编码, 人工复核标记 }
```

**关键设计决策**：能走确定性规则的绝不让 LLM 猜（禁限品识别、编码校验）；
LLM 只做它擅长的——商品特征与品目条文的语义匹配；所有结论必须挂可复核的条文原文。

## 快速开始

```bash
# 1. 克隆 & 安装
git clone https://github.com/xiaohe9/hs-code-classifier.git
cd hs-code-classifier
pip install -r requirements.txt

# 2. 启动本地模型（Ollama）
ollama pull qwen3:4b && ollama pull bge-m3

# 3. 初始化税则知识库（内置示例章节数据）
python scripts/build_taxonomy_kb.py

# 4. 启动服务
docker compose up -d   # 或 uvicorn app.main:app --port 8000
```

```bash
# 示例调用
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"description": "圣诞节用LED灯串，低压24V，装饰用"}'
```

<details>
<summary>返回示例</summary>

```json
{
  "hs_code": "9405.42.00",
  "confidence": 0.87,
  "basis": {
    "heading_text": "品目94.05 条文原文……",
    "chapter_note": "第94章章注引用……",
    "gir_rule": "归类总规则一"
  },
  "alternatives": ["8539.52.00"],
  "needs_human_review": false,
  "latency_ms": 1830
}
```
</details>

## 评测指标

> 测试集：50 组"商品描述 → 标准 HS Code"标注样本（公开归类案例整理），CI 自动回归

| 指标 | 结果 |
|---|---|
| Top-1 准确率（8位税号） | _待 v0.4 评测后填入_ |
| Top-3 命中率 | _待填_ |
| 品目级（前4位）准确率 | _待填_ |
| 防幻觉拦截率（对抗样本） | _待填_ |
| 平均响应时间 | _待填_ |

## 技术栈

LangGraph · MCP · ChromaDB · bge-m3 · Ollama(qwen3:4b) · FastAPI · SQLite · Docker · pytest · GitHub Actions

## 项目结构

```
hs-code-classifier/
├── app/
│   ├── agent/          # LangGraph ReAct Agent（规划/追问/反思）
│   ├── mcp_server/     # MCP Tool 封装（税则检索/规则引擎/依据链）
│   ├── rag/            # 切分/embedding/混合检索/重排序
│   ├── guardrails/     # 三层防幻觉
│   └── main.py         # FastAPI 入口
├── data/taxonomy/      # 税则示例章节（公开文本节选）
├── tests/eval/         # 50组标注测试集 + 评测脚本
├── docs/架构设计.md     # 完整架构决策记录
├── docker-compose.yml
└── README.md
```

## 路线图

- [x] v0.1 架构设计 + 仓库骨架
- [ ] v0.2 RAG 链路（税则入库 + 混合检索）
- [ ] v0.3 Agent 闭环（追问机制 + 防幻觉 + 结构化输出）
- [ ] v0.4 评测体系（50组测试集 + CI 回归）
- [ ] v1.0 Docker 交付 + SSE 推理过程可视化演示页

## 合规声明

本项目仅使用公开税则文本与自编测试样例，归类结果定位为**报关员辅助工具**，
输出始终附带依据链与置信度，最终归类责任在持证报关员。

## License

MIT © 陈晓河
