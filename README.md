# 遥感任务智能助手 MVP

详细实现讲解：

- [RAG、Memory 与长程变化检测任务实现详解](docs/rag_memory_change_detection_long_task_implementation.md)

这是根据 `remote_sensing_agent_tech_design.md` 第 20 节实现的最小闭环原型。当前版本聚焦双时相变化检测，使用本地 JSON 文件代替数据库、对象存储和向量库，但保留了后续工程化替换的接口边界。

## 已实现闭环

1. 选择两期影像 URI 或已登记资产。
2. 用自然语言创建变化检测任务。
3. Agent 通过 `raster.inspect_metadata` 读取元数据。
4. 本地 RAG 检索变化检测 SOP、模型卡和工具说明。
5. 生成结构化计划，并支持人工确认。
6. 自动执行对齐、NDBI、变化检测、后处理、矢量化、统计、预览和报告。
7. 每个阶段写入 event 和 checkpoint。
8. 输出 PNG 预览、GeoJSON 图斑、CSV 面积统计和 Markdown 报告。
9. 支持用户反馈。
10. 将任务摘要和反馈写入项目记忆。

## 快速运行

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m rs_agent.demo --auto-confirm
```

启动 API：

```powershell
.\.venv\Scripts\uvicorn rs_agent.api.main:app --reload
```

启动 React 前端开发服务器：

```powershell
cd frontend
npm install
npm run dev
```

打开 React 工作台：

```text
http://127.0.0.1:5173/
```

构建后也可以由 FastAPI 直接托管：

```powershell
cd frontend
npm run build
```

```text
http://127.0.0.1:8000/
```

创建 demo 任务：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/tasks `
  -ContentType "application/json" `
  -Body '{"user_goal":"帮我对两期 Sentinel-2 影像做建设用地扩张变化检测，输出图斑、面积统计和报告。","image_t1_uri":"demo://image_t1","image_t2_uri":"demo://image_t2","auto_confirm":true}'
```

默认数据会写入 `.rs_agent_data/`。端到端 demo 使用内置的 `demo://image_t1` 和 `demo://image_t2`，方便在没有真实 GeoTIFF 的情况下验证闭环。

第二阶段已引入 `rasterio`，真实 GeoTIFF/COG 输入会读取 CRS、transform、bbox、分辨率、波段描述等元数据；带地理参考的中间/结果栅格会优先输出 GeoTIFF。

## 长程 Agent 模式

长程模式使用真实大模型选择受约束的遥感 workflow，并通过 checkpoint 分批执行。
模型只能选择已注册步骤和调整允许的参数，不能执行任意代码或绕过工具注册表。

配置 OpenAI 兼容接口：

```powershell
$env:RS_AGENT_LLM_API_KEY="your-api-key"
$env:RS_AGENT_LLM_MODEL="your-model-name"
$env:RS_AGENT_LLM_BASE_URL="https://api.openai.com/v1"
```

DeepSeek V4 Pro 使用：

```powershell
$env:RS_AGENT_LLM_API_KEY="your-api-key"
$env:RS_AGENT_LLM_MODEL="deepseek-v4-pro"
$env:RS_AGENT_LLM_BASE_URL="https://api.deepseek.com"
$env:RS_AGENT_LLM_RESPONSE_FORMAT="json_object"
```

配置后重启 API。创建 Agent 任务时设置 `agent_mode`，并用
`execution_budget` 限制每次请求最多执行的工具步骤：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/tasks `
  -ContentType "application/json" `
  -Body '{"user_goal":"分析两期影像的建设用地扩张并生成报告","image_t1_uri":"demo://image_t1","image_t2_uri":"demo://image_t2","agent_mode":"agent","execution_budget":3,"auto_confirm":true}'
```

任务达到执行预算后进入 `paused`，调用恢复接口继续下一批步骤：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/tasks/{task_id}/resume
```

规划或工具执行失败后可重试最近失败阶段：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/tasks/{task_id}/retry
```

工作台也提供 Agent/Workflow 模式选择、每轮步骤预算、继续执行和失败重试按钮。

## BIT 深度变化检测工具

项目内置 BIT_LEVIR 预训练模型，并封装为 Agent 工具：

```text
ml.bit_change_detection
```

当任务目标明确包含 `BIT`、`Transformer`、`深度模型` 或 `深度变化检测` 时，
规划器会在变化检测节点选择该工具。工具要求两期影像已经配准，自动选择 RGB 波段，
使用 256 像素切片和重叠融合进行推理，并输出保留原始 CRS、transform 和 bbox 的变化栅格。

示例目标：

```text
请使用 BIT Transformer 深度模型分析两期影像的建筑物变化，输出图斑、面积统计和报告。
```

模型来源代码的许可证限定非商业和科研用途，商业部署前需要获得原作者许可。

## 长程变化检测任务

变化检测任务现在采用 13 阶段长程模板：

```text
元数据读取
→ 输入适用性质量门
→ 双时相对齐
→ 配准质量门
→ 两期特征计算
→ 规则模型或 BIT 推理
→ 结果合理性质量门
→ 小图斑过滤
→ 矢量化
→ 面积统计
→ 预览图
→ 报告
```

新增 Agent 工具：

- `quality.assess_change_inputs`：检查尺寸、CRS、分辨率、波段数和有效像元。
- `quality.assess_alignment`：检查网格一致性和双时相影像相关性。
- `quality.assess_change_result`：检查变化比例、连通图斑数量和掩膜类型。

任一质量门不通过时，任务进入 `waiting_human` 并生成
`quality_gate_review` 中断。批准后可以继续执行，所有判断、问题和人工决定都会写入事件与 checkpoint。

## 生产级 RAG 与长期记忆

知识和长期记忆使用 SQLite 数据库：

```text
.rs_agent_data/knowledge_memory.sqlite3
```

数据库启用 WAL、外键、索引和 FTS5。知识检索综合全文 BM25、向量相似度和任务标签；
记忆召回还会综合用户、项目、置信度、重要性、更新时间和访问次数。

支持的知识文件为 Markdown、TXT 和 JSON，单文件最大 10 MB。上传文档：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/knowledge/documents/upload?source_type=project_sop&version=1&task_tags=change_detection,quality" `
  -F "file=@path/to/document.md"
```

知识管理接口：

```text
GET    /api/knowledge/documents
POST   /api/knowledge/documents/upload
DELETE /api/knowledge/documents/{document_id}
POST   /api/knowledge/search
```

记忆管理接口：

```text
GET    /api/memories
POST   /api/memories/search
DELETE /api/memories/{memory_id}
POST   /api/memories/purge-expired
```

旧 `.rs_agent_data/memories/memories.jsonl` 会在首次启动时自动迁移，原文件保留。
相同记忆重复写入时不会产生重复记录，而会提高置信度并记录强化次数。
质量门人工放行、任务摘要和用户反馈都会形成项目级长期记忆。

默认向量器为无需网络的 `hashing-charword-v1`，用于保证离线可用。生产环境可配置
OpenAI 兼容的 Embedding 服务：

```powershell
$env:RS_AGENT_EMBEDDING_API_KEY="..."
$env:RS_AGENT_EMBEDDING_MODEL="your-embedding-model"
$env:RS_AGENT_EMBEDDING_BASE_URL="https://your-provider.example/v1"
```

配置变化后需要重新摄取文档，确保数据库中的向量维度与当前 Embedding 模型一致。

未配置模型时，原有 `workflow` 模式仍可正常使用；请求 `agent` 模式会明确返回配置错误。

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```
